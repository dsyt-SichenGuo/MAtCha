"""Memory-bounded helpers for generated-image dense training.

Copyright (C) 2023, Inria / GRAPHDECO components are imported under the
2d-gaussian-splatting/LICENSE.md terms (non-commercial research/evaluation).
"""
from collections import OrderedDict
import gc
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2d-gaussian-splatting"))


class ViewSampler:
    def __init__(self, chart_count, dense_count, seed):
        self.rng = random.Random(seed)
        self.counts = {"chart": chart_count, "dense": dense_count}
        self.stacks = {"chart": [], "dense": []}

    def next(self, iteration, chart_every):
        kind = "chart" if iteration % chart_every == 0 else "dense"
        if not self.stacks[kind]:
            self.stacks[kind] = list(range(self.counts[kind]))
            self.rng.shuffle(self.stacks[kind])
        return kind, self.stacks[kind].pop()

    def state_dict(self):
        return dict(rng=self.rng.getstate(), stacks={k:list(v) for k,v in self.stacks.items()})

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.stacks = {k:list(v) for k,v in state["stacks"].items()}


def tensor_tree(value, device):
    import torch
    if torch.is_tensor(value):
        moved=value.detach().to(device)
        if value.requires_grad:
            moved.requires_grad_(True)
        return moved
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree(v, device) for v in value)
    return value


def load_cameras(scene_path, model_args):
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
    info = readColmapSceneInfo(str(scene_path), "images", False)
    cameras = cameraList_from_camInfos(info.train_cameras, 1., model_args)
    camera_json = [camera_to_JSON(i, c) for i, c in enumerate(info.train_cameras)]
    return cameras, info.point_cloud, info.nerf_normalization["radius"], camera_json


def chart_valid_masks(data, far):
    scale = float(data["scale_factor"])
    depth, prior = data["depths"] / scale, data["prior_depths"] / scale
    geometry = np.isfinite(depth) & (depth > .01) & (depth < far) & np.isfinite(data["pts"]).all(-1)
    prior_valid = np.isfinite(prior) & (prior > .01) & (prior < far)
    return geometry, prior_valid


def initialize_all(gaussians, charts, chart_cameras, point_cloud, extent, valid):
    """All upstream-valid chart faces + all dense points; only stream memory."""
    import torch
    from matcha.dm_scene.charts import get_gaussian_parameters_from_charts_data
    from simple_knn._C import distCUDA2
    params = {k: [] for k in ("means", "scales", "quaternions", "colors")}
    chart_counts = []
    h, w = charts["depths"].shape[-2:]
    with torch.no_grad():
        for i, cam in enumerate(chart_cameras):
            local = {k: torch.as_tensor(v if k == "scale_factor" else v[i:i+1], device="cuda")
                     for k, v in charts.items()}
            local["confs"] = local["confs"].clone()
            local["confs"][~torch.as_tensor(valid[i:i+1], device="cuda")] = -2.
            image = torch.nn.functional.interpolate(cam.original_image[None].cuda(), (h, w),
                         mode="bilinear", align_corners=False, antialias=True)[0].permute(1, 2, 0)
            result = get_gaussian_parameters_from_charts_data(local, [image], conf_th=-1.,
                         ratio_th=5., normal_scale=1e-10, normalized_scales=.5)
            count = len(result["means"])
            chart_counts.append(count)
            for key in params:
                a = result[key][..., :2] if key == "scales" else result[key]
                if not torch.isfinite(a).all():
                    raise FloatingPointError(f"Nonfinite chart initialization: {cam.image_name} {key}")
                params[key].append(a.cpu())
            print(f"[init] Chart {i+1}/{len(chart_cameras)}: {count:,} valid surfels", flush=True)
            del result, local, image
        for key in params:
            params[key] = torch.cat(params[key])
        xyz = torch.as_tensor(np.asarray(point_cloud.points), dtype=torch.float32, device="cuda")
        dense_count = len(xyz)
        if not torch.isfinite(xyz).all():
            raise ValueError("Nonfinite dense point cloud")
        # Same nearest-neighbor scale rule as GaussianModel.create_from_pcd.
        scale = torch.sqrt(torch.clamp_min(distCUDA2(xyz), 1e-7))[:, None].repeat(1, 2).cpu()
        params["means"] = torch.cat([params["means"], xyz.cpu()])
        params["scales"] = torch.cat([params["scales"], scale])
        params["quaternions"] = torch.cat([params["quaternions"], torch.rand(dense_count, 4)])
        params["colors"] = torch.cat([params["colors"], torch.as_tensor(np.asarray(point_cloud.colors), dtype=torch.float32)])
        del xyz, scale
        gc.collect()
        torch.cuda.empty_cache()
        gaussians.create_from_parameters(params["means"].cuda(), params["scales"].cuda(),
                                         params["quaternions"].cuda(), params["colors"].cuda(), extent)
    return dict(chart_gaussians=sum(chart_counts), per_chart=chart_counts,
                dense_gaussians=dense_count, total=sum(chart_counts)+dense_count, point_budget=None)


def compute_mip_chunked(gaussians, cameras, chunk_size=500_000):
    """Original mip rule with bounded projection temporaries, not point removal."""
    import torch
    with torch.no_grad():
        output = torch.empty((len(gaussians.get_xyz), 1), device="cuda")
        focal = max(c.focal_x for c in cameras)
        max_seen = None
        unseen = []
        for start in range(0, len(output), chunk_size):
            xyz = gaussians.get_xyz[start:start+chunk_size]
            distance = torch.full((len(xyz),), 100000., device="cuda")
            seen = torch.zeros(len(xyz), dtype=torch.bool, device="cuda")
            for cam in cameras:
                view = xyz @ torch.as_tensor(cam.R, device="cuda", dtype=torch.float32) + torch.as_tensor(cam.T, device="cuda", dtype=torch.float32)
                z = view[:, 2]
                x = view[:, 0] / z.clamp_min(.001) * cam.focal_x + cam.image_width/2
                y = view[:, 1] / z.clamp_min(.001) * cam.focal_y + cam.image_height/2
                good = (z>.2) & (x>=-.15*cam.image_width) & (x<=1.15*cam.image_width) & (y>=-.15*cam.image_height) & (y<=1.15*cam.image_height)
                distance[good] = torch.minimum(distance[good], z[good])
                seen |= good
            if seen.any():
                local_max = distance[seen].max()
                max_seen = local_max if max_seen is None else torch.maximum(max_seen, local_max)
            output[start:start+len(xyz), 0] = distance
            unseen.append(~seen)
        if max_seen is None:
            raise ValueError("No initialization points are visible in training cameras")
        invalid = torch.cat(unseen)
        output[invalid, 0] = max_seen
        gaussians.mip_filter = output / focal * (.2**.5)


class ChartPriorCache:
    def __init__(self, charts, cameras, valid, prior_valid):
        self.charts, self.cameras = charts, cameras
        self.valid, self.prior_valid = valid, prior_valid
        self.cache = OrderedDict()

    def get(self, index):
        import torch
        import torch.nn.functional as F
        from matcha.dm_scene.charts import build_priors_from_charts_data
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        local = {k: torch.as_tensor(v if k=="scale_factor" else v[index:index+1], device="cuda")
                 for k,v in self.charts.items()}
        # Fill invalid depths only for stable finite derivative calculation;
        # supervision masks exclude these pixels and their derivative neighbors.
        for key, mask in (("depths", self.valid[index]), ("prior_depths", self.prior_valid[index])):
            t = local[key]
            good = torch.as_tensor(mask[None], device="cuda")
            if not good.any():
                raise ValueError(f"No valid {key}: chart {index}")
            local[key] = torch.where(good, t, t[good].median())
        result = build_priors_from_charts_data(local, [self.cameras[index]])
        result = {k: v[0] if k!="scale_factor" else v for k,v in result.items()}
        size = (self.cameras[index].image_height, self.cameras[index].image_width)
        for key, a in (("valid", self.valid[index]), ("prior_valid", self.prior_valid[index])):
            up = F.interpolate(torch.tensor(a[None, None].astype(np.float32), device="cuda"), size=size, mode="bilinear", align_corners=False)
            result[key] = (up[0]>.999).float()
        both = result["valid"] * result["prior_valid"]
        result["derivative_valid"] = -F.max_pool2d(-both[None], 3, 1, 1)[0]
        self.cache[index] = result
        if len(self.cache)>2:
            self.cache.popitem(last=False)
        return result


def masked_mean(value, mask):
    # Keep the upstream full-image denominator; invalid pixels contribute zero.
    return (value*mask).mean()


def chart_regularization(pkg, prior, gaussians, iteration):
    import torch
    from matcha.dm_scene.charts import schedule_regularization_factor_2
    from matcha.dm_utils.rendering import normal2curv
    factor = schedule_regularization_factor_2(iteration, .5)
    conf = prior["confs"]
    depth = factor*.75 * masked_mean(.5*conf*torch.log1p(prior["scale_factor"]*(prior["depths"]-pkg["surf_depth"]).abs()), prior["valid"])
    depth = depth + factor*.5 * masked_mean(torch.exp(-(conf-1.)**2/2) * (1.-(pkg["surf_normal"]*prior["normals"]).sum(0)), prior["derivative_valid"])
    normal = factor*.5 * masked_mean(1.-(pkg["rend_normal"]*prior["normals"]).sum(0), prior["derivative_valid"])
    curvature = normal2curv(pkg["rend_normal"], torch.ones_like(pkg["rend_normal"][0:1]))
    curv = factor*.25 * masked_mean((prior["curvs"]-curvature).abs(), prior["derivative_valid"])
    scales = gaussians.get_scaling
    anisotropy = .1 * (torch.clamp_min(scales.max(1).values/scales.min(1).values, 5.)-5.).mean()
    return depth+normal+curv+anisotropy


def save_checkpoint(path, gaussians, iteration, sampler, identity):
    import torch
    # CPU serialization avoids allocating a second GPU-sized optimizer state.
    state = dict(schema="matcha_generated_dense_checkpoint_v1", iteration=iteration,
                 model=tensor_tree(gaussians.capture(), "cpu"),
                 mip_filter=gaussians.mip_filter.cpu() if gaussians.use_mip_filter else None,
                 use_mip_filter=gaussians.use_mip_filter, sampler=sampler.state_dict(), identity=identity,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix(".pending.pth")
    torch.save(state, temporary)
    temporary.replace(path)


def save_ply_streaming(path, gaussians, chunk_size=100_000):
    """Same PLY fields as GaussianModel.save_ply, without millions of tuples."""
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".pending.ply")
    attributes=gaussians.construct_list_of_attributes()
    count=len(gaussians.get_xyz)
    header="ply\nformat binary_little_endian 1.0\n"+f"element vertex {count}\n"
    header+="".join(f"property float {name}\n" for name in attributes)+"end_header\n"
    with temporary.open("wb") as f,torch.no_grad():
        f.write(header.encode())
        for start in range(0,count,chunk_size):
            end=min(start+chunk_size,count)
            xyz=gaussians._xyz[start:end].detach().cpu().numpy()
            columns=[xyz,np.zeros_like(xyz),
                gaussians._features_dc[start:end].detach().transpose(1,2).flatten(1).cpu().numpy(),
                gaussians._features_rest[start:end].detach().transpose(1,2).flatten(1).cpu().numpy(),
                gaussians._opacity[start:end].detach().cpu().numpy(),
                gaussians._scaling[start:end].detach().cpu().numpy(),
                gaussians._rotation[start:end].detach().cpu().numpy()]
            if gaussians.use_mip_filter:
                columns.append(gaussians.mip_filter[start:end].detach().cpu().numpy())
            matrix=np.concatenate(columns,axis=1).astype("<f4",copy=False)
            assert matrix.shape[1]==len(attributes)
            matrix.tofile(f)
    temporary.replace(path)
