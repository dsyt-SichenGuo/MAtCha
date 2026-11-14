"""Stage 3: train selected charts plus every SfM image; arbitrary scene/view counts.

Copyright (C) 2023, Inria / GRAPHDECO components are imported under the
2d-gaussian-splatting/LICENSE.md terms (non-commercial research/evaluation).
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2d-gaussian-splatting"))


from dense_training_utils import (
    ViewSampler, tensor_tree, load_cameras, chart_valid_masks,
    initialize_all, compute_mip_chunked, ChartPriorCache, chart_regularization,
    save_checkpoint, save_ply_streaming,
)

def main():
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/free_gaussians_refinement/generated_dense.yaml")
    selected, _ = pre.parse_known_args()
    config = yaml.safe_load(Path(selected.config).resolve().read_text())
    parser = argparse.ArgumentParser(parents=[pre], description=__doc__)
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument("--sfm_scene", type=Path, required=True)
    parser.add_argument("--chart_scene", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--validate_only", action="store_true", help="Validate prepared data/config without loading CUDA models")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Run 40 steps, including two scheduled chart updates, PLY and checkpoint saving")
    defaults = {**config["model"], **config["optimization"], **config["pipeline"]}
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.model_path = str(args.output_path.resolve())
    args.source_path = str(args.chart_scene.resolve()/"dense_scene")
    from dense_scene_io import metadata_identity
    metadata = json.loads((args.sfm_scene.resolve()/"cameras.json").read_text())
    if args.smoke_test:
        args.iterations=40
    if args.eval:
        parser.error("All images are used for training; --eval is not supported by this entry")
    if args.data_device not in ("cpu","cuda") or args.resolution!=1:
        parser.error("Use --data_device cpu or cuda and full-resolution images (resolution=1)")
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    torch.set_num_threads(8)
    dataset, opt, pipe = mp.extract(args), op.extract(args), pp.extract(args)
    output = Path(dataset.model_path)
    if not args.resume and ((output/"point_cloud").exists() or (output/"run_config.json").exists()):
        raise FileExistsError(f"Existing training output: {output}; choose --output_path or --resume")
    chart_path, dense_path = args.chart_scene.resolve(), Path(dataset.source_path)
    chart_manifest = json.loads((chart_path/"scene/manifest.json").read_text())
    dense_manifest = json.loads((dense_path/"manifest.json").read_text())
    if not (chart_manifest["source_identity"] == dense_manifest["source_identity"] == metadata_identity(metadata)):
        raise ValueError("Charts, dense cameras and supplied SfM coordinate sources differ")
    expected_names = {Path(p).name for p in metadata["filepaths"]}
    if expected_names != {e["name"] for e in dense_manifest["frames"]}:
        raise ValueError("Dense scene must contain ALL SfM views")
    source_types = {Path(e["name"]).stem:e["source_type"] for e in dense_manifest["frames"]}
    with np.load(chart_path/"charts_data.npz", allow_pickle=False) as data:
        charts = {k:data[k].copy() for k in data.files}
    chart_names = [Path(e["name"]).stem for e in chart_manifest["frames"]]
    if len(chart_names)!=len(charts["depths"]) or not set(chart_names)<=set(source_types):
        raise ValueError("Chart camera/order or dense subset mismatch")
    # Output identity excludes iteration override so a shorter run can resume to
    # 30000. Geometry, loss schedules and learning-rate horizon remain fixed.
    # Image residency changes memory/transfer behavior, not optimization.
    # Normalize it to the original CPU identity so existing checkpoints can
    # resume with GPU-resident images, and vice versa.
    identity_config=json.loads(json.dumps(config))
    identity_config["model"]["data_device"]="cpu"
    from dense_pipeline_validation import validate_training_inputs, file_hash
    validate_training_inputs(config, args, charts, chart_manifest, dense_manifest)
    effective = dict(model=dict(vars(dataset)), optimization=dict(vars(opt)), pipeline=dict(vars(pipe)))
    effective["model"] = {k:v for k,v in effective["model"].items()
                          if k not in ("source_path", "model_path", "data_device")}
    effective["optimization"].pop("iterations", None)
    identity = hashlib.sha256((json.dumps(identity_config, sort_keys=True)+
        json.dumps(effective, sort_keys=True)+json.dumps(chart_manifest,sort_keys=True)+
        json.dumps(dense_manifest,sort_keys=True)+file_hash(chart_path/"charts_data.npz")).encode()).hexdigest()
    if args.validate_only:
        print(json.dumps(dict(status="passed", dense_images=len(source_types), charts=len(chart_names))))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from scene import GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if state["identity"]!=identity:
            raise ValueError("Checkpoint dataset/configuration identity mismatch")
    dense_cameras, pcd, extent, camera_json = load_cameras(dense_path, dataset)
    if pcd is None or not len(pcd.points):
        raise ValueError("Dense SfM point cloud is missing or empty")
    lookup = {c.image_name:c for c in dense_cameras}
    assert list(lookup)==[Path(e["name"]).stem for e in dense_manifest["frames"]]
    if len(dense_cameras) != len(source_types) or not all(v=="generated" for v in source_types.values()):
        raise ValueError("Expected all SfM views with generated-image source labels")
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("SfM camera centers must span a nonzero spatial extent")
    chart_cameras = [lookup[n] for n in chart_names]
    print("[depth] Generated-only: dense DAV2 depth-order priors disabled", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output/"cfg_args").write_text(str(argparse.Namespace(**vars(dataset))))
    (output/"cameras.json").write_text(json.dumps(camera_json))
    run = dict(config=config, arguments=vars(args), identity=identity, initialization_budget=None,
               source_counts=dict(generated=len(dense_cameras)), chart_counts=dict(generated=len(chart_cameras)))
    run["arguments"] = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (output/"run_config.json").write_text(json.dumps(run, indent=2)+"\n")
    valid, prior_valid = chart_valid_masks(charts)
    invalid_report = [dict(name=n, invalid_geometry_fraction=float((~valid[i]).mean()),
                      invalid_prior_fraction=float((~prior_valid[i]).mean())) for i,n in enumerate(chart_names)]
    (output/"chart_validity.json").write_text(json.dumps(invalid_report, indent=2)+"\n")
    gaussians = GaussianModel(dataset.sh_degree)
    sampler = ViewSampler(len(chart_cameras),len(dense_cameras),config["seed"])
    first=0
    if args.resume:
        restored=list(tensor_tree(state["model"],"cuda"))
        for i in range(1,7):
            restored[i].requires_grad_(True)
        gaussians.restore(tuple(restored), opt)
        gaussians.set_mip_filter(state["use_mip_filter"])
        if state["mip_filter"] is not None:
            gaussians.mip_filter=state["mip_filter"].cuda()
        first=state["iteration"]
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state,restored
    else:
        stats = initialize_all(gaussians,charts,chart_cameras,pcd,extent,valid)
        (output/"initialization_report.json").write_text(json.dumps(stats,indent=2)+"\n")
        gaussians.training_setup(opt)
        gaussians.set_mip_filter(config["use_mip_filter"])
        if gaussians.use_mip_filter:
            print("[init] Computing mip filter for all initialization points",flush=True)
            compute_mip_chunked(gaussians,dense_cameras)
    del pcd
    initialization_peak = torch.cuda.max_memory_allocated()/2**30
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    if first>=opt.iterations:
        raise ValueError("Checkpoint is already at or beyond requested iteration count")
    priors=ChartPriorCache(charts,chart_cameras,valid,prior_valid)
    background=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    saves=set(config["save_iterations"]+[opt.iterations])
    checkpoints=set(config["checkpoint_iterations"]+[opt.iterations])
    log_path=output/"training_log.jsonl"
    start=time.time()
    counts={"chart_generated":0,"dense_generated":0}
    from tqdm import trange
    progress=trange(first+1,opt.iterations+1,desc="Generated-only MAtCha")
    with log_path.open("a" if args.resume else "w",buffering=1) as log:
        for iteration in progress:
            gaussians.update_learning_rate(iteration)
            if iteration%1000==0:
                gaussians.oneupSHdegree()
            kind,index=sampler.next(iteration,config["chart_every"])
            cam=chart_cameras[index] if kind=="chart" else dense_cameras[index]
            label=kind+"_generated"
            counts[label]+=1
            pkg=render(cam,gaussians,pipe,background)
            image=pkg["render"];gt=cam.original_image.cuda()
            rgb=(1.-opt.lambda_dssim)*l1_loss(image,gt)+opt.lambda_dssim*(1.-ssim(image,gt))
            loss=config["generated_rgb_weight"]*rgb
            normal_weight=opt.lambda_normal if iteration>config["normal_consistency_from"] else 0.
            loss=loss+normal_weight*(1.-(pkg["rend_normal"]*pkg["surf_normal"]).sum(0)).mean()
            if iteration>config["distortion_from"] and opt.lambda_dist:
                loss=loss+opt.lambda_dist*pkg["rend_dist"].mean()
            if kind=="chart":
                loss=loss+chart_regularization(pkg,priors.get(index),gaussians,iteration)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {iteration} {cam.image_name}")
            loss.backward()
            with torch.no_grad():
                # Apply this iteration's optimizer step before checkpointing or
                # resizing parameters, so resume never skips a pending update.
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration<opt.densify_until_iter:
                    visible=pkg["visibility_filter"]
                    gaussians.max_radii2D[visible]=torch.maximum(gaussians.max_radii2D[visible],pkg["radii"][visible])
                    gaussians.add_densification_stats(pkg["viewspace_points"],visible)
                    if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0:
                        size_threshold=20 if iteration>opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold,opt.opacity_cull,extent,size_threshold)
                        if gaussians.use_mip_filter:
                            compute_mip_chunked(gaussians,dense_cameras)
                    if iteration%opt.opacity_reset_interval==0:
                        gaussians.reset_opacity()
                elif iteration%100==0 and iteration<opt.iterations-100 and gaussians.use_mip_filter:
                    compute_mip_chunked(gaussians,dense_cameras)
                record=dict(iteration=iteration,view=cam.image_name,source=label,rgb=float(rgb),loss=float(loss),
                            gaussians=len(gaussians.get_xyz),allocated_gib=torch.cuda.memory_allocated()/2**30,
                            peak_gib=torch.cuda.max_memory_allocated()/2**30)
                if iteration%10==0 or iteration==first+1:
                    log.write(json.dumps(record)+"\n")
                    progress.set_postfix(loss=f"{record['loss']:.4f}",points=record["gaussians"])
                if iteration in saves:
                    target=output/"point_cloud"/f"iteration_{iteration}"/"point_cloud.ply"
                    target.parent.mkdir(parents=True,exist_ok=True)
                    save_ply_streaming(target,gaussians)
                if iteration in checkpoints:
                    save_checkpoint(output/f"chkpnt{iteration}.pth",gaussians,iteration,sampler,identity)
            del pkg,image,gt,rgb,loss
    report=dict(completed_iteration=opt.iterations,elapsed_seconds=time.time()-start,source_steps=counts,
                gaussians=len(gaussians.get_xyz),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, initialization_peak_gib=initialization_peak)
    (output/"training_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
