"""Portable, consistent subsets of the repository's single-track MASt3R export.

No camera estimation, world-coordinate scaling, or point-budget sampling occurs.
Non-MASt3R/multi-track exports are rejected rather than silently reinterpreted.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OBS = np.dtype([('xy', '<f8', (2,)), ('id', '<i8')])
POINT = np.dtype([('id', '<u8'), ('xyz', '<f8', (3,)), ('rgb', 'u1', (3,)),
                  ('error', '<f8'), ('track_length', '<u8'),
                  ('image_id', '<i4'), ('point2D_idx', '<i4')])
PLY = np.dtype([('xyz', '<f4', (3,)), ('normal', '<f4', (3,)), ('rgb', 'u1', (3,))])


def read_keyframes(path):
    names = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not names or len(names) != len(set(names)):
        raise ValueError('Keyframes must be a nonempty list of unique image filenames')
    if any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Keyframes must contain exact image filenames, one per line, not paths/indices')
    return sorted(names)


def colmap_module():
    spec = importlib.util.spec_from_file_location('dense_colmap', ROOT/'mast3r/colmap/read_write_model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_images(path):
    result = {}
    with Path(path).open('rb') as f:
        total, = struct.unpack('<Q', f.read(8))
        for _ in range(total):
            header = f.read(64)
            values = struct.unpack('<idddddddi', header)
            raw = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Truncated images.bin')
                if c == b'\0':
                    break
                raw.extend(c)
            name = raw.decode()
            count, = struct.unpack('<Q', f.read(8))
            data = f.read(count*OBS.itemsize)
            if len(data) != count*OBS.itemsize or name in result:
                raise ValueError('Truncated observations or duplicate image name')
            result[name] = (values, header, np.frombuffer(data, OBS).copy())
    return result


def write_images(path, names, images):
    with Path(path).open('wb') as f:
        f.write(struct.pack('<Q', len(names)))
        for name in names:
            _, header, obs = images[name]
            f.write(header)
            f.write(name.encode()+b'\0')
            f.write(struct.pack('<Q', len(obs)))
            obs.tofile(f)


def relative_link(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(Path(source).resolve(), destination.parent.resolve()))


def metadata_identity(metadata):
    """Ignore machine-specific filepaths; retain names, poses, intrinsics/order."""
    value = dict(metadata)
    value['filepaths'] = [Path(p).name for p in value['filepaths']]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_scenes(sfm, keyframes, output):
    sfm, output = Path(sfm).resolve(), Path(output).resolve()
    selected = read_keyframes(keyframes)
    if len(selected) < 2:
        raise ValueError("Chart alignment requires at least two selected views")
    metadata = json.loads((sfm/'cameras.json').read_text())
    names = [Path(p).name for p in metadata['filepaths']]
    if not names or len(names) != len(set(names)) or len({Path(n).stem for n in names}) != len(names):
        raise ValueError('SfM must contain unique filenames and unique image stems')
    if not set(selected) <= set(names):
        raise ValueError(f'Keyframes absent from SfM: {sorted(set(selected)-set(names))}')
    n = len(names)
    if any(not isinstance(v, list) or len(v) != n for v in metadata.values()):
        raise ValueError('Expected aligned per-image MASt3R camera metadata arrays')
    identity = metadata_identity(metadata)
    manifest_path = output/'pipeline_manifest.json'
    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f'Incomplete/existing output; choose a new directory: {output}')
        old = json.loads(manifest_path.read_text())
        if old['source_identity'] != identity or old['chart_names'] != selected:
            raise ValueError('Prepared output belongs to a different SfM or keyframe selection')
        # Avoid silently reusing links to a different export with similar cameras.
        if (output/'dense_scene/images'/names[0]).resolve() != (sfm/'images'/names[0]).resolve():
            raise ValueError('Prepared links refer to another SfM directory; prepare a new output')
        return old
    colmap = colmap_module()
    cameras = colmap.read_cameras_binary(sfm/'sparse/0/cameras.bin')
    images = read_images(sfm/'sparse/0/images.bin')
    if set(images) != set(names):
        raise ValueError('COLMAP and cameras.json image sets differ')
    for name in names:
        with Image.open(sfm/'images'/name) as im:
            cam = cameras[images[name][0][-1]]
            if im.size != (cam.width, cam.height):
                raise ValueError(f'Image/camera resolution mismatch: {name}')
        obs = images[name][2]
        if len(obs) and (np.any(obs['id'] < 0) or np.any(np.diff(obs['id']) <= 0)):
            raise ValueError('Expected sorted unique MASt3R point observations')
    for name in selected:
        pm = json.loads((sfm/'pointmaps'/(Path(name).stem+'.json')).read_text())
        conf, xyz = np.asarray(pm['confs']), np.asarray(pm['points'])
        if conf.ndim != 2 or xyz.size != conf.size*3 or not np.isfinite(xyz).all() or not np.isfinite(conf).all():
            raise ValueError(f'Invalid pointmap: {name}')
    source = sfm/'sparse/0/points3D.bin'
    with source.open('rb') as f:
        count, = struct.unpack('<Q', f.read(8))
    if not count or source.stat().st_size != 8+count*POINT.itemsize:
        raise ValueError('Expected nonempty single-track MASt3R points3D.bin export')
    points = np.memmap(source, dtype=POINT, mode='r', offset=8, shape=(count,))
    image_by_id = {entry[0][0]: name for name, entry in images.items()}
    if len(image_by_id) != len(images):
        raise ValueError('Duplicate COLMAP image IDs')
    # Validate before creating output. Match ID to compact confidence-filtered
    # observations; the upstream exporter instead stores raw pixel indices.
    seen = {i: 0 for i in image_by_id}
    for start in range(0, count, 500_000):
        chunk = points[start:start+500_000]
        if np.any(chunk['track_length'] != 1) or not np.isfinite(chunk['xyz']).all():
            raise ValueError('Unsupported tracks or nonfinite SfM points')
        for iid in np.unique(chunk['image_id']):
            if int(iid) not in image_by_id:
                raise ValueError('Point track refers to an unknown image')
            ids = chunk['id'][chunk['image_id'] == iid]
            obs = images[image_by_id[int(iid)]][2]['id']
            idx = np.searchsorted(obs, ids)
            if np.any(idx >= len(obs)) or not np.array_equal(obs[idx], ids):
                raise ValueError('Point IDs disagree with image observations')
            seen[int(iid)] += len(ids)
    if any(seen[i] != len(images[name][2]) for i, name in image_by_id.items()):
        raise ValueError('Point/observation counts disagree')
    output.mkdir(parents=True)
    index = {name: i for i, name in enumerate(names)}
    for folder_name, subset_names in [('scene', selected), ('dense_scene', sorted(names))]:
        folder = output/folder_name
        sparse = folder/'sparse/0'
        sparse.mkdir(parents=True)
        image_ids = {images[name][0][0] for name in subset_names}
        camera_ids = {images[name][0][-1] for name in subset_names}
        colmap.write_cameras_binary({i: cameras[i] for i in sorted(camera_ids)}, sparse/'cameras.bin')
        write_images(sparse/'images.bin', subset_names, images)
        num = sum(seen[i] for i in image_ids)
        with (sparse/'points3D.bin').open('wb') as binary, (sparse/'points3D.ply').open('wb') as ply:
            binary.write(struct.pack('<Q', num))
            ply.write(('ply\nformat binary_little_endian 1.0\n'+f'element vertex {num}\n'+
                       'property float x\nproperty float y\nproperty float z\n'+
                       'property float nx\nproperty float ny\nproperty float nz\n'+
                       'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
            for start in range(0, count, 500_000):
                chunk = points[start:start+500_000]
                part = chunk[np.isin(chunk['image_id'], list(image_ids))].copy()
                for iid in np.unique(part['image_id']):
                    mask = part['image_id'] == iid
                    obs = images[image_by_id[int(iid)]][2]['id']
                    part['point2D_idx'][mask] = np.searchsorted(obs, part['id'][mask])
                part.tofile(binary)
                vertices = np.zeros(len(part), PLY)
                vertices['xyz'], vertices['rgb'] = part['xyz'], part['rgb']
                vertices.tofile(ply)
        for name in subset_names:
            relative_link(sfm/'images'/name, folder/'images'/name)
            if folder_name == 'scene':
                relative_link(sfm/'pointmaps'/(Path(name).stem+'.json'), folder/'pointmaps'/(Path(name).stem+'.json'))
        subset = {k: [v[index[name]] for name in subset_names] for k, v in metadata.items()}
        subset['filepaths'] = [str(folder/'images'/name) for name in subset_names]
        (folder/'cameras.json').write_text(json.dumps(subset)+'\n')
        manifest = dict(source_identity=identity, images=len(subset_names), points=num,
                        frames=[dict(name=name, source_type='generated') for name in subset_names])
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = dict(schema='matcha_generated_dense_v1', source_identity=identity,
                  chart_names=selected, dense_names=sorted(names), coordinate_system_preserved=True)
    manifest_path.write_text(json.dumps(result, indent=2)+'\n')
    return result
