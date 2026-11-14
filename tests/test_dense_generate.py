"""CPU tests: generic subsets, camera/point identity and training schedules."""
import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import dense_scene_io as io
from dense_training_utils import ViewSampler, chart_valid_masks, save_ply_streaming
from dense_pipeline_validation import validate_training_inputs


class DenseGenerateTests(unittest.TestCase):
    def make_sfm(self, root, count):
        source = root/'sfm'
        sparse = source/'sparse/0'
        sparse.mkdir(parents=True)
        (source/'images').mkdir()
        (source/'pointmaps').mkdir()
        names = [f'image_{i:03d}.png' for i in range(count)]
        cameras, images = {}, {}
        metadata = dict(filepaths=[], focals=[], cams2world=[])
        points = np.zeros(count*2, io.POINT)
        for i, name in enumerate(names):
            iid = 10+i
            Image.new('RGB', (8, 6), (i, 2, 3)).save(source/'images'/name)
            (source/'pointmaps'/(Path(name).stem+'.json')).write_text(json.dumps(dict(
                points=np.ones((6,3)).tolist(), confs=np.ones((2,3)).tolist())))
            camera = io.colmap_module().Camera(iid, 'PINHOLE', 8, 6, np.array([5.,5.,4.,3.]))
            cameras[iid] = camera
            ids = np.array([i*10+1, i*10+7])
            obs = np.zeros(2, io.OBS)
            obs['id'] = ids
            obs['xy'] = [[1.,1.], [3.,2.]]
            values = (iid,1.,0.,0.,0.,float(i),0.,0.,iid)
            images[name] = (values, struct.pack('<idddddddi', *values), obs)
            part = points[2*i:2*i+2]
            part['id'] = ids
            part['xyz'] = [[i,0.,1.], [i,1.,2.]]
            part['rgb'] = [20,30,40]
            part['track_length'] = 1
            part['image_id'] = iid
            # Intentionally raw pixel indices (the original export's behavior).
            part['point2D_idx'] = [1,7]
            metadata['filepaths'].append('/old/machine/'+name)
            metadata['focals'].append(5.)
            pose = np.eye(4); pose[0,3] = -i
            metadata['cams2world'].append(pose.tolist())
        io.colmap_module().write_cameras_binary(cameras, sparse/'cameras.bin')
        io.write_images(sparse/'images.bin', names, images)
        with (sparse/'points3D.bin').open('wb') as f:
            f.write(struct.pack('<Q', len(points))); points.tofile(f)
        (source/'cameras.json').write_text(json.dumps(metadata))
        return source, names, points, images

    def test_arbitrary_counts_preserve_geometry_and_compact_tracks(self):
        for count in (3, 7):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, names, original, images = self.make_sfm(root, count)
                selected = [names[-1], names[0]]  # Non-sorted supplied order.
                keyframes = root/'keys.txt'; keyframes.write_text('\n'.join(selected))
                output = root/'charts'
                result = io.prepare_scenes(source, keyframes, output)
                self.assertEqual(len(result['dense_names']), count)
                self.assertEqual(result['chart_names'], sorted(selected))
                for folder, expected in [('scene', sorted(selected)), ('dense_scene', names)]:
                    path = output/folder
                    written = io.read_images(path/'sparse/0/images.bin')
                    self.assertEqual(list(written), expected)
                    pts = np.fromfile(path/'sparse/0/points3D.bin', io.POINT, offset=8)
                    np.testing.assert_array_equal(pts['xyz'], original['xyz'][np.isin(original['image_id'], [images[n][0][0] for n in expected])])
                    for name in expected:
                        self.assertEqual(written[name][1], images[name][1])
                        subset = pts[pts['image_id']==written[name][0][0]]
                        np.testing.assert_array_equal(written[name][2]['id'][subset['point2D_idx']], subset['id'])
                        self.assertFalse((path/'images'/name).readlink().is_absolute())
                self.assertEqual(result, io.prepare_scenes(source, keyframes, output))
                keyframes.write_text(names[1])
                with self.assertRaises(ValueError):
                    io.prepare_scenes(source, keyframes, output)

    def test_bad_keyframes_and_bad_tracks_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,names,points,_=self.make_sfm(root,3)
            keys=root/'keys.txt'; keys.write_text(names[0]+'\n'+names[0])
            with self.assertRaises(ValueError): io.prepare_scenes(source,keys,root/'charts')
            keys.write_text('missing.png')
            with self.assertRaises(ValueError): io.prepare_scenes(source,keys,root/'charts')
            keys.write_text(names[0])
            points['track_length'][0]=2
            with (source/'sparse/0/points3D.bin').open('wb') as f:
                f.write(struct.pack('<Q',len(points)));points.tofile(f)
            with self.assertRaises(ValueError): io.prepare_scenes(source,keys,root/'charts')
            self.assertFalse((root/'charts').exists())

    def test_pointmaps_follow_selected_camera_names_including_two_views(self):
        from matcha.pointmap.mast3r import get_pointmap_with_mast3r
        with tempfile.TemporaryDirectory() as temp:
            source,names,_,_=self.make_sfm(Path(temp),3)
            for i,name in enumerate(names):
                (source/'pointmaps'/(Path(name).stem+'.json')).write_text(json.dumps(dict(
                    rgb=np.zeros((2,3,3)).tolist(), points=(np.ones((2,3,3))*i).tolist(),
                    confs=np.ones((2,3)).tolist())))
            pm=get_pointmap_with_mast3r(str(source),image_indices=[2,0],max_img_size=8,device='cpu')
            np.testing.assert_array_equal(pm.points3d[:,0,0,0].numpy(),[2.,0.])
            pm=get_pointmap_with_mast3r(str(source),n_images_in_pointmap=2,max_img_size=8,device='cpu')
            np.testing.assert_array_equal(pm.points3d[:,0,0,0].numpy(),[0.,2.])

    def test_source_identity_is_machine_independent(self):
        a=dict(filepaths=['/old/a.png'],focals=[5.],cams2world=[np.eye(4).tolist()])
        b=dict(a,filepaths=['/server/a.png'])
        self.assertEqual(io.metadata_identity(a),io.metadata_identity(b))
        b['focals']=[6.]
        self.assertNotEqual(io.metadata_identity(a),io.metadata_identity(b))

    def test_fixed_schedule_and_exact_resume(self):
        sampler=ViewSampler(3,7,42)
        choices=[sampler.next(i,20) for i in range(1,8001)]
        self.assertTrue(all((kind=='chart')==(i%20==0) for i,(kind,_) in enumerate(choices,1)))
        dense=[i for kind,i in choices if kind=='dense']
        self.assertEqual(set(dense[:7]),set(range(7)))
        restored=ViewSampler(3,7,999);restored.load_state_dict(sampler.state_dict())
        self.assertEqual([sampler.next(i,20) for i in range(8001,8100)],
                         [restored.next(i,20) for i in range(8001,8100)])

    def test_chart_validation_and_invalid_geometry_mask(self):
        shape=(2,2,3)
        charts=dict(depths=np.ones(shape),prior_depths=np.ones(shape),confs=np.ones(shape),
                    pts=np.ones(shape+(3,)),scale_factor=np.array(1.))
        config=dict(chart_every=20,generated_rgb_weight=1.,save_iterations=[7],checkpoint_iterations=[5])
        args=argparse.Namespace(iterations=40,densification_interval=100,opacity_reset_interval=3000)
        manifest=dict(frames=[dict(name=n,source_type='generated') for n in ['a.png','b.png']])
        validate_training_inputs(config,args,charts,manifest,manifest)
        charts['depths'][0,0,0]=-1
        mask,_=chart_valid_masks(charts,100)
        self.assertFalse(mask[0,0,0]);self.assertEqual(charts['depths'][0,0,0],-1)
        charts['depths'][0]=-1
        with self.assertRaises(ValueError): validate_training_inputs(config,args,charts,manifest,manifest)

    def test_chart_scale_matches_upstream_without_far_clipping(self):
        # Chart alignment saved normalized depths, but stage 3 must consume
        # their original SfM scale, even when depths exceed Camera.zfar=100.
        scale = .0017412
        shape = (2, 2, 3)
        charts = dict(depths=np.full(shape, 5.), prior_depths=np.full(shape, 6.),
                      confs=np.ones(shape), pts=np.ones(shape+(3,)),
                      scale_factor=np.array(scale))
        original = {k: v.copy() for k, v in charts.items()}
        config = dict(chart_every=20, generated_rgb_weight=1.,
                      save_iterations=[7], checkpoint_iterations=[5])
        args = argparse.Namespace(iterations=40, densification_interval=100,
                                  opacity_reset_interval=3000)
        manifest = dict(frames=[dict(name=n, source_type='generated')
                                for n in ['a.png', 'b.png']])
        validate_training_inputs(config, args, charts, manifest, manifest)
        valid, prior_valid = chart_valid_masks(charts, 100.)
        self.assertTrue(valid.all())
        self.assertTrue(prior_valid.all())
        # Input checks must not apply a near-plane visibility rule either.
        charts['depths'][0, 0, 0] = .001 * scale
        self.assertTrue(chart_valid_masks(charts)[0][0, 0, 0])
        charts['depths'][0, 0, 0] = original['depths'][0, 0, 0]
        for key in charts:
            np.testing.assert_array_equal(charts[key], original[key])
        charts['pts'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'a.png'):
            validate_training_inputs(config, args, charts, manifest, manifest)
        charts['pts'] = original['pts']
        charts['prior_depths'][1] = np.inf
        with self.assertRaisesRegex(ValueError, 'b.png'):
            validate_training_inputs(config, args, charts, manifest, manifest)


    def test_streamed_ply_preserves_sh_order_and_mip(self):
        import torch
        from plyfile import PlyData
        class TinyModel:
            def __init__(self):
                self._xyz=torch.arange(21,dtype=torch.float32).reshape(7,3)
                self._features_dc=torch.arange(21,dtype=torch.float32).reshape(7,1,3)
                self._features_rest=torch.arange(7*15*3,dtype=torch.float32).reshape(7,15,3)
                self._opacity=torch.ones(7,1)
                self._scaling=torch.ones(7,2)*2
                self._rotation=torch.ones(7,4)*3
                self.mip_filter=torch.ones(7,1)*.01
                self.use_mip_filter=True
            @property
            def get_xyz(self):
                return self._xyz
            def construct_list_of_attributes(self):
                return ['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity','scale_0','scale_1']+[f'rot_{i}' for i in range(4)]+['mip_filter']
        model=TinyModel()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.ply'
            save_ply_streaming(path,model,chunk_size=3)
            data=PlyData.read(path)['vertex']
            np.testing.assert_array_equal(data['x'],model._xyz[:,0].numpy())
            expected=model._features_rest.transpose(1,2).flatten(1).numpy()
            for i in range(45):
                np.testing.assert_array_equal(data[f'f_rest_{i}'],expected[:,i])
            np.testing.assert_array_equal(data['mip_filter'],model.mip_filter[:,0].numpy())


if __name__=='__main__':
    unittest.main()
