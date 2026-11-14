"""Stage 2: export selected charts/all dense views, then run upstream alignment."""
import argparse
from pathlib import Path
import subprocess
import sys

from dense_scene_io import ROOT, prepare_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm_scene', type=Path, required=True)
    parser.add_argument('--keyframes', type=Path, required=True, help='One exact image filename per line')
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--config', default='default', choices=['default', 'strong'])
    parser.add_argument('--depthanything_encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=Path, default=ROOT/'Depth-Anything-V2/checkpoints')
    parser.add_argument('--prepare_only', action='store_true', help='Export/validate subsets without GPU alignment')
    args = parser.parse_args()
    output = args.output_path.resolve()
    if (output/'charts_data.npz').exists():
        raise FileExistsError('Aligned charts already exist; choose a new output_path')
    result = prepare_scenes(args.sfm_scene, args.keyframes, output)
    print(f"Prepared {len(result['chart_names'])} charts and {len(result['dense_names'])} dense cameras", flush=True)
    if args.prepare_only:
        return
    command = [sys.executable, str(ROOT/'scripts/align_charts.py'),
               '--source_path', str(output/'scene'), '--mast3r_scene', str(output/'scene'),
               '--output_path', str(output), '--config', args.config,
               '--depthanything_encoder', args.depthanything_encoder,
               '--depthanythingv2_checkpoint_dir', str(args.depthanythingv2_checkpoint_dir.resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    if not (output/'charts_data.npz').is_file():
        raise RuntimeError('Alignment finished without charts_data.npz')


if __name__ == '__main__':
    main()
