"""
Export the trained PyTorch model to ONNX with INT8 quantization.

Produces:
  models/tetris_eval_fp32.onnx  — full-precision export (for reference)
  models/tetris_eval.onnx       — INT8 quantized (deployed to public/models/)

After running this script, copy the output to the web app:
  cp models/tetris_eval.onnx ../tetris-web/public/models/tetris_eval.onnx

Usage:
    python export_onnx.py [--checkpoint models/tetris_eval.pt]
                          [--out-fp32 models/tetris_eval_fp32.onnx]
                          [--out models/tetris_eval.onnx]
"""

import argparse
import os
import numpy as np
import torch
from train import TetrisCNN


def export(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = TetrisCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    # FP32 export — opset 12 is most compatible with onnxruntime quantization.
    # keep_initializers_as_inputs=False ensures weights are embedded in the file
    # rather than split into an external .data file.
    dummy = torch.zeros(1, 1, 20, 10)
    torch.onnx.export(
        model, dummy, args.out_fp32,
        input_names=['board'],
        output_names=['value'],
        dynamic_axes={'board': {0: 'batch_size'}, 'value': {0: 'batch_size'}},
        opset_version=12,
        keep_initializers_as_inputs=False,
    )
    fp32_size = os.path.getsize(args.out_fp32) / 1024
    print(f'FP32 model: {args.out_fp32}  ({fp32_size:.1f} KB)')

    # INT8 dynamic quantization (falls back to FP32 if quantization fails)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    try:
        quantize_dynamic(
            args.out_fp32, args.out,
            weight_type=QuantType.QUInt8,
        )
        int8_size = os.path.getsize(args.out) / 1024
        print(f'INT8 model: {args.out}  ({int8_size:.1f} KB)')
        print(f'Size reduction: {fp32_size / int8_size:.1f}×')
        deploy_path = args.out
    except Exception as e:
        print(f'INT8 quantization failed ({e}), using FP32 model instead')
        import shutil
        shutil.copy(args.out_fp32, args.out)
        deploy_path = args.out

    # Quick sanity check: run inference on a random batch
    import onnxruntime as ort
    sess = ort.InferenceSession(deploy_path)
    batch = np.random.rand(8, 1, 20, 10).astype(np.float32)
    out = sess.run(['value'], {'board': batch})[0]
    print(f'\nSanity check — output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]')
    print('\nDeploy to the web app (use FP32 — ORT Web WASM does not support INT8 dynamic quantization):')
    print(f'  cp {args.out_fp32} ../tetris-web/public/models/tetris_eval.onnx')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/tetris_eval.pt')
    parser.add_argument('--out-fp32',   default='models/tetris_eval_fp32.onnx')
    parser.add_argument('--out',        default='models/tetris_eval.onnx')
    args = parser.parse_args()
    export(args)


if __name__ == '__main__':
    main()
