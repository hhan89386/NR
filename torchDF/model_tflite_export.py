import argparse
import tempfile
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, optimizations=None) -> str:
    """Convert an ONNX model to TensorFlow Lite format."""
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_rep.export_graph(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        if optimizations:
            converter.optimizations = optimizations
        tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def _parse_opts(opts: str):
    if not opts:
        return None
    opts_map = {
        "DEFAULT": tf.lite.Optimize.DEFAULT,
        "SIZE": tf.lite.Optimize.OPTIMIZE_FOR_SIZE,
        "LATENCY": tf.lite.Optimize.OPTIMIZE_FOR_LATENCY,
    }
    results = []
    for name in opts.split(','):
        key = name.strip().upper()
        if key:
            if key not in opts_map:
                raise ValueError(f"Unknown optimization flag: {name}")
            results.append(opts_map[key])
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite")
    parser.add_argument("onnx_path", help="Path to source ONNX model")
    parser.add_argument("tflite_path", help="Destination path for TFLite model")
    parser.add_argument(
        "--optimizations",
        help="Comma separated list of optimizations: DEFAULT,SIZE,LATENCY",
    )
    args = parser.parse_args()
    opts = _parse_opts(args.optimizations)
    convert_onnx_to_tflite(args.onnx_path, args.tflite_path, opts)


if __name__ == "__main__":
    main()
