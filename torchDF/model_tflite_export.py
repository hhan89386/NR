import argparse
import tempfile
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Sanitize ONNX names so TensorFlow accepts them."""

    def clean(name: str) -> str:
        return name.lstrip("/").replace("/", "_")

    name_map = {}
    # Initializers
    for init in model.graph.initializer:
        new_name = clean(init.name)
        if new_name != init.name:
            name_map[init.name] = new_name
            init.name = new_name
    # Value infos (inputs/outputs/intermediate)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        new_name = clean(vi.name)
        if new_name != vi.name:
            name_map[vi.name] = new_name
            vi.name = new_name
    # Nodes
    for node in model.graph.node:
        node.name = clean(node.name)
        node.input[:] = [name_map.get(i, clean(i)) for i in node.input]
        node.output[:] = [name_map.get(o, clean(o)) for o in node.output]
    return model


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, optimizations=None) -> str:
    """Convert an ONNX model to TensorFlow Lite format."""
    model = onnx.load(onnx_path)
    from onnx import version_converter

    try:
        model = version_converter.convert_version(model, 15)
    except Exception:
        pass
    model = sanitize_model(model)
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
    for name in opts.split(","):
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
