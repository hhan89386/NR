import tempfile
import tarfile
import os
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

from model_tflite_export import sanitize_model, convert_onnx_to_tflite


def _random_inputs(model: onnx.ModelProto):
    inputs = {}
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[inp.type.tensor_type.elem_type]
        inputs[inp.name] = np.random.rand(*dims).astype(dtype)
    return inputs


def test_onnx_tflite_equivalence():
    tar_path = "../models/DeepFilterNet3_torchDF_onnx.tar"
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmpdir)
        onnx_path = os.path.join(tmpdir, "DeepFilterNet3_torchDF_onnx", "denoiser_model.onnx")
        model = sanitize_model(onnx.load(onnx_path))
        sanitized_onnx = os.path.join(tmpdir, "model.onnx")
        tflite_path = os.path.join(tmpdir, "model.tflite")
        onnx.save(model, sanitized_onnx)
        convert_onnx_to_tflite(sanitized_onnx, tflite_path)

        inputs = _random_inputs(model)

        session = ort.InferenceSession(sanitized_onnx, providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(None, inputs)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        for detail in interpreter.get_input_details():
            interpreter.set_tensor(detail["index"], inputs[detail["name"]])
        interpreter.invoke()
        tflite_outputs = [
            interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()
        ]

        for o, t in zip(onnx_outputs, tflite_outputs):
            assert np.allclose(o, t, atol=1e-3)
