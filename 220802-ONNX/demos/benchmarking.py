import logging
import os
import pathlib
import time

import coloredlogs
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import torch
from onnx2torch import convert
from onnx_pytorch import code_gen
from onnx_tf.backend import prepare

tf.get_logger().setLevel("INFO")

logger = logging.getLogger(__name__)

model_type = "n3fit"  # sffit or n3fit
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=logger)


def _gen_integration_input(nx=int(2e3)):
    """
    Generates a np.array (shaped (nx,1)) of nx elements where the
    nx/2 first elements are a logspace between 0 and 0.1
    and the rest a linspace from 0.1 to 0
    """
    lognx = int(nx / 2)
    linnx = int(nx - lognx)
    xgrid_log = np.logspace(-9, -1, lognx + 1, dtype="float32")
    xgrid_lin = np.linspace(0.1, 1, linnx, dtype="float32")
    xgrid = np.concatenate([xgrid_log[:-1], xgrid_lin]).reshape(nx, 1)
    return np.expand_dims(xgrid, axis=0)


cwd = pathlib.Path.cwd()
onnx_model_path = cwd / "model.onnx"
tf_model_path = cwd / "n3fit_model"
sffit_tf_model_path = cwd / "sffit_model"
sffit_onnx_model_path = cwd / "sffit_model.onnx"

input_xgrid = np.geomspace(1e-9, 1.0, num=int(1e5), dtype="float32")
input_xgrid = input_xgrid.reshape(1, input_xgrid.size, 1)


if model_type == "n3fit":
    int_grid = _gen_integration_input()

    # if not onnx_model_path.exists():
    os.system(
        "python -m tf2onnx.convert --saved-model n3fit_model --output model.onnx --opset 12"
    )

    ### Pure Tensorflow
    model_rep = tf.saved_model.load(tf_model_path)
    st = time.time()
    logger.info("running inteference with TF...")
    out1 = model_rep({"input_1": input_xgrid, "integration_grid": int_grid})
    et = time.time()
    elapsed_time = et - st
    logger.info(f"TensorFlow execution time: {elapsed_time} seconds")

    ### TensorFlow through onnx
    onnx_model = onnx.load(onnx_model_path)  # load onnx model
    model_rep = prepare(onnx_model)  # prepare tf representation
    st = time.time()
    logger.info("running inteference with onnx_tf...")
    model_rep.run([input_xgrid, int_grid])
    et = time.time()
    elapsed_time = et - st
    logger.info(f"onnx_tf execution time: {elapsed_time} seconds")

    ### Pytorch (third party packages don't support all operators)

    # model_rep = convert(onnx_model_path)
    # # import ipdb; ipdb.set_trace()
    # torch_input_xgrid = torch.from_numpy(input_xgrid).cpu()
    # torch_int_grid = torch.from_numpy(int_grid)
    # logger.info("running inteference with pytorch...")
    # out = model_rep([torch_input_xgrid, torch_int_grid])

    # import ipdb; ipdb.set_trace()
    # code_gen.gen(sffit_onnx_model_path, "./")

    ### ONNX runtime
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    # sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = rt.InferenceSession(
        path_or_bytes=str(onnx_model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    st = time.time()
    logger.info("running inteference with onnx runtime...")
    out2 = sess.run(
        output_names=None,
        input_feed={"input_1": input_xgrid, "integration_grid": int_grid},
    )
    et = time.time()
    elapsed_time = et - st
    logger.info(f"onnx runtime execution time: {elapsed_time} seconds")


    ### LHAPDF
    np.savetxt("xgrid.out", input_xgrid[0,:])
    os.system(
        "/home/roy/miniconda3/envs/nnpdf/bin/python run_vp_pdf.py"
    )

elif model_type == "sffit":

    os.system(
        "python -m tf2onnx.convert --saved-model sffit_model --output sffit_model.onnx --opset 12"
    )

    qgrid = np.array([1.65] * input_xgrid.shape[1]).reshape(1, input_xgrid.shape[1], 1)
    agrid = np.array([20] * input_xgrid.shape[1]).reshape(1, input_xgrid.shape[1], 1)
    input_grid = np.concatenate([input_xgrid, qgrid, agrid], axis=2).astype("float32")

    ### Pure Tensorflow
    model_rep = tf.saved_model.load(sffit_tf_model_path)
    st = time.time()
    logger.info("running inteference with TF...")
    out1 = model_rep(input_grid)
    et = time.time()
    elapsed_time = et - st
    logger.info(f"TensorFlow execution time: {elapsed_time} seconds")

    ### TensorFlow through onnx
    onnx_model = onnx.load(sffit_onnx_model_path)  # load onnx model
    model_rep = prepare(onnx_model)  # prepare tf representation
    st = time.time()
    logger.info("running inteference with onnx_tf...")
    model_rep.run([input_grid])
    et = time.time()
    elapsed_time = et - st
    logger.info(f"onnx_tf execution time: {elapsed_time} seconds")

    ### Pytorch  (third party packages don't support all operators)

    ### ONNX runtime
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 8
    # sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = rt.InferenceSession(
        path_or_bytes=str(sffit_onnx_model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    st = time.time()
    logger.info("running inteference with onnx runtime...")
    out2 = sess.run(
        output_names=None,
        input_feed={
            "input_24": input_grid,
        },
    )
    et = time.time()
    elapsed_time = et - st
    logger.info(f"onnx runtime execution time: {elapsed_time} seconds")

if np.allclose(out2[0], out1[0], atol=1e-5):
    logger.info("onnx runtime and native TF give the same result!")
else:
    logger.warning("WARNING: onnx runtime and native TF outputs differ ")
