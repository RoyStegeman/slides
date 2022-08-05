import logging
import time

import coloredlogs
import numpy as np

# import lhapdf
from validphys.api import API

logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=logger)


xgrid_input = np.loadtxt("xgrid.out")

# pdf = lhapdf.mkPDF("model_for_onnx")
pdf = API.pdf(pdf="model_for_onnx").load().members[1]
st = time.time()
for x_value in xgrid_input:
    pdf.xfxQ([-6, -5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5, 6, 22], x_value, 1.65)
et = time.time()
elapsed_time = et - st
logger.info(f"Validphys execution time: {elapsed_time} seconds")
