import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

pointLKPO_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_pointLKPO_so.so'))

def pointLKPO_select(xyz, radius):
    idx = pointSIFT_module.lkpo_select(xyz, radius)
    return idx
ops.NoGradient('LKPOSelect')

def pointLKPO_select_four(xyz, radius):
    idx = pointSIFT_module.lkpo_select_four(xyz, radius)
    return idx
ops.NoGradient('LKPOSelectFour')

