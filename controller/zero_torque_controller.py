import numpy as np
import onnxruntime as ort
from common.bydmimic_utils import *
from controller.base_controller import BaseController

class ZeroTorqueController(BaseController):
    def __init__(self):
        self.default_dof_pos = np.zeros(29, dtype=np.float32)
        self.kps = np.zeros(29, dtype=np.float32)
        self.kds = np.zeros(29, dtype=np.float32)
        
    def reset(self, env_data=None):
        pass
        
    def step(self, mujoco_data):
        target_q = np.zeros(29, dtype=np.float32)
        return target_q, self.kps, self.kds