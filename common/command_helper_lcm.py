
from lcm_types import low_cmd_lcmt, low_state_lcmt
from typing import Union
import numpy as np

class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def create_damping_cmd(cmd: low_cmd_lcmt):
    size = 29
    cmd.q = np.zeros(size)
    cmd.dq = np.zeros(size)
    cmd.tau = np.zeros(size)
    cmd.kp = np.zeros(size)
    cmd.kd = np.zeros(size)+8


def create_zero_cmd(cmd: low_cmd_lcmt):
    size = 29
    cmd.q = np.zeros(size)
    cmd.dq = np.zeros(size)
    cmd.tau = np.zeros(size)
    cmd.kp = np.zeros(size)
    cmd.kd = np.zeros(size)


def init_cmd_hg(cmd: low_cmd_lcmt, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = 29
    cmd.q = np.zeros(size)
    cmd.dq = np.zeros(size)
    cmd.tau = np.zeros(size)
    cmd.kp = np.zeros(size)
    cmd.kd = np.zeros(size)
