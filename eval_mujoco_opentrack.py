from glob import glob
import re
import time
import argparse
from enum import Enum

from tqdm import tqdm

import numpy as np
import os
import torch
import joblib
import mujoco
# import mujoco_viewer
import onnxruntime as ort
from mujoco import MjModel, MjData, mj_step, mj_forward
from collections import deque



# ============================================================
# ======================== Constants =========================
# ============================================================


class AnchorBody(Enum):
    PELVIS = 0
    TORSO_LINK = 9


anchor_body = AnchorBody.PELVIS

isaaclab_joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

mujoco_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(n) for n in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(n) for n in isaaclab_joint_names]

# ============================================================
# ==================== Joint parameters ======================
# ============================================================

stiffness_dict = {
    ".*_hip_pitch_joint": 100,
    ".*_hip_roll_joint": 100,
    ".*_hip_yaw_joint": 100,
    ".*_knee_joint": 200,
    ".*_ankle_pitch_joint": 80,
    ".*_ankle_roll_joint": 20,
    "waist_roll_joint": 300,
    "waist_pitch_joint": 300,
    "waist_yaw_joint": 300,
    ".*_shoulder_pitch_joint": 90,
    ".*_shoulder_roll_joint": 60,
    ".*_shoulder_yaw_joint": 20,
    ".*_elbow_joint": 60,
    ".*_wrist_roll_joint": 20,
    ".*_wrist_pitch_joint": 20,
    ".*_wrist_yaw_joint": 20,
}

damping_dict = {
    ".*_hip_pitch_joint": 2,
    ".*_hip_roll_joint": 2,
    ".*_hip_yaw_joint": 2,
    ".*_knee_joint": 4,
    ".*_ankle_pitch_joint": 2,
    ".*_ankle_roll_joint": 1,
    "waist_roll_joint": 10,
    "waist_pitch_joint": 10,
    "waist_yaw_joint": 10,
    ".*_shoulder_pitch_joint": 2,
    ".*_shoulder_roll_joint": 2,
    ".*_shoulder_yaw_joint": 1,
    ".*_elbow_joint": 1,
    ".*_wrist_roll_joint": 1,
    ".*_wrist_pitch_joint": 1,
    ".*_wrist_yaw_joint": 1,
}

scale_dict = {
    ".*_hip_yaw_joint": 1.0,
    ".*_hip_roll_joint": 1.0,
    ".*_hip_pitch_joint": 1.0,
    ".*_knee_joint": 1.0,
    ".*_ankle_pitch_joint": 1.0,
    ".*_ankle_roll_joint": 1.0,
    "waist_roll_joint": 1.0,
    "waist_pitch_joint": 1.0,
    "waist_yaw_joint": 1.0,
    ".*_shoulder_pitch_joint": 1.0,
    ".*_shoulder_roll_joint": 1.0,
    ".*_shoulder_yaw_joint": 1.0,
    ".*_elbow_joint": 1.0,
    ".*_wrist_roll_joint": 1.0,
    ".*_wrist_pitch_joint": 1.0,
    ".*_wrist_yaw_joint": 1.0,
}

joint_pos_config = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_pitch_joint": 0.2,
}


def _get_by_pattern(joint_name, table):
    for p, v in table.items():
        if p.startswith(".*"):
            if joint_name.endswith(p[3:]):
                return v
        elif joint_name == p:
            return v
    raise ValueError(joint_name)


kps = np.array([_get_by_pattern(n, stiffness_dict) for n in mujoco_joint_names], np.float32)
kds = np.array([_get_by_pattern(n, damping_dict) for n in mujoco_joint_names], np.float32)
action_scale = np.array([_get_by_pattern(n, scale_dict) for n in mujoco_joint_names], np.float32)
default_angles = np.array(
    [
        _get_by_pattern(n, joint_pos_config)
        if any(n.endswith(k[3:]) for k in joint_pos_config if k.startswith(".*"))
        else joint_pos_config.get(n, 0.0)
        for n in mujoco_joint_names
    ],
    np.float32,
)

# ============================================================
# ======================= MotionLoader =======================
# ============================================================
def quat_apply_inverse_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector (numpy version).

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    # cross product for batches
    t = np.cross(xyz, vec) * 2
    result = (vec - quat[:, 0:1] * t + np.cross(xyz, t))
    result = result.reshape(shape)
    return result

class MotionLoader:
    def __init__(self, motion_file):
        data = np.load(motion_file)
        import pdb;pdb.set_trace()
        self.joint_pos = data["joint_pos"]
        self.joint_vel = data["joint_vel"]
        self.body_pos = data["body_pos_w"]
        self.body_ori = data["body_quat_w"]
        self.body_vel = data["body_lin_vel_w"]
        self.body_ang_vel = data["body_ang_vel_w"]
        self.fps = data["fps"]
        self.T = self.joint_pos.shape[0]
        self.body_pos[:, :, :2] -= self.body_pos[0, 0, :2]

        self.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.anchor_body_name = anchor_body.name.lower()
        self.anchor_body_index = anchor_body.value
        self.future_steps = 5
        self.root_pos = self.body_pos[:,self.anchor_body_index]
        self.root_ori = self.body_ori[:,self.anchor_body_index]
        self.root_vel = self.body_vel[:,self.anchor_body_index]
        # self.root_ang_vel = self.body_ang_vel[:,self.anchor_body_index]
        # self.root_vel =  self.body_vel[:,self.anchor_body_index]

        self.root_ang_vel = quat_apply_inverse_np(self.root_ori, self.body_ang_vel[:,self.anchor_body_index])
        self.root_vel = quat_apply_inverse_np(self.root_ori, self.body_vel[:,self.anchor_body_index])
        self.joint_pos[:,[5, 8, 25, 26, 27, 28]] = 0.0
        self.joint_vel[:,[5, 8, 25, 26, 27, 28]] = 0.0
# ============================================================
# ===================== Observation (原封) ===================
# ============================================================



def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion into a 9-dimensional rotation matrix."""
    q = np.outer(q, q)

    return np.array(
        [
            [
                q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                2 * (q[1, 2] - q[0, 3]),
                2 * (q[1, 3] + q[0, 2]),
            ],
            [
                2 * (q[1, 2] + q[0, 3]),
                q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                2 * (q[2, 3] - q[0, 1]),
            ],
            [
                2 * (q[1, 3] - q[0, 2]),
                2 * (q[2, 3] + q[0, 1]),
                q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
            ],
        ]
    )
    
FEET_ALL_SITES = [
    "left_foot",
    "right_foot",
    "left_foot_top",
    "right_foot_top",
]

OBS_KEYS = [
    "dif_joint_pos",
    "dif_joint_vel",
    "gvec_pelvis",
    "gyro_pelvis",
    "joint_pos",
    "joint_vel",
    "last_motor_targets",
    "ref_feet_height",
    "ref_root_angvel",
    "ref_root_height",
    "ref_root_linvel"
]

def compute_observation_opentrack(mj_model, sim_data, ref_data, last_motor_targets, proprio_history_buf):
    dof_pos = sim_data.qpos[7:]
    dof_vel = sim_data.qvel[6:]
    
    ref_dof_pos = ref_data.qpos[7:]
    ref_dof_vel = ref_data.qvel[6:]
    # import ipdb;ipdb.set_trace()
    sensor_id = mj_model.sensor("gyro_pelvis").id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    gyro_pelvis = sim_data.sensordata[sensor_adr : sensor_adr + sensor_dim]
    _pelvis_imu_site_id = mj_model.site("imu_in_pelvis").id
    gvec_pelvis = sim_data.site_xmat[_pelvis_imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])
    _feet_all_site_id = np.array([mj_model.site(name).id for name in FEET_ALL_SITES])
    ref_feet_height = ref_data.site_xpos[_feet_all_site_id, 2]
    dif_joint_pos = ref_dof_pos - dof_pos
    dif_joint_vel = ref_dof_vel - dof_vel
    traj_root_rot_mat = quat_to_mat(ref_data.qpos[3:7])

    state_dict = {
        "gyro_pelvis": gyro_pelvis * 0.05,
        "gvec_pelvis": gvec_pelvis,
        "joint_pos": (dof_pos - default_angles),
        "joint_vel": dof_vel * 0.05,
        "last_motor_targets": last_motor_targets,
        "dif_joint_pos": dif_joint_pos,
        "dif_joint_vel": dif_joint_vel * 0.05,
        "ref_feet_height": ref_feet_height,
        "ref_root_height": ref_data.qpos[2:3],
        "ref_root_linvel": (traj_root_rot_mat.T @ ref_data.qvel[:3]) * 0.05,
        "ref_root_angvel": ref_data.qvel[3:6] * 0.05,
    }
    state = np.hstack([state_dict[k] for k in OBS_KEYS])
    
    
    return torch.from_numpy(state).float().unsqueeze(0).cuda()


# ============================================================
# ======================= PolicyRunner =======================
# ============================================================


class PolicyRunner:
    def __init__(self, policy_path):
        self.session = ort.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

    def act(self, obs):
        obs = obs.reshape(1, -1).astype(np.float32)
        return self.session.run(None, {self.input_name: obs})[0].squeeze()

class PolicyRunnerOpentrack:
    def __init__(self, policy_path):
        self.policy_jit = torch.jit.load(policy_path, map_location='cuda')

    def act(self, obs):
        obs = obs.reshape(1, -1)
        with torch.no_grad():
            action = self.policy_jit(obs).cpu().numpy().squeeze()
        return action

# ============================================================
# ===================== MujocoTrackingSim ====================
# ============================================================


# ==========================
# MujocoTrackingSim (可重用不同 motion)
# ==========================
class MujocoTrackingSim:
    def __init__(self, xml_path, dt=0.002, decimation=10):
        self.xml_path = xml_path
        self.dt = dt
        self.decimation = decimation
        self.m = None
        self.d = None
        self.motion = None
        self.action = np.zeros(29, dtype=np.float32)
        self.last_motor_targets = np.zeros(29, dtype=np.float32)
        self.t = 0
        self.traj_root_trans = []
        self.traj_root_rot = []
        self.traj_dof = []
        self.history_len = 10
        self.n_obs_single = 35 + 3 + 2 + 3*29
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

    def load_motion(self, motion):
        self.motion = motion
        if self.m is None:
            xml_path = "/home/lenovo/OpenTrack/data/xmls/unitree_g1/scene_mjx_wholebody_flat_terrain.xml"
            if not isinstance(xml_path, str):
                xml_path = str(xml_path)
            spec = mujoco.MjSpec.from_file(xml_path)
            self.m = spec.compile()
            self.d = MjData(self.m)
            self.ref_mj_data = mujoco.MjData(self.m)
            self.m.opt.timestep = self.dt
        self.reset()

    def reset(self):
        self.t = 0
        self.traj_root_trans.clear()
        self.traj_root_rot.clear()
        self.traj_dof.clear()
        
        self.d.qpos[:3] = self.motion.root_pos[0]
        self.d.qpos[3:7] = self.motion.root_ori[0]
        self.d.qpos[7:] = self.motion.joint_pos[0][isaaclab_to_mujoco_reindex]
        self.d.qvel[:3] = self.motion.root_vel[0]
        self.d.qvel[3:6] = self.motion.root_ang_vel[0]
        self.d.qvel[6:] = self.motion.joint_vel[0][isaaclab_to_mujoco_reindex]
        self.ref_mj_data.qpos[:3] = self.motion.root_pos[0]
        self.ref_mj_data.qpos[3:7] = self.motion.root_ori[0]
        self.ref_mj_data.qpos[7:] = self.motion.joint_pos[0][isaaclab_to_mujoco_reindex]
        self.ref_mj_data.qvel[:3] = self.motion.root_vel[0]
        self.ref_mj_data.qvel[3:6] = self.motion.root_ang_vel[0]
        self.ref_mj_data.qvel[6:] = self.motion.joint_vel[0][isaaclab_to_mujoco_reindex]
        
        self.d.ctrl[:] = self.motion.joint_pos[0][isaaclab_to_mujoco_reindex]
        
        mj_forward(self.m, self.d)

        self.last_motor_targets = self.d.qpos[7:].copy()

    def get_obs(self):
        if self.finished():
            return None
        return compute_observation_opentrack(self.m, self.d, self.ref_mj_data, self.last_motor_targets, self.proprio_history_buf)

    def step_policy(self, action):
        self.ref_mj_data.qpos[:3] = self.motion.root_pos[self.t]
        self.ref_mj_data.qpos[3:7] = self.motion.root_ori[self.t]
        self.ref_mj_data.qvel[:3] = self.motion.root_vel[self.t]
        self.ref_mj_data.qvel[3:6] = self.motion.root_ang_vel[self.t]
        self.ref_mj_data.qpos[7:] = self.motion.joint_pos[self.t][isaaclab_to_mujoco_reindex]
        self.ref_mj_data.qvel[6:] = self.motion.joint_vel[self.t][isaaclab_to_mujoco_reindex]
        mujoco.mj_forward(self.m, self.ref_mj_data)
            
        lower_motor_targets = self.motion.joint_pos[self.t][isaaclab_to_mujoco_reindex] + action
        motor_targets = default_angles.copy()
        motor_targets = lower_motor_targets
        self.last_motor_targets = motor_targets.copy() 
        for _ in range(self.decimation):
            self.d.ctrl[:] = motor_targets
            mj_step(self.m, self.d)
        self.traj_root_trans.append(self.d.qpos[:3].copy())
        self.traj_root_rot.append(self.d.qpos[3:7].copy())
        self.traj_dof.append(self.d.qpos[7:].copy())
        self.t += 1

    def finished(self):
        return self.t >= self.motion.T

    def get_trajectory(self, fps=500):
        motion_data = {
            "root_trans_offset": np.stack(self.traj_root_trans, axis=0).astype(np.float32),
            # "root_rot": np.stack(self.traj_root_rot, axis=0).astype(np.float32),
            "root_rot": np.stack(self.traj_root_rot, axis=0)[:, [1,2,3,0]].astype(np.float32), # WXYZ => XYZW
            "dof": np.stack(self.traj_dof, axis=0).astype(np.float32),
            "fps": fps,
        }
        return {"motion_0": motion_data}


def run_sequential_rollout(motion_pairs, policy_path):
    """
    motion_pairs: list of (input_motion_path, output_motion_path)
    policy_path: policy onnx file
    """
    policy = PolicyRunnerOpentrack(policy_path)
    xml_path = "/home/lenovo/OpenTrack/data/xmls/unitree_g1/scene_mjx_wholebody_flat_terrain.xml"

    # 创建一个 sim
    sim = MujocoTrackingSim(xml_path, dt=0.002, decimation=10)

    # 过滤已处理的 motion
    tasks_to_run = [(inp, out) for inp, out in motion_pairs if not os.path.exists(out)]

    for motion_input_path, motion_output_path in tqdm(tasks_to_run, desc="Rollout"):
        motion = MotionLoader(motion_input_path)
        sim.load_motion(motion)

        while not sim.finished():
            obs = sim.get_obs()
            action = policy.act(obs)
            sim.step_policy(action)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(motion_output_path), exist_ok=True)
        joblib.dump(sim.get_trajectory(), motion_output_path)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="JSON file with motion_pairs and policy_path")
    args = parser.parse_args()

    # 读取配置文件
    with open(args.input_file, "r") as f:
        cfg = json.load(f)

    motion_pairs = cfg["motion_pairs"]
    policy_path = cfg["policy_path"]

    run_sequential_rollout(motion_pairs, policy_path)
