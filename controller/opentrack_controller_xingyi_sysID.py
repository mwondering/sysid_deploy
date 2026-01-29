import numpy as np
import onnxruntime as ort
# from common.bydmimic_utils import *
from common.opentrack_constants import *
from controller.base_controller import BaseController
from common.math_np import (
    subtract_frame_transforms, 
    matrix_from_quat,
    quat_apply,
    compute_transform_with_z_rotation,
    apply_pose_transform,
    quat_conjugate
)

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
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
class OpentrackControllerXingyiSysID(BaseController):
    def __init__(self, config_path: str,policy_path: str):
        # Load policy
        task_config = default_config()
        self.env_cfg = task_config.env_config
        self.config_path = config_path
        with open(config_path, "r") as f:
            config = json.load(f)
        self.env_cfg.update(config["env_config"])
        self.obs_dim = 0
        self.obs_keys = []
        for obs_name in self.env_cfg.obs_keys:
            self.obs_dim += OBS_SIZE_DICT[obs_name]
            self.obs_keys.append(obs_name)
        print("计算得到观测维度:", self.obs_dim)
        self.session = ort.InferenceSession(policy_path)
        self.obs_name = self.session.get_inputs()[0].name
        self.time_step_name = self.session.get_inputs()[1].name
        output = self.session.run(None, {self.obs_name: np.zeros(self.obs_dim, dtype=np.float32).reshape(1, -1), 
                                       self.time_step_name: np.array([[0]], dtype=np.int64)})
        print("output[-1] =", output[-1])
        print("output type:", type(output))
        print("output length:", len(output))
        print("output[-1] shape:", getattr(output[-1], 'shape', 'no shape'))
        self.max_time_step = int(output[-1])
        # Control variables
        self.num_actions = 29
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_qpos = np.zeros(36, dtype=np.float32)
        self.motion_qvel = np.zeros(35, dtype=np.float32)
        self.motion_site_xpos = np.zeros((31, 3), dtype=np.float32)
        # Time step counter for policy
        self.time_step = 0
        
        # set kp, kd
        self.kps = KPs
        self.kds = KDs
        self.action_scale = self.env_cfg.action_scale
        # 1/0


    def reset(self, env_data):
        output = self.session.run(None, {self.obs_name: np.zeros(self.obs_dim, dtype=np.float32).reshape(1, -1), 
                                       self.time_step_name: np.array([[0]], dtype=np.int64)})
        self.action = output[0].squeeze()
        self.motion_qpos = output[1].squeeze()
        self.motion_qvel = output[2].squeeze()
        self.motion_site_xpos = output[3].squeeze()
        self.time_step = 0
        self.last_motor_targets = np.array([-0.20315   ,  0.188144  ,  0.33371699,  0.46765599, -0.295228  ,
       -0.045593  , -0.21267   , -0.220919  , -0.32586601,  0.43477499,
       -0.31505501,  0.101499  , -0.002469  ,  0.        ,  0.        ,
       -0.174326  ,  1.67636395,  0.154378  ,  1.06195295,  0.025205  ,
        0.        ,  0.        , -0.166197  , -1.60932195, -0.122216  ,
        1.00633705, -0.17442299,  0.        ,  0.        ], dtype=np.float32)

    def compute_observation_opentrack(self,mj_model, sim_data):
        dof_pos = sim_data.qpos[7:]
        dof_vel = sim_data.qvel[6:]
        
        # import ipdb;ipdb.set_trace()
        sensor_id = mj_model.sensor("gyro_pelvis").id
        sensor_adr = mj_model.sensor_adr[sensor_id]
        sensor_dim = mj_model.sensor_dim[sensor_id]
        gyro_pelvis = sim_data.sensordata[sensor_adr : sensor_adr + sensor_dim]
        _pelvis_imu_site_id = mj_model.site("imu_in_pelvis").id
        gvec_pelvis = sim_data.site_xmat[_pelvis_imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])
        pre_obs_dict = {
            "gyro_pelvis": gyro_pelvis * 0.05,
            "gvec_pelvis": gvec_pelvis,
            "joint_pos": (dof_pos - np.array(DEFAULT_QPOS[7:])),
            "joint_vel": dof_vel * 0.05,
            # "last_motor_targets": last_motor_targets,
            # "dif_joint_pos": dif_joint_pos,
            # "dif_joint_vel": dif_joint_vel * 0.05,
            # "ref_feet_height": ref_feet_height,
            # "ref_root_height": ref_data.qpos[2:3],
            # "ref_root_linvel": (traj_root_rot_mat.T @ ref_data.qvel[:3]) * 0.05,
            # "ref_root_angvel": ref_data.qvel[3:6] * 0.05,
        }
        return pre_obs_dict
    def step(self, env_data):
        """
        Compute target DOF positions from mujoco data and time step
        
        Args:
            mujoco_data: MuJoCo data object containing current simulation state
            time_step: Current time step
            
        Returns:
            target_dof_pos: Target joint positions (29,) array
        """
        self.mj_model = env_data["mj_model"]
        self.mj_data = env_data["mj_data"]
        self.pre_obs_dict =  self.compute_observation_opentrack(self.mj_model, self.mj_data)
        
        # Create observation
        obs_buf = []
        # print("--------------------------------")
        for obs_name in self.obs_keys:
            # import pdb;pdb.set_trace()
            actor_obs = getattr(self, f"_get_obs_{obs_name}")()
            # print("obs_name:", obs_name, "actor_obs.shape:", actor_obs.shape)
            obs_buf.append(actor_obs)
            # print(actor_obs.shape)
        obs = np.concatenate(obs_buf, axis=-1, dtype=np.float32).reshape(1, -1)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # Prepare inputs for ONNX model+
        self.time_step += 1
        print(self.time_step)
        time_step = np.array([[self.time_step]], dtype=np.int64)
        # Run inference
        
        output = self.session.run(None, {self.obs_name: obs, 
                                       self.time_step_name: time_step})
        
        # Update state variables
        self.action = output[0].squeeze()
        #用于action观测的obs,对应的轨迹帧提取
        target_q = self.action * self.action_scale + self.motion_qpos[7:]
        # target_q = self.motion_qpos[7:]
        # target_q = self.action * self.action_scale 
        # target_q = np.array([-2.0751131e-01,
        #     1.8927042e-01,  3.3717939e-01,  4.7067657e-01, -2.9756886e-01,
        # -4.5981191e-02, -2.1732287e-01, -2.2189973e-01, -3.2392603e-01,
        #     4.3571496e-01, -3.1612945e-01,  1.0218049e-01, -1.1089288e-03,
        #     0.0000000e+00,  0.0000000e+00, -1.7148182e-01,  1.6644515e+00,
        #     1.7124164e-01,  1.0203608e+00,  4.5246001e-02,  0.0000000e+00,
        #     0.0000000e+00, -1.7178409e-01, -1.5849643e+00, -1.4988783e-01,
        #     9.6195900e-01, -2.1607061e-01,  0.0000000e+00,  0.0000000e+00],dtype=np.float32)
        self.motion_qpos = output[1].squeeze()
        self.motion_qvel = output[2].squeeze()
        self.motion_site_xpos = output[3].squeeze()
        # self.motion_qpos[[12, 15, 32, 33, 34, 35]] = 0.0
        # self.motion_qvel[[11, 14, 31,32, 33, 34]] = 0.0
        
        self.motion_qpos[[5, 8, 25, 26, 27, 28]] = 0.0
        self.motion_qvel[[5, 8, 25, 26, 27, 28]] = 0.0

        # self.motion_qpos[[5, 8]] = 0.0
        # self.motion_qvel[[5, 8]] = 0.0
        # Transform action to target joint positions
        self.last_motor_targets = target_q
            
        return target_q, self.kps, self.kds
    




    # =========== obs ============
    
    # def _get_obs_dif_joint_pos(self):
    #     return np.concatenate((self.joint_pos, self.joint_vel),axis=0)

    # def _get_obs_motion_anchor_pos_b(self):
    #     return self._obs_motion_anchor_pos_b.squeeze()

    # def _get_obs_motion_anchor_ori_b(self):
    #     return self._obs_motion_anchor_ori_b.squeeze()

    # def _get_obs_base_ang_vel(self):
    #     return self._obs_root_angular

    # def _get_obs_joint_pos(self):
    #     return self._obs_joint_pos

    # def _get_obs_joint_vel(self):
    #     return self._obs_joint_vel

    # def _get_obs_actions(self):
    #     return self.action

    # def _get_obs_projected_gravity(self):
    #     return self._obs_projected_gravity

    # def _get_obs_motion_ref_ang_vel(self):
    #     return self.ref_ang_vel

    # def _get_obs_ref_motion_phase(self):
    #     return self._obs_ref_motion_phase
    def _get_obs_dif_joint_pos(self):
        return (self.motion_qpos[7:] - self.mj_data.qpos[7:])
    def _get_obs_dif_joint_vel(self):
        return (self.motion_qvel[6:] - self.mj_data.qvel[6:])*0.05
    def _get_obs_gvec_pelvis(self):
        return self.pre_obs_dict["gvec_pelvis"]
    def _get_obs_gyro_pelvis(self):
        return self.pre_obs_dict["gyro_pelvis"]
    def _get_obs_joint_pos(self):
        return self.pre_obs_dict["joint_pos"]
    def _get_obs_joint_vel(self):
        return self.pre_obs_dict["joint_vel"]
    def _get_obs_last_motor_targets(self):
        return self.last_motor_targets
    def _get_obs_ref_feet_height(self):
        feet_site_ids = np.array([self.mj_model.site(name).id for name in FEET_ALL_SITES])
        ref_feet_height = self.motion_site_xpos[feet_site_ids, 2]
        return ref_feet_height
    def _get_obs_ref_root_angvel(self):
        return self.motion_qvel[3:6] * 0.05
    def _get_obs_ref_root_height(self):
        return self.motion_qpos[2:3]
    def _get_obs_ref_root_linvel(self):
        traj_root_rot_mat = quat_to_mat(self.motion_qpos[3:7])
        return (traj_root_rot_mat.T @ self.motion_qvel[:3]) * 0.05