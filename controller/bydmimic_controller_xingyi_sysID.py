import numpy as np
import onnxruntime as ort
from common.bydmimic_utils import *
from controller.base_controller import BaseController
from common.math_np import (
    subtract_frame_transforms, 
    matrix_from_quat,
    quat_apply,
    compute_transform_with_z_rotation,
    apply_pose_transform,
    quat_conjugate
)



import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

class BydMimicControllerXingyiSysID(BaseController):
    def __init__(self, policy_path: str):
        # Load policy

        self.env_cfg = load_env_config(policy_path)
        # import pdb; pdb.set_trace()
        self.obs_dim = 0
        self.obs_keys = []
        for obs_name in self.env_cfg.observations.policy:
            if isinstance(self.env_cfg.observations.policy[obs_name], dict):
                # import ipdb; ipdb.set_trace()
                self.obs_dim += obs_size_dict[obs_name]
                self.obs_keys.append(obs_name)
            pass

        self.session = ort.InferenceSession(policy_path)
        self.obs_name = self.session.get_inputs()[0].name
        # import pdb;pdb.set_trace()
        self.time_step_name = self.session.get_inputs()[1].name
        self.default_dof_pos = default_angles.copy()
        
        # import ipdb; ipdb.set_trace()
        # get max time step
        output = self.session.run(None, {self.obs_name: np.zeros(self.obs_dim, dtype=np.float32).reshape(1, -1), 
                                       self.time_step_name: np.array([[0]], dtype=np.float32)})
        # import pdb;pdb.set_trace()
        # import ipdb; ipdb.set_trace()
        print("output[-1] =", output[-1])
        print("output type:", type(output))
        print("output length:", len(output))
        print("output[-1] shape:", getattr(output[-1], 'shape', 'no shape'))
        self.max_time_step = int(output[-1])
        # Control variables
        self.num_actions = 29
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.joint_pos = np.zeros(29, dtype=np.float32)
        self.joint_vel = np.zeros(29, dtype=np.float32)
        self.yaw_diff = np.zeros(1, dtype=np.float32)
        self.yaw_threshold = 0.3 # 1e-6
        self.cur_goal_distance = np.zeros(1, dtype=np.float32)


        self.ref_ang_vel = np.zeros(3, dtype=np.float32)
        self.body_pos_w = np.zeros(3, dtype=np.float32)
        self.body_quat_w = np.zeros(4, dtype=np.float32)
        self.diff_ori = np.zeros(4, dtype=np.float32)
        self.diff_pos = np.zeros(3, dtype=np.float32)
        self.body_quat_w[0] = 1.0
        self.diff_ori[0] = 1.0

        self.tmp_list = []
        # Time step counter for policy
        self.time_step = 0
        
        # set kp, kd
        self.kps = kps
        self.kds = kds
        self.action_scale = action_scale



    def reset(self, env_data):
        self.action.fill(0)
        self.joint_pos.fill(0)
        self.joint_vel.fill(0)
        self.body_pos_w.fill(0)
        self.body_quat_w.fill(0)
        self.ref_ang_vel.fill(0)
        self.time_step = 0
        output = self.session.run(None, {self.obs_name: np.zeros(self.obs_dim, dtype=np.float32).reshape(1, -1), 
                                self.time_step_name: np.array([[0]], dtype=np.float32)})
        self.joint_pos = output[1].squeeze()
        self.body_pos_w = output[3].squeeze()[0]
        self.body_quat_w = output[4].squeeze()[0]
        
        world_anchor = np.concatenate((self.body_pos_w, self.body_quat_w), axis=0)
        robot_pelvis_pos = env_data['root_pos']
        robot_pelvis_quat = env_data['root_quat']

        init_anchor = np.concatenate((robot_pelvis_pos, robot_pelvis_quat), axis=0)
        self.T_align = compute_transform_with_z_rotation(world_anchor, init_anchor)



        # 1/0

    def _pre_compute_observations_callback(self, env_data):
        robot_pelvis_pos = env_data['root_pos'].reshape(1,3)
        robot_pelvis_quat = env_data['root_quat'].reshape(1,4)
        motion_pos = np.concatenate((self.body_pos_w, self.body_quat_w), axis=0)
        align_pose = apply_pose_transform(motion_pos, self.T_align[0], self.T_align[1])
        align_pos, align_quat = align_pose[:3], align_pose[3:7]
        self._obs_motion_anchor_pos_b , diff_ori = subtract_frame_transforms(robot_pelvis_pos, robot_pelvis_quat,
                                                       align_pos[np.newaxis], align_quat[np.newaxis])
        mat = matrix_from_quat(diff_ori)
        self._obs_motion_anchor_ori_b = mat[..., :2].reshape(1, -1)
        self._obs_projected_gravity = get_gravity_orientation(robot_pelvis_quat[0])
        self._obs_root_angular = env_data['root_angular']
        self._obs_joint_pos = (env_data['joint_pos'] - default_angles)[mujoco_to_isaaclab_reindex]
        self._obs_joint_vel = env_data['joint_vel'][mujoco_to_isaaclab_reindex]
        self._obs_ref_motion_phase = np.array([self.time_step / self.max_time_step])


        pass



    def step(self, env_data):
        """
        Compute target DOF positions from mujoco data and time step
        
        Args:
            mujoco_data: MuJoCo data object containing current simulation state
            time_step: Current time step
            
        Returns:
            target_dof_pos: Target joint positions (29,) array
        """


        self._pre_compute_observations_callback(env_data)
        
        # Create observation
        obs_buf = []
        # print("--------------------------------")
        for obs_name in self.obs_keys:
            actor_obs = getattr(self, f"_get_obs_{obs_name}")()
            # print("obs_name:", obs_name, "actor_obs.shape:", actor_obs.shape)
            obs_buf.append(actor_obs)
            # print(actor_obs.shape)
        obs = np.concatenate(obs_buf, axis=-1, dtype=np.float32).reshape(1, -1)
        
        # Prepare inputs for ONNX model
        time_step = np.array([[self.time_step]], dtype=np.float32)
        # Run inference
        # import pdb;pdb.set_trace()s
        output = self.session.run(None, {self.obs_name: obs, 
                                       self.time_step_name: time_step})
        
        # Update state variables
        self.action = output[0].squeeze()
        self.joint_pos = output[1].squeeze()
        self.joint_vel = output[2].squeeze()
        self.body_pos_w = output[3].squeeze()[0]
        self.body_quat_w = output[4].squeeze()[0]
        self.ref_ang_vel = output[6].squeeze()[7]
        
        # Transform action to target joint positions
        target_q = self.action[isaaclab_to_mujoco_reindex] * action_scale + default_angles
        self.time_step += 1    
        
        return target_q, self.kps, self.kds
    




    # =========== obs ============
    
    def _get_obs_command(self):
        return np.concatenate((self.joint_pos, self.joint_vel),axis=0)

    def _get_obs_motion_anchor_pos_b(self):
        return self._obs_motion_anchor_pos_b.squeeze()

    def _get_obs_motion_anchor_ori_b(self):
        return self._obs_motion_anchor_ori_b.squeeze()

    def _get_obs_base_ang_vel(self):
        return self._obs_root_angular

    def _get_obs_joint_pos(self):
        return self._obs_joint_pos

    def _get_obs_joint_vel(self):
        return self._obs_joint_vel

    def _get_obs_actions(self):
        return self.action

    def _get_obs_projected_gravity(self):
        return self._obs_projected_gravity

    def _get_obs_motion_ref_ang_vel(self):
        return self.ref_ang_vel

    def _get_obs_ref_motion_phase(self):
        return self._obs_ref_motion_phase