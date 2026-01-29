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

class MotionTransformer:
    def __init__(self):
        # 注意：这里保持你的原始数据格式 [w, x, y, z]
        # self.motion_first_pos = np.array([0.0993, -0.1143, 0.8143], dtype=np.float32)
        # self.motion_first_quat = np.array([0.7203, 0.0317, 0.0270, -0.6924], dtype=np.float32)  # [w, x, y, z]
        
        
        
        
        # self.motion_target_anchor_pos = np.array([0.3394948, -1.7758863, 0.7453501], dtype=np.float32)






        self.motion_target_anchor_pos = np.array([0.3394948, -1.7758863,  0.7453501], dtype=np.float32) # np.zeros(3, dtype=np.float32)
        # TODO
        # 朝向没对齐 （重要）[ 0.7203,  0.0317,  0.0270, -0.6924]第一帧
        # 30FPS第一帧
        self.motion_first_pos = np.array([0.0993, -0.1143,  0.8143], dtype=np.float32)
        self.motion_first_quat = np.array([ 0.7203,  0.0317,  0.0270, -0.6924], dtype=np.float32)



        self.motion_target_anchor_pos = np.array([0.24349397, -1.6485329 ,  0.73984027], dtype=np.float32) # np.zeros(3, dtype=np.float32)
        # 50FPSkick-soccer-high-foot-50FPS-start_70第一帧
        self.motion_first_pos = np.array([0.12807685, -0.03481922,  0.5621515], dtype=np.float32)
        self.motion_first_quat = np.array([0.7825797 , -0.05606039, -0.10312551, -0.6113847], dtype=np.float32)



        self.motion_target_anchor_pos = np.array([0.48366493, -1.6199236 ,  0.5948607 ], dtype=np.float32) # np.zeros(3, dtype=np.float32)
        # 50FPSkick-soccer-high-foot-50FPS-start_70第一帧
        self.motion_first_pos = np.array([0.12750271, -0.03133325,  0.5630009], dtype=np.float32)
        self.motion_first_quat = np.array([0.7949083 , -0.05843739, -0.09494285, -0.59639907], dtype=np.float32)



        center = torch.tensor(self.motion_first_pos)
        base_target = torch.tensor(self.motion_target_anchor_pos)

        radius_vec_xy = base_target[:2] - center[:2]
        radius_norm = torch.linalg.norm(radius_vec_xy, dim=-1, keepdim=True)
        radius_norm_flat = radius_norm.squeeze(-1)
        safe_radius = radius_norm_flat.clamp_min(1e-6)

        arc_samples =  0.0 # TODO 手动调整这里


        base_angle = torch.atan2(radius_vec_xy[1], radius_vec_xy[0])
        delta_angle = arc_samples / safe_radius
        new_angle = base_angle + delta_angle

        new_xy = torch.stack([torch.cos(new_angle), torch.sin(new_angle)], dim=-1) * radius_norm

        new_target = base_target.clone()
        new_target[:2] = center[:2] + new_xy

        zero_mask = radius_norm_flat < 1e-6
        if zero_mask.any():
            new_target[zero_mask] = base_target[zero_mask]



        self.motion_target_anchor_pos = new_target.numpy()
        print(self.motion_target_anchor_pos)






    def transform_target_position(self, robot_pos, robot_quat):
        """
        计算当轨迹第一帧移动到新位置时的目标点新位置（世界坐标系下）
        
        参数:
            robot_pos: 新的第一帧位置 [x, y, z] (世界坐标系)
            robot_quat: 新的第一帧四元数 [w, x, y, z] (世界坐标系)
            
        返回:
            new_target_pos: 新的目标点位置 (世界坐标系)
        """
        # 将 [w, x, y, z] 格式转换为 scipy 使用的 [x, y, z, w] 格式
        original_quat_xyzw = np.array([
            self.motion_first_quat[1],  # x
            self.motion_first_quat[2],  # y  
            self.motion_first_quat[3],  # z
            self.motion_first_quat[0]   # w
        ])
        
        robot_quat_xyzw = np.array([
            robot_quat[1],  # x
            robot_quat[2],  # y
            robot_quat[3],  # z
            robot_quat[0]   # w
        ])
        
        # 创建旋转对象
        original_rot = R.from_quat(original_quat_xyzw)
        new_rot = R.from_quat(robot_quat_xyzw)
        
        # 计算相对旋转：从原始姿态到新姿态
        relative_rot = new_rot * original_rot.inv()
        
        # 计算原始目标点相对于原始第一帧的偏移向量
        relative_offset = self.motion_target_anchor_pos - self.motion_first_pos
        
        # 对偏移向量应用相对旋转
        rotated_offset = relative_rot.apply(relative_offset)
        
        # 计算新的目标点位置
        new_target_pos = robot_pos + rotated_offset
        
        return new_target_pos
    
    def transform_target_position_compact(self, robot_pos, robot_quat):
        """
        更紧凑的版本
        """
        # 直接转换四元数格式
        def wxyz_to_xyzw(quat):
            return np.array([quat[1], quat[2], quat[3], quat[0]])
        
        original_rot = R.from_quat(wxyz_to_xyzw(self.motion_first_quat))
        new_rot = R.from_quat(wxyz_to_xyzw(robot_quat))
        
        # 计算相对变换并应用
        relative_rot = new_rot * original_rot.inv()
        relative_offset = self.motion_target_anchor_pos - self.motion_first_pos
        rotated_offset = relative_rot.apply(relative_offset)
        
        return robot_pos + rotated_offset




class BydMimicController(BaseController):
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
        self.time_step_name = self.session.get_inputs()[1].name
        self.default_dof_pos = default_angles.copy()
        
        # import ipdb; ipdb.set_trace()
        # get max time step
        output = self.session.run(None, {self.obs_name: np.zeros(self.obs_dim, dtype=np.float32).reshape(1, -1), 
                                       self.time_step_name: np.array([[0]], dtype=np.float32)})
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

        self.transformer = MotionTransformer()
        self.robot_pos = np.array([0.0, 0.0, 0.8143], dtype=np.float32)
        self.robot_quat = np.array([1.0, 0.0, 0.0, -0.0], dtype=np.float32)  # [w, x, y, z]

        self.target_anchor_pos = self.transformer.transform_target_position(self.robot_pos, self.robot_quat)
        print(self.target_anchor_pos)


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

        self.after_kick = True

        self.localization_flag = 0
        self.real_robot_pos = np.zeros(3, dtype=np.float32)
        self.real_robot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        # 1/0


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

        # self.robot_pos = np.array([0.0, 0.0, 0.8143], dtype=np.float32)
        # self.robot_quat = np.array([1.0, 0.0, 0.0, -0.0], dtype=np.float32)  # [w, x, y, z]
        self.robot_pos = env_data['root_pos']
        self.robot_quat = env_data['root_quat']

        self.target_anchor_pos = self.transformer.transform_target_position(self.robot_pos, self.robot_quat)
        print("----",self.target_anchor_pos)
        import joblib
        joblib.dump(self.target_anchor_pos,"target_pos.pkl")

        self.after_kick = True

        self.localization_flag = 0
        self.real_robot_pos = np.zeros(3, dtype=np.float32)
        self.real_robot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
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
        # self._obs_root_linear = env_data['root_linear'] # be carefull to use root_linear
        self._obs_root_angular = env_data['root_angular']
        self._obs_joint_pos = (env_data['joint_pos'] - default_angles)[mujoco_to_isaaclab_reindex]
        self._obs_joint_vel = env_data['joint_vel'][mujoco_to_isaaclab_reindex]
        self._obs_ref_motion_phase = np.array([self.time_step / self.max_time_step])

        # 更新yaw和goal distance
        self._update_yaw_diff(env_data)
        self._update_goal_distance(env_data)
        pass

    def get_target_anchor_pos(self):
        return self.target_anchor_pos

    def step(self, env_data):
        """
        Compute target DOF positions from mujoco data and time step
        
        Args:
            mujoco_data: MuJoCo data object containing current simulation state
            time_step: Current time step
            
        Returns:
            target_dof_pos: Target joint positions (29,) array
        """

        env_data1 = env_data.copy()
        if self.localization_flag != 1:
            env_data1['root_pos'] = self.real_robot_pos
            env_data1['root_quat'] = self.real_robot_quat
            self.localization_flag += 1
        else:
            self.real_robot_pos = env_data['root_pos']
            self.real_robot_quat = env_data['root_quat']
            self.localization_flag = 0
        self._pre_compute_observations_callback(env_data1)
        
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
    
    def set_cmd(self, cmd):
        pass

    # def _compute_yaw_diff_raw(self) -> torch.Tensor:
    #     self._forward_reference = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    #     heading_reference = self._forward_reference.expand(self.num_envs, -1)
    #     robot_forward = quat_apply(self.robot_anchor_quat_w, heading_reference)
    #     robot_yaw = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])

    #     target_anchor_pos = self.target_position_after_curved + self._env.scene.env_origins
    #     direction = target_anchor_pos - self.robot_anchor_pos_w
    #     direction_xy = direction[:, :2]
    #     target_yaw = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])

    #     near_zero = torch.linalg.norm(direction_xy, dim=-1) < self.yaw_threshold
    #     target_yaw = torch.where(near_zero, robot_yaw, target_yaw)

    #     diff = target_yaw - robot_yaw
    #     diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    #     return diff.unsqueeze(-1)

    def _update_yaw_diff(self, env_data):
        robot_pos = np.asarray(env_data['root_pos'])
        robot_quat = np.asarray(env_data['root_quat'])

        forward_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        robot_forward = quat_apply(robot_quat, forward_ref)
        robot_yaw = np.arctan2(robot_forward[..., 1], robot_forward[..., 0])

        direction = self.target_anchor_pos - robot_pos
        direction_xy = direction[..., :2]
        target_yaw = np.arctan2(direction_xy[..., 1], direction_xy[..., 0])

        dist_xy = np.linalg.norm(direction_xy, axis=-1)
        # print("dist_xy:", dist_xy)
        target_yaw = np.where(dist_xy < self.yaw_threshold, robot_yaw, target_yaw)





        #################3
        ################
        if (dist_xy < self.yaw_threshold and self.after_kick) or not self.after_kick:
            # print("Reached target yaw threshold, locking yaw_diff to 0.")
            self.after_kick = False
            self.yaw_diff.fill(0.0)
            # print('pos',env_data['root_pos'],'quat',env_data['root_quat'],"    dist_xy:", dist_xy,"  yaw_diff:", self.yaw_diff)
            return



        diff = target_yaw - robot_yaw
        # import pdb; pdb.set_trace()
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        self.yaw_diff = np.atleast_1d(diff).astype(np.float32)
        # print('pos',env_data['root_pos'],'quat',env_data['root_quat'],"    dist_xy:", dist_xy,"  yaw_diff:", self.yaw_diff)

    # def _compute_goal_distance_raw(self) -> torch.Tensor:
    #     target_anchor_pos = self.target_position_after_curved + self._env.scene.env_origins
    #     direction = target_anchor_pos - self.robot_anchor_pos_w
    #     return torch.linalg.norm(direction, dim=-1, keepdim=True)
    def _update_goal_distance(self, env_data):
        robot_pos = np.asarray(env_data['root_pos'])
        target_pos = np.asarray(self.target_anchor_pos)
        direction = target_pos - robot_pos
        distance = np.linalg.norm(direction, axis=-1)
        self.cur_goal_distance = np.atleast_1d(distance).astype(np.float32)
        #################3
        # ################
        # if not self.after_kick:
            
        #     self.cur_goal_distance.fill(0.0)
            
        #     return


    # =========== obs ============
    
    def _get_obs_command(self):
        return np.concatenate((self.joint_pos, self.joint_vel),axis=0)

    def _get_obs_motion_anchor_pos_b(self):
        return self._obs_motion_anchor_pos_b.squeeze()

    def _get_obs_motion_anchor_ori_b(self):
        return self._obs_motion_anchor_ori_b.squeeze()

    def _get_obs_base_lin_vel(self):
        # TODO: print warnning
        return self._obs_root_linear

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