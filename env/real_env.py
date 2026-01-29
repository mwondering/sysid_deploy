import numpy as np
import time
import os
from threading import Thread
from termcolor import colored

from common.command_helper_lcm import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.remote_controller import RemoteController, KeyMap
from root_states_listener import RootStateListener
from object_states_listener import ObjectStateListener
from common.bydmimic_utils import *
from env.base_env import BaseEnv
from scipy.spatial.transform import Rotation as sRot

import lcm
from lcm_types import low_cmd_lcmt, low_state_lcmt, sport_state_lcmt

from common.math_np import (
    yaw_quat,
    quat_mul,
    quat_inv,
)

USE_PRINT = True

class RealEnv(BaseEnv):
    
    def __init__(self, lcm_url: str = "udpm://239.255.76.68:7667?ttl=255"):
        # LCM setup
        self.lc = lcm.LCM(lcm_url)
        self.control_dt = 0.02
        
        # Robot state
        self._is_alive = True
        self.remote_controller = RemoteController()
        self.num_actions = 29
        
        self.low_cmd = low_cmd_lcmt()
        self.low_state = low_state_lcmt()
        self.sport_state = sport_state_lcmt()
        self._tick = self.low_state.tick
        
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 5  # 5: free waist, 6: fixed waist
        
        self.lowcmd_topic = "low_cmd_topic"
        self.lowstate_topic = "low_state_topic"
        self.sportstate_topic = "sport_state_topic"
        
        # Default robot parameters
        self.default_dof_pos = default_angles.copy()
        self.kps = kps
        self.kds = kds
        
        # Safety limits
        self.max_joint_delta = 4.5  # Maximum joint position change
        self.max_joint_velocity = 35.0  # Maximum joint velocity
        
        # Setup LCM communication
        self.lowstate_subscriber = self.lc.subscribe(self.lowstate_topic, self.receive_state_handler)
        self.sportstate_subscriber = self.lc.subscribe(self.sportstate_topic, self.receive_sport_state_handler)
        self.lowcmd_publisher = lambda cmd: self.lc.publish(self.lowcmd_topic, cmd.encode())
        # 这要改一下？
        # self.rootstate_listener = RootStateListener("192.168.123.164")
        # self.objstate_listener = ObjectStateListener("192.168.123.164","192.168.123.164")
        # 7438
        self.rootstate_listener = RootStateListener("192.168.155.211")
        self.objstate_listener = ObjectStateListener("192.168.155.211","192.168.155.211")
        # 4531
        # self.rootstate_listener = RootStateListener("192.168.155.190")
        # self.objstate_listener = ObjectStateListener("192.168.155.190","192.168.155.190")        
        
        #第三台
        # self.rootstate_listener = RootStateListener("192.168.155.190")
        # self.objstate_listener = ObjectStateListener("192.168.155.190","192.168.155.190")          
        
        # Start LCM poller thread
        self._poller = Thread(target=self.lcm_poller)
        self._poller.daemon = True
        self._poller.start()
        
        # Wait for robot connection
        self._wait()
        time.sleep(1)
        # self.align_lio_by_imu()
        self.align_imu_by_lio()
        # Initialize command
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        
        print("Real robot initialized successfully")
    
    def lcm_poller(self):
        """LCM message polling thread"""
        try:
            while self._is_alive:
                self.lc.handle()
        except Exception as e:
            print("LCM Poller Exception: ", e)
            self.safe_exit()
    
    def receive_state_handler(self, channel: str, msg: bytes):
        """Handle incoming robot state messages"""
        if not self._is_alive:
            return
        try:
            self.low_state = low_state_lcmt.decode(msg)
            self.mode_machine_ = self.low_state.mode_machine
            self.remote_controller.set(self.low_state.wireless_remote)
            self._tick = self.low_state.tick
            
            self.check_safety()
            self._low_state_callback()
            
        except Exception as e:
            print("Receive State Handler Exception: ", e)
            self.safe_exit()
            
    def receive_sport_state_handler(self, channel: str, msg: bytes):
        self.sport_state = sport_state_lcmt.decode(msg)
        pass
    
    def _low_state_callback(self):
        """Handle low-level state updates"""
        # Emergency stop with select button
        if self.remote_controller.button[KeyMap.select] == 1:
            print("Emergency stop triggered!")
            self.safe_exit()
        if self.remote_controller.button[KeyMap.B] == 1:
            print("Emergency stop triggered!")
            self.safe_exit()
    
    def check_safety(self):
        """Check safety constraints"""
        for motor_idx in range(len(self.low_cmd.q)):
            # Check joint position delta
            joint_delta = abs(self.low_cmd.q[motor_idx] - self.low_state.q[motor_idx])
            if joint_delta > self.max_joint_delta:
                if USE_PRINT:
                    print(colored(
                        f"SAFETY: Joint {motor_idx} position delta too large: {joint_delta:.3f} > {self.max_joint_delta}",
                        "white", "on_red"
                    ))
                self.safe_exit()
                return
            
            # Check joint velocity
            joint_velocity = abs(self.low_state.dq[motor_idx])
            if joint_velocity > self.max_joint_velocity:
                if USE_PRINT:
                    print(colored(
                        f"SAFETY: Joint {motor_idx} velocity too high: {joint_velocity:.3f} > {self.max_joint_velocity}",
                        "white", "on_red"
                    ))
                self.safe_exit()
                return
    
    def send_command(self):
        """Send command to robot after safety check"""
        # return 
        self.check_safety()
        if self._is_alive:
            self.lowcmd_publisher(self.low_cmd)
    
    def _wait(self):
        print("Waiting for robot connection...")
        while self.low_state.tick == 0:
            time.sleep(0.1)            
        if self.mode_machine_ != 5:
            raise ValueError(f"Invalid mode machine: {self.mode_machine_} != 5\nCheck the waist locked/unlocked status.")
        print("Successfully connected to robot!")
        
    def align_lio_by_imu(self):
        root_state = self.rootstate_listener.get_latest_transform()
        align_pose = np.concatenate([np.array(root_state[:3]), np.array(self.low_state.quaternion)[[1, 2, 3, 0]]], axis=0)
        self.rootstate_listener.set_align_pose(align_pose)

    def align_imu_by_lio(self):
        root_state = self.rootstate_listener.get_latest_transform()
        print(root_state)
        lio_yaw_quat = yaw_quat(np.array(root_state[[6,3,4,5]]))
        imu_yaw_quat = yaw_quat(np.array(self.low_state.quaternion))
        self.imu_align_q = quat_mul(lio_yaw_quat, quat_inv(imu_yaw_quat))

    def get_env_data(self):
        root_state = self.rootstate_listener.get_latest_transform()
        # print(root_state)
        # import ipdb; ipdb.set_trace()
        # root_state[2] = self.sport_state.position[2] # 里程计
        # print(root_state[2])
        
        obj_state = self.objstate_listener.get_latest_transform()
        
        robot_rot = sRot.from_quat(root_state[3:7])
        obj_in_world_rot = robot_rot * sRot.from_quat(obj_state[3:7])
        obj_in_world_trans = robot_rot.apply(np.array(obj_state[:3])) + root_state[:3]

        env_data = {
            'joint_pos': np.array([self.low_state.q[i] for i in range(29)], dtype=np.float32),
            'joint_vel': np.array([self.low_state.dq[i] for i in range(29)], dtype=np.float32),
            'root_pos': np.array(root_state[:3],  dtype=np.float32) + np.array([0, 0, 0.05]),
            # 'root_pos': np.array(self.sport_state.position,  dtype=np.float32) + np.array([0, 0, 0.05]),
            'root_quat': np.array(quat_mul(self.imu_align_q, self.low_state.quaternion),  dtype=np.float32),
            'root_angular': np.array(self.low_state.gyroscope, dtype=np.float32),
            'tick': self.low_state.tick,
            'mode_machine': self.low_state.mode_machine,
            'rel_object_pos': np.array(obj_state[:3], dtype=np.float32),
            'rel_object_quat': np.array(obj_state[3:7], dtype=np.float32),
            'object_pos': np.array(obj_in_world_trans, dtype=np.float32),
            'object_quat': np.array(obj_in_world_rot.as_quat()[[3,0,1,2]], dtype=np.float32),
        }
        
        return env_data

    def get_joystick_val(self):
        return {
            "stick":{
                'lx': self.remote_controller.lx,
                'ly': -self.remote_controller.ly,
                'rx': self.remote_controller.rx,
                'ry': self.remote_controller.ry,
                'lt': 0.0, # TODO
                'rt': 0.0, # TODO
                },
            "button": {
                'R1': self.remote_controller.button[KeyMap.R1],
                'L1': self.remote_controller.button[KeyMap.L1],
                'start': self.remote_controller.button[KeyMap.start],
                'select': self.remote_controller.button[KeyMap.select],
                'R2': self.remote_controller.button[KeyMap.R2],
                'L2': self.remote_controller.button[KeyMap.L2],
                'F1': self.remote_controller.button[KeyMap.F1],
                'F2': self.remote_controller.button[KeyMap.F2],
                'A': self.remote_controller.button[KeyMap.A],
                'B': self.remote_controller.button[KeyMap.B],
                'X': self.remote_controller.button[KeyMap.X],
                'Y': self.remote_controller.button[KeyMap.Y],
                'up': self.remote_controller.button[KeyMap.up],
                'right': self.remote_controller.button[KeyMap.right],
                'down': self.remote_controller.button[KeyMap.down],
                'left': self.remote_controller.button[KeyMap.left],
                }
        }
        
    def reset(self):
        pass
    
    def is_alive(self):
        """Check if robot is alive and connected"""
        return self._is_alive
    
    def safe_exit(self):
        """Safely exit robot control"""
        print("Initiating safe robot exit...")
        self._is_alive = False
        
        # Send damping command
        try:
            create_damping_cmd(self.low_cmd)
            self.lowcmd_publisher(self.low_cmd)
            time.sleep(0.1)  # Give time for command to be sent
        except Exception as e:
            print(f"Error sending damping command: {e}")
        
        print("Robot safely exited")
    
    def should_run_control(self):
        return True
    
    def step(self, target_q, kps=None, kds=None):
        
        if not self._is_alive:
            return
        
        for i in range(29):
            self.low_cmd.q[i] = target_q[i]
            self.low_cmd.dq[i] = 0.0
            self.low_cmd.kp[i] = kps[i]
            self.low_cmd.kd[i] = kds[i]
            self.low_cmd.tau[i] = 0.0
        
        self.send_command()

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real Robot Interface Test')
    parser.add_argument('--test_mode', type=str, default='basic',
                       help='Test mode to run')
    args = parser.parse_args()
    
    robot = RealEnv()
    
    time.sleep(5)    
        
    robot.safe_exit()    
    print('Robot Exited')
        