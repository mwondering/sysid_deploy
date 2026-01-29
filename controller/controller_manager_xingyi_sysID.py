from env.base_env import BaseEnv
from controller.base_controller import BaseController
from controller.bydmimic_controller import BydMimicController
from controller.default_pos_controller import DefaultPosController
from controller.zero_torque_controller import ZeroTorqueController
from controller.amp_controller import AMPController
from typing import List
import numpy as np

class ControllerManagerXingyiSysID:
    def __init__(self, env: BaseEnv):
        self.env = env
        self.cur_controller: BaseController = None
        self.cur_idx = 0
        self.motion_controller_list: List[BaseController] = []
        self.zero_torque_controller = ZeroTorqueController()
        self.default_pos_controller = DefaultPosController()
        
        amp_policy_path = 'ckpts/amp_slope_walk_stand.onnx'
        self.amp_controller = AMPController(policy_path=amp_policy_path)
        
        self.tick = 0
    
    def add_controller(self, controller):
        self.motion_controller_list.append(controller)
    
    def set_controller(self, controller: BaseController):
        self.cur_controller = controller
        
    def set_controller_by_id(self, curr_id: int):
        self.cur_controller = self.motion_controller_list[curr_id]
    
    def reset_all_controllers(self):
        for controller in self.motion_controller_list:
            controller.reset(self.env.get_env_data())
        self.amp_controller.reset()
        self.set_controller(self.zero_torque_controller)
    
    def check_motion_end(self):
        if self.cur_controller is not None and (isinstance(self.cur_controller, BydMimicController) ):
            if self.cur_controller.time_step >= self.cur_controller.max_time_step - 1:
                print('Motion ended, switched to default policy')
                self.cur_idx = 0
                self.set_controller(self.amp_controller)

    @property
    def num_motion_controllers(self):
        return len(self.motion_controller_list)

    def check_state(self):
        self.check_motion_end()
        joystick_data = self.env.get_joystick_val()
        
        cmd = np.array([0., 0., 0.], dtype=np.float32)
        lx, ly, rx = joystick_data['stick']['lx'], joystick_data['stick']['ly'], joystick_data['stick']['rx']
        cmd[0] = -ly
        cmd[1] = -lx
        cmd[2] = -rx
        self.cur_controller.set_cmd(cmd)
        
        if not isinstance(self.cur_controller, ZeroTorqueController) and joystick_data["button"]["X"] == 1:
            print('Button X pressed - enter zero torque')
            self.set_controller(self.zero_torque_controller)
            return 
        if isinstance(self.cur_controller, ZeroTorqueController) and joystick_data["button"]["start"] == 1:
            print('Button start pressed - enter default pose')
            self.default_pos_controller.reset()
            self.set_controller(self.default_pos_controller)
            return
        if isinstance(self.cur_controller, DefaultPosController) and joystick_data["button"]["A"] == 1:
            print('Button A pressed - enter AMP policy')
            self.amp_controller.reset()
            self.set_controller(self.amp_controller)
            return
        if isinstance(self.cur_controller, AMPController) and joystick_data["button"]["up"] == 1:
            print(isinstance(self.cur_controller, AMPController))
            print('Button up pressed - enter motion policy')
            # target_anchor_pos, global_destination_pos = self.motion_controller_list[self.cur_idx].reset(self.env.get_env_data())
            # print("motion ID:", self.motion_controller_list[self.cur_idx])
            self.set_controller(self.motion_controller_list[self.cur_idx])
            # self.env.set_object_pos([target_anchor_pos[0], target_anchor_pos[1], target_anchor_pos[2], 0, 0, 0, 1]) # TODO: hard code
            # self.env.get_marker_manager().update_position(name='destination', pos=global_destination_pos)
            return 
        if isinstance(self.cur_controller, BydMimicController) and joystick_data["button"]["down"] == 1:
            print('Button down pressed - enter AMP policy')
            self.amp_controller.reset()
            self.set_controller(self.amp_controller)
            return
        env_tick_time = self.env.get_tick_time()
        if env_tick_time - self.tick >= 500:            
            if joystick_data["button"]["right"] == 1:
                self.tick = env_tick_time
                if self.num_motion_controllers > 0:
                    self.cur_idx = (self.cur_idx + 1) % self.num_motion_controllers
                    print('cur_idx:', self.cur_idx)
            if joystick_data["button"]["left"] == 1:
                self.tick = env_tick_time
                if self.num_motion_controllers > 0:
                    self.cur_idx = (self.cur_idx - 1) % self.num_motion_controllers
                    print('cur_idx:', self.cur_idx)

    def set_zero_torque(self):
        self.set_controller(self.zero_torque_controller)