import numpy as np
from controller.base_controller import BaseController

class DefaultPosController(BaseController):
    def __init__(self):
        self.num_actions = 29
        self.num_obs = 154
        self.kps = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
                            100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
                            400.0, 400.0, 400.0, 
                            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0, 
                            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0],dtype=np.float32)
        self.kds = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
                            2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
                            5.0, 5.0, 5.0, 
                            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 
                            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,],dtype=np.float32)
        self.action_scale = 0.25
        self.default_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                0.0, 0.0, 0.0,
                                0.3, 0.18, 0.0, 0.8, 0.0, 0.0, 0.0,
                                0.3, -0.18, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)

        self.dt = 0.02
        self.time_step = 0
        self.duration = 2 # time to move to default pos
        self.max_steps = int(self.duration / self.dt)
        self.init_pos = None
        self.is_init = False
        
    def reset(self, env_data=None):
        self.time_step = 0
        self.init_pos = None
        self.is_init = False

    def move_to_default_pos(self, env_data):
        print(f"Moving to default position {self.time_step}/{self.max_steps}")
        
        if not self.is_init:
            self.init_pos = np.zeros(29, dtype=np.float32)
            for i in range(29):
                self.init_pos[i] = env_data['joint_pos'][i]
            self.is_init = True
        
        alpha = self.time_step / self.max_steps
        target_q = np.zeros(29, dtype=np.float32)
        for i in range(29):
            target_q[i] = self.init_pos[i] + alpha * (self.default_dof_pos[i] - self.init_pos[i])

        return target_q, self.kps, self.kds
    
    def keep_default_pos(self):
        target_q = self.default_dof_pos.copy()
        return target_q, self.kps, self.kds
        
    def step(self, env_data):
        self.time_step += 1
        if self.time_step < self.max_steps:
            return self.move_to_default_pos(env_data)
        else:
            return self.keep_default_pos()
