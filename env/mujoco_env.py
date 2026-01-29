import mujoco
import mujoco.viewer
from common.bydmimic_utils import *
import numpy as np
import joblib
import os
from datetime import datetime
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from env.base_env import BaseEnv
from common.joystick_controller import JoyStickController

class MujocoEnv(BaseEnv):
    """MuJoCo Env - handless simulation, PD control, and rendering"""
    def __init__(self, 
                 xml_path: str='sysid_xmls/mjcf/scene_mjx_wholebody_flat_terrain.xml', 
                 dt: float = 0.005, 
                 control_decimation: int = 4,
                 object_name: str='',
                 use_log: bool=False,
                 view_motion: bool=False,
                 is_render: bool=True):
        # Initialize MuJoCo
        self.xml_path = xml_path
        self.joystick_pc = JoyStickController()
        self.control_dt = dt * control_decimation
        self.is_render = is_render
        if self.joystick_pc.is_connected():
            self.has_joystick = True
        else:
            self.has_joystick = False
        
        self.load_object = False
        if object_name.strip():
            with open(self.xml_path, "rb") as f:
                robot_xml_data = f.read()
            new_string = self.xml_add_obj(robot_xml_data, object_name)
            self.xml_path = f"source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/robot_obj.xml"
            open(self.xml_path, "wb").write(new_string)
            self.load_object = True
        # import ipdb; ipdb.set_trace()
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.keyboard_state = {
            "button": {
                'A': 0,
                'start': 0,
                'X': 0,
                'Y': 0,
                'up': 0,
                'right': 0,
                'down': 0,
                'left': 0,
                
                "cmd_forward": 0,
                "cmd_back": 0,
                "cmd_left": 0,
                "cmd_right": 0,
                "cmd_q": 0,
                "cmd_e": 0,
            }
        }
        self.mapping = {
            ".": "start",
            "\\": "up",
            "[": "left",
            "]": "right",
            "P": "down",
            ",": "A",
            "/": "X",
            
            "ĉ": "cmd_forward",
            "Ĉ": "cmd_back",
            "ć": "cmd_left",
            "Ć": "cmd_right",
            ";" : "cmd_q",
            "'" : "cmd_e",
        }
        if is_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_call_back)

        self.default_dof_pos = default_angles.copy()
        
        # Control parameters
        self.dt = dt
        self.control_decimation = control_decimation
        
        # Counters
        self.sim_step_counter = 0
        
        # Control state
        self.is_running = False
        self.view_motion = view_motion
        self.pause = False
        self.log = {
            'time_step':[],
            'target_q':[],
            'q':[],
            'dq':[],
            'tau':[]
        }
        self.use_log = use_log
    
    # def xml_add_obj(self, xml, obj_name, init_obj_pose=[-0.269, -0.25, 0.191, 1,0,0,0]): # TODO: hard code
    def xml_add_obj(self, xml, obj_name, init_obj_pose=[1.0, 0.0, 0.10, 1,0,0,0]): # TODO: hard code
        # 0.131,-1.212,0.931,-0.077,0.965,0.250,0.025 // mocap 29
        # -0.269, -0.25, 0.191, -0.057, 0.967, 0.244, 0.049 //mocap 28
        # import ipdb; ipdb.set_trace()
        parser = XMLParser(remove_blank_text=True)
        tree = parse(BytesIO(xml), parser=parser)
        worldroot = tree.getroot().find("worldbody")
        obj_node = Element("body", {"name": obj_name, "pos": f"{init_obj_pose[0]} {init_obj_pose[1]} {init_obj_pose[2]}",
                                    "quat": f"{init_obj_pose[3]} {init_obj_pose[4]} {init_obj_pose[5]} {init_obj_pose[6]}"})
        table_node = Element("body", {"name": "table", "pos": f"{init_obj_pose[0]} {init_obj_pose[1]} {init_obj_pose[2]+5}", "quat": "0.707 0.707 0 0"})
        scale = 0.1 # 1.05
        SubElement(
            obj_node,
            "joint",
            {"limited": "false", "name": obj_name, "type": "free"},
        )
        SubElement(
            obj_node,
            "geom",
            {
                "type": "mesh",
                # "mesh": "largebox_real_close",    # TODO: hard code
                "mesh": "ImageToStl.com_Ball+OBJ",
                "contype": "1",
                "conaffinity": "1",
                "friction": "1.0 0.01 0.0001",
                "density": "10",
                # "stiffness": "30000",
                # "damping": "1000",
                "rgba": "0.7 0.5 0.3 1",
                # "rgba": "0.8 0.6 .4 1",
            },
        )
        # SubElement(
        #     table_node,
        #     "geom",
        #     {
        #         "type": "mesh",
        #         "mesh": "table",
        #         "contype": "1",
        #         "conaffinity": "1",
        #         "friction": "0.9 0.01 0.0001",
        #         "density": "500",  # 桌子通常密度较大，避免移动
        #         "rgba": "1 1 1 1",  # 木头色
        #     },
        # )
        SubElement(
            obj_node,
            "geom",
            {
                "type": "mesh",
                "mesh": "ImageToStl.com_Ball+OBJ",

                # 碰撞
                "contype": "1",
                "conaffinity": "1",

                # 足球摩擦
                "friction": "0.8 0.01 0.0001",

                # 等效真实足球质量
                "density": "50",

                # 弹性 / 接触参数（关键）
                "solref": "0.02 1",
                "solimp": "0.9 0.95 0.01",

                # 视觉
                "rgba": "1 1 1 1",
            },
        )
        worldroot.append(obj_node)
        worldroot.append(table_node)

        asset = tree.getroot().find("asset")
        # asset.append(
        #     Element(
        #         "mesh",
        #         {
        #             "file": f"../objects/mocap/meshes/{obj_name}/ImageToStl.com_Ball+OBJ.stl",
        #             # "file": f"../objects/mocap/meshes/{obj_name}/{obj_name}_real_close.stl",
        #             "scale": f"{scale} {scale} {scale}",
        #         },
        #     )
        # )
        asset.append(
            Element(
                "mesh",
                {
                    "name": "ImageToStl.com_Ball+OBJ",  # 强烈建议显式命名
                    "file": f"../objects/mocap/meshes/{obj_name}/ImageToStl.com_Ball+OBJ.stl",
                    "scale": f"{scale} {scale} {scale}",
                },
            )
        )
        asset.append(
        Element(
                "mesh",
                {
                    "name": "table",
                    "file": f"../objects/mocap/meshes/table/table.stl",
                },
            )
        )
        return etree.tostring(tree)
    
    def reset(self, object_pose=[0.5,0,0.1, 1,0,0,0]):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:7+29] = self.default_dof_pos
        if self.load_object:
            self.data.qpos[7+29 : 7+29+3] = object_pose[:3]
            self.data.qpos[7+29+3:7+29+7] = object_pose[3:7]
        # mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset counters
        self.sim_step_counter = 0
    
    def set_object_pos(self, object_pose):
        self.data.qpos[7+29 : 7+29+3] = object_pose[:3]
        self.data.qpos[7+29+3:7+29+7] = object_pose[3:7]
        self.sim_step_counter = 0
    
    def step(self, target_q, kps, kds):
        """Advance simulation by one step"""
        tau = pd_control(target_q, self.data.qpos[7:7+29], kps, 
                        np.zeros_like(kds), self.data.qvel[6:6+29], kds)
        # Get actuator force range limits from XML and clip torques
        tau_min = self.model.actuator_forcerange[:, 0]
        tau_max = self.model.actuator_forcerange[:, 1]
        tau = np.clip(tau, tau_min, tau_max)
        # Apply control torques
        self.data.ctrl[:] = tau
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        if self.use_log:
            self.log['time_step'].append(self.sim_step_counter)
            self.log['target_q'].append(target_q.copy())
            self.log['q'].append(self.data.qpos[7:7+29].copy())
            self.log['dq'].append(self.data.qvel[6:6+29].copy())
            self.log['tau'].append(tau.copy())

        self.sim_step_counter += 1
        self.render()
    
    def view_motion_step(self, motion):
        self.data.qpos[:3] = motion['root_pos']
        self.data.qpos[3:7] = motion['root_quat']
        self.data.qpos[7:7+29] = motion['joint_pos']
        if self.load_object:
            self.data.qpos[7+29 : 7+29+3] = motion['object_pos']
            self.data.qpos[7+29+3:7+29+7] = motion['object_quat']
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        self.sim_step_counter += 1
        self.render()
        
    def get_env_data(self):
        # 获取机器人torso的位姿
        link_name = "pelvis"  # 替换为你的link名称
        link_id = self.model.body(link_name).id
        link_pos = self.data.body(link_id).xpos.copy()
        link_quat = np.zeros(4)
        mujoco.mju_mat2Quat(link_quat, self.data.body(link_id).xmat)
        link_lin_vel = self.data.body(link_id).cvel[:3].copy()
        link_ang_vel = self.data.body(link_id).cvel[3:6].copy()

    

        root_pos = self.data.qpos[:3].copy()
        # root_pos[2] = 0.84
        env_data = {
            'joint_pos': np.array(self.data.qpos[7:7+29].copy(), dtype=np.float32),
            'joint_vel': np.array(self.data.qvel[6:6+29].copy(), dtype=np.float32),
            'root_pos': np.array(root_pos, dtype=np.float32), # + np.random.normal(0, 0.25, size=3)
            'root_quat': np.array(self.data.qpos[3:7].copy(), dtype=np.float32),
            'root_linear': np.array(self.data.qvel[0:3].copy(), dtype=np.float32),
            'root_angular': np.array(self.data.qvel[3:6].copy(), dtype=np.float32),
            'mj_data':self.data,
            'mj_model':self.model,

        }

        # if self.load_object:
        #     env_data.update({
        #         'object_pos': np.array(self.data.qpos[7+29 : 7+29+3].copy(), dtype=np.float32), # TODO tmp test
        #         'object_quat': np.array(self.data.qpos[7+29+3:7+29+7].copy(), dtype=np.float32),
        #     })
        return env_data
 
    # def get_joystick_val(self):
    #     stick_val = self.joystick_pc.get_stick_val()
    #     button_dict = self.joystick_pc.get_button_val()
    #     # print(self.keyboard_state["button"])
    #     keyboard_lx, keyboard_ly, keyboard_rx = 0, 0, 0
    #     keyboard_lx += self.keyboard_state["button"]['cmd_right'] * 0.5
    #     keyboard_lx -= self.keyboard_state["button"]['cmd_left'] * 0.5
    #     keyboard_ly += self.keyboard_state["button"]['cmd_back'] * 0.5
    #     keyboard_ly -= self.keyboard_state["button"]['cmd_forward'] * 0.5
    #     keyboard_rx += self.keyboard_state["button"]['cmd_e'] * 0.5
    #     keyboard_rx -= self.keyboard_state["button"]['cmd_q'] * 0.5

    #     control_cmd = {
    #         "stick":{
    #             'lx': stick_val["lx"] if abs(stick_val["lx"]) > abs(keyboard_lx) else keyboard_lx,
    #             'ly': stick_val["ly"] if abs(stick_val["ly"]) > abs(keyboard_ly) else keyboard_ly,
    #             'rx': stick_val["rx"] if abs(stick_val["rx"]) > abs(keyboard_rx) else keyboard_rx,
    #             'ry': stick_val["ry"],
    #             'lt': stick_val["lt"],
    #             'rt': stick_val["rt"],
    #         },
    #         "button": {
    #             'R1': button_dict['RB'],
    #             'L1': button_dict['LB'],
    #             # 'start': button_dict['start'],
    #             # 'select': button_dict['select'],
    #             # 'R2': button_dict['R2'],
    #             # 'L2': button_dict['L2'],
    #             # 'F1': button_dict['F1'],
    #             # 'F2': button_dict['F2'],
    #             'A': button_dict['A'] | self.keyboard_state["button"]['A'],
    #             'start': button_dict['B'] | self.keyboard_state["button"]['start'],
    #             'X': button_dict['X'] | self.keyboard_state["button"]['X'],
    #             'Y': button_dict['Y'],
    #             'up': button_dict['up'] | self.keyboard_state["button"]['up'],
    #             'right': button_dict['right'] | self.keyboard_state["button"]['right'],
    #             'down': button_dict['down'] | self.keyboard_state["button"]['down'],
    #             'left': button_dict['left'] | self.keyboard_state["button"]['left'],
    #         }
    #     }

    #     return control_cmd

    def get_joystick_val(self):
        lx, ly, rx, ry = self.joystick_pc.get_stick_val()
        button_dict = self.joystick_pc.get_button_val()
        
        keyboard_lx, keyboard_ly, keyboard_rx = 0, 0, 0
        keyboard_lx += self.keyboard_state["button"]['cmd_right'] * 0.5
        keyboard_lx -= self.keyboard_state["button"]['cmd_left'] * 0.5
        keyboard_ly += self.keyboard_state["button"]['cmd_back'] * 0.5
        keyboard_ly -= self.keyboard_state["button"]['cmd_forward'] * 0.5
        keyboard_rx += self.keyboard_state["button"]['cmd_e'] * 0.5
        keyboard_rx -= self.keyboard_state["button"]['cmd_q'] * 0.5
        
        control_cmd = {
            "stick":{
                'lx': lx if abs(lx) > abs(keyboard_lx) else keyboard_lx,
                'ly': ly if abs(ly) > abs(keyboard_ly) else keyboard_ly,
                'rx': rx if abs(rx) > abs(keyboard_rx) else keyboard_rx,
                'ry': ry,
            },
            "button": {
                'R1': button_dict['RB'],
                'L1': button_dict['LB'],
                # 'start': button_dict['start'],
                # 'select': button_dict['select'],
                # 'R2': button_dict['R2'],
                # 'L2': button_dict['L2'],
                # 'F1': button_dict['F1'],
                # 'F2': button_dict['F2'],
                'A': button_dict['A'] | self.keyboard_state["button"]['A'],
                'start': button_dict['B'] | self.keyboard_state["button"]['start'],
                'X': button_dict['X'] | self.keyboard_state["button"]['X'],
                'Y': button_dict['Y'],
                'up': button_dict['up'] | self.keyboard_state["button"]['up'],
                'right': button_dict['right'] | self.keyboard_state["button"]['right'],
                'down': button_dict['down'] | self.keyboard_state["button"]['down'],
                'left': button_dict['left'] | self.keyboard_state["button"]['left'],
            }
        }
        # for k in self.keyboard_state["button"]:
        #     self.keyboard_state["button"][k] = 0
        return control_cmd


    def get_marker_manager(self):
        return self.marker_manager
    
    # ========== key callback =============
    def key_call_back(self, keycode):
        key_char = chr(keycode)
        for k in self.keyboard_state["button"]:
            self.keyboard_state["button"][k] = 0
        if key_char in self.mapping:
            button = self.mapping[key_char]
            self.keyboard_state["button"][button] = 1
        elif key_char == "R":
            print("Reset")
        elif key_char == " ":
            self.pause = not self.pause
            pass 
        else:
            print("not mapped", chr(keycode))
            
    def plot(self):
        import matplotlib.pyplot as plt

        if not self.log['time_step']:
            print("No data to plot")
            return

        time_steps = np.array(self.log['time_step'])
        target_q = np.array(self.log['target_q'])
        q = np.array(self.log['q'])
        dq = np.array(self.log['dq'])
        tau = np.array(self.log['tau'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Joint Data Over Time')

        # Plot target_q
        axes[0, 0].set_title('Target Joint Positions')
        for joint in range(29):
            axes[0, 0].plot(time_steps, target_q[:, joint], label=f'Joint {joint}')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Position (rad)')
        axes[0, 0].grid(True)

        # Plot q
        axes[0, 1].set_title('Actual Joint Positions')
        for joint in range(29):
            axes[0, 1].plot(time_steps, q[:, joint], label=f'Joint {joint}')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Position (rad)')
        axes[0, 1].grid(True)

        # Plot dq
        axes[1, 0].set_title('Joint Velocities')
        for joint in range(29):
            axes[1, 0].plot(time_steps, dq[:, joint], label=f'Joint {joint}')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Velocity (rad/s)')
        axes[1, 0].grid(True)

        # Plot tau
        axes[1, 1].set_title('Joint Torques')
        for joint in range(29):
            axes[1, 1].plot(time_steps, tau[:, joint], label=f'Joint {joint}')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Torque (Nm)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
    
    def save_log(self, dir):
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_log_{timestamp}.pkl"
        filepath = os.path.join(dir, filename)

        # Save the log using joblib
        joblib.dump(self.log, filepath)
        print(f"Log saved to: {filepath}")
        
    def should_run_control(self):
        """Check if controller should run (based on control decimation)"""
        return self.sim_step_counter % self.control_decimation == 0
        
    def render(self):
        """Render the simulation"""
        if self.is_render:
            self.viewer.sync()
        
    def is_alive(self):
        """Check if viewer is still alive"""
        if self.is_render:
            return self.viewer.is_running()
        else:
            return True