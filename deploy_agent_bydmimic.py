from controller.amp_controller import AMPController

from controller.bydmimic_controller_xingyi_sysID import BydMimicControllerXingyiSysID
# from controller.opentrack_controller_xingyi_sysID import OpentrackControllerXingyiSysID
from controller.controller_manager_xingyi_sysID import ControllerManagerXingyiSysID
import numpy as np
import time
import argparse
import joblib

PLOT = True
SAVE_LOG = True


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--env',type=str, default="mujoco", choices=["mujoco", "real"],)
    args.add_argument('--object_name',type=str, default="", choices=["largebox"],)
    args.add_argument('--policy_path',type=str, required=True)
    args.add_argument('--save_motion', action='store_true')

    import datetime
    import time

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_motion_path = f"motions_deployed/save_motion_{current_time}"
    # import pdb;pdb.set_trace()
    
    args = args.parse_args()
    
    amp_controller = AMPController()

    if args.env == 'mujoco':
        from env.mujoco_env import MujocoEnv
        env = MujocoEnv(object_name=args.object_name,xml_path = "sysid_xmls/mjcf/g1_modified.xml")
    elif args.env == 'real':
        from env.real_env import RealEnv
        env = RealEnv()
    else:
        raise ValueError(f'invade env name: {args.env}')
            
    main_controller = ControllerManagerXingyiSysID(env)
    main_controller.add_controller(amp_controller)
    byd_path_list = [
        # '/home/lenovo/project/BeyondMimic/logs/rsl_rl/g1_flat/2026-01-13_16-42-06_pufu_uniform_sampling_small_tol/exported/policy_15000.onnx',
        # '/home/lenovo/sysid_deploy/onnxs/policy_012901.onnx',
        '/home/unitree/workspace/sysid_deploy/onnxs/policy_012901.onnx',
    ]
    for idx,byd_path in enumerate(byd_path_list):
        print(idx)
        print("byd_path:", byd_path)
        main_controller.add_controller(BydMimicControllerXingyiSysID(byd_path))


    # Initialize simulation
    env.reset()
    main_controller.reset_all_controllers()

    target_q = main_controller.cur_controller.default_dof_pos
    kps = main_controller.cur_controller.kps
    kds = main_controller.cur_controller.kds

    motion_to_save = {
        'base_ang_vel': [],
        'base_lin_vel': [],
        'base_rotation': [],
        'base_pos': [],
        'joint_pos': [],
        'joint_vel': [],
        'action': [],
        'target_q': [],
        # 'joint_torque': [],
    }


    # marker_manager = env.get_marker_manager()
    # marker_manager.add_marker('destination', value=np.array([0.8, 0.0, 0.0]))


    try:
        while env.is_alive():
            t1 = time.time()
            main_controller.check_state()


            if env.should_run_control():
                # import pdb; pdb.set_trace()
                # import ipdb; ipdb.set_trace()
                env_data = env.get_env_data()
                # import pdb; pdb.set_trace()
                target_q, kps, kds = main_controller.cur_controller.step(env_data)

                if args.save_motion:
                    if main_controller.cur_controller is not None and (isinstance(main_controller.cur_controller, BydMimicControllerXingyiSysID)):
                        motion_to_save['base_ang_vel'].append(env_data['root_angular'])
                        # motion_to_save['base_lin_vel'].append(env_data['root_linear'])
                        motion_to_save['base_rotation'].append(env_data['root_quat'])
                        motion_to_save['base_pos'].append(env_data['root_pos'])
                        motion_to_save['joint_pos'].append(env_data['joint_pos'])
                        motion_to_save['joint_vel'].append(env_data['joint_vel'])
                        
                        motion_to_save['action'].append(main_controller.cur_controller.action)
                        motion_to_save['target_q'].append(target_q)


                        # ref record data. 
                        # env_data = {
                        #     'joint_pos': np.array(self.data.qpos[7:7+29].copy(), dtype=np.float32),
                        #     'joint_vel': np.array(self.data.qvel[6:6+29].copy(), dtype=np.float32),
                        #     'root_pos': np.array(root_pos, dtype=np.float32), # + np.random.normal(0, 0.25, size=3)
                        #     'root_quat': np.array(self.data.qpos[3:7].copy(), dtype=np.float32),
                        #     'root_linear': np.array(self.data.qvel[0:3].copy(), dtype=np.float32),
                        #     'root_angular': np.array(self.data.qvel[3:6].copy(), dtype=np.float32),
                        # }

                        # data_dict = {
                        #     'base_ang_vel': (robot.data.root_ang_vel_b)[0].cpu().numpy(),
                        #     'base_lin_vel': robot.data.root_lin_vel_w[0].cpu().numpy(),
                        #     'base_rotation': (robot.data.root_link_quat_w)[0].cpu().numpy(),  
                        #     'base_pos': (robot.data.root_link_pos_w)[0].cpu().numpy(),  
                        #     'joint_pos': (robot.data.joint_pos)[0].cpu().numpy(),
                        #     'joint_vel': (robot.data.joint_vel)[0].cpu().numpy(),
                        #     'action': (action * self.action_scale + self.robot.data.default_joint_pos)[0].cpu().numpy(),
                        #     }

                    else:
                        if len(motion_to_save['joint_pos']) != 0:
                            joblib.dump(motion_to_save, f"{save_motion_path}.pkl")
                        motion_to_save = {
                            'base_ang_vel': [],
                            'base_lin_vel': [],
                            'base_rotation': [],
                            'base_pos': [],
                            'joint_pos': [],
                            'joint_vel': [],
                            'action': [],
                            'target_q': [],
                            # 'joint_torque': [],
                        }
                 
            env.step(target_q, kps, kds)
            # import pdb; pdb.set_trace() 
            t2 = time.time()
            elapsed = t2 - t1
            if args.env == 'real':
                if elapsed < env.control_dt:
                    time.sleep(env.control_dt - elapsed)
                else:
                    print(f'Warning, control loop time cost {elapsed:.3f} s')

    except KeyboardInterrupt:
        if len(motion_to_save['joint_pos']) != 0:
            joblib.dump(motion_to_save,f"{save_motion_path}.pkl")
            print("save motion to:", f"{save_motion_path}.pkl")
        print("Simulation interrupted")
    finally:
        if len(motion_to_save['joint_pos']) != 0:
            joblib.dump(motion_to_save,f"{save_motion_path}.pkl")
        # listener.stop()
        print("Simulation ended")

    if len(motion_to_save['joint_pos']) != 0:
        joblib.dump(motion_to_save,f"{save_motion_path}.pkl")
