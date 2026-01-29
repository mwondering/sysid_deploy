import zmq
import json
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_homogeneous(p):
    """p = [x,y,z, qx,qy,qz,qw] -> 4x4 homogeneous transform"""
    T = np.eye(4)
    T[:3,:3] = R.from_quat(p[3:]).as_matrix()
    T[:3,3] = p[:3]
    return T

def homogeneous_to_pose(T):
    """4x4 -> [x,y,z, qx,qy,qz,qw] (quat xyzw)"""
    pos = T[:3,3]
    rot = R.from_matrix(T[:3,:3]).as_quat()  # xyzw
    return np.concatenate([pos, rot])

def transform_object_cam_to_pelvis(pose_cam, tf_cam2pelvis):
    T_target_obj = pose_to_homogeneous(pose_cam)        # object expressed in camera basis
    T_pel_cam = pose_to_homogeneous(tf_cam2pelvis)  # camera expressed in pelvis basis

    T_pel_obj = T_pel_cam @ T_target_obj

    return homogeneous_to_pose(T_pel_obj)

# 同时监听obj 和 robot
class ObjectStateListener:
    def __init__(self, ip: str, tf_ip: str, port: int = 5556, tf_port: int = 5555):
        self.ip = ip
        self.tf_ip = tf_ip
        self.port = port
        self.tf_port = tf_port
        self.camera2pelvis = np.array([0,0,0, 0,0,0,1],dtype=float)
        self.obj_in_camera = np.array([0,0,0, 0,0,0,1],dtype=float)
        self.running = True
        self.lock = threading.Lock()
        self.align_transform = np.eye(4)
        self.context = zmq.Context()

        self.tf_socket = self.context.socket(zmq.SUB)
        self.tf_socket.connect(f"tcp://{self.tf_ip}:{self.tf_port}")
        self.tf_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.obj_socket = self.context.socket(zmq.SUB)
        self.obj_socket.connect(f"tcp://{self.ip}:{self.port}")
        self.obj_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        self.poller = zmq.Poller()
        self.poller.register(self.tf_socket, zmq.POLLIN)
        self.poller.register(self.obj_socket, zmq.POLLIN)

        
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def _receive_loop(self):
        while self.running:
            socks = dict(self.poller.poll(timeout=10))  # 10ms 等待
            if self.tf_socket in socks:
                message = self.tf_socket.recv_json()
                # print(message)
                for pose in message.get("camera_tf", []):
                    with self.lock:
                        self.camera2pelvis[:3] = [
                            pose["translation"]["x"],
                            pose["translation"]["y"],
                            pose["translation"]["z"],
                        ]
                        self.camera2pelvis[3:] = [
                            pose["rotation"]["x"],
                            pose["rotation"]["y"],
                            pose["rotation"]["z"],
                            pose["rotation"]["w"],
                        ]

            if self.obj_socket in socks:
                message = self.obj_socket.recv_json()
                # print(message)
                for pose in message.get("object_det", []):
                    with self.lock:
                        self.obj_in_camera[:3] = [
                            pose["translation"]["x"],
                            pose["translation"]["y"],
                            pose["translation"]["z"],
                        ]
                        self.obj_in_camera[3:] = [
                            pose["rotation"]["x"],
                            pose["rotation"]["y"],
                            pose["rotation"]["z"],
                            pose["rotation"]["w"],
                        ]
                

    def get_latest_transform(self):
        with self.lock:
            # return self.obj_in_camera
            return transform_object_cam_to_pelvis(self.obj_in_camera, self.camera2pelvis)
        
    def shutdown(self):
        self.running = False
        self.thread.join()
        self.tf_socket.close()
        self.obj_socket.close()
        self.context.term()

# 示例用法
if __name__ == "__main__":
    listener = ObjectStateListener("192.168.123.164","192.168.123.164")

    try:
        while True:
            tf = listener.get_latest_transform()
            # print("Latest Transform:")
            # print(tf)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping listener...")
        listener.shutdown()
