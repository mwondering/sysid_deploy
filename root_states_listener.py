import zmq
import json
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(position, quaternion):
    """将位置和四元数转为4x4变换矩阵"""
    rot = R.from_quat(quaternion)  # xyzw
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T

def compute_alignment_transform(source_pose, target_pose):
    """
    source_pose 和 target_pose 是长度为7的数组：[x, y, z, qx, qy, qz, qw]
    返回 T，使得 T @ source_pose = target_pose
    """
    p1 = source_pose[:3]
    q1 = source_pose[3:]
    
    p2 = target_pose[:3]
    q2 = target_pose[3:]
    
    T1 = pose_to_matrix(p1, q1)
    T2 = pose_to_matrix(p2, q2)
    
    T = T2 @ np.linalg.inv(T1)
    return T

def compute_alignment_transform_yaw(source_pose, target_pose):
    """
    source_pose 和 target_pose 是长度为7的数组：[x, y, z, qx, qy, qz, qw]
    返回 T，使得 T @ source_pose ≈ target_pose（但只考虑 target 的 yaw）
    """
    p1 = source_pose[:3]
    q1 = source_pose[3:]

    p2 = target_pose[:3]
    q2 = target_pose[3:]

    # 原始变换
    T1 = pose_to_matrix(p1, q1)

    # 从 target 的四元数中提取 yaw
    r = R.from_quat(q2)
    _, _, yaw = r.as_euler('xyz', degrees=False)

    # 构造仅包含 yaw 的旋转
    r_yaw = R.from_euler('z', yaw).as_matrix()

    # 构造 target 的 yaw-only 变换矩阵
    T2_yaw_only = np.eye(4)
    T2_yaw_only[:3, :3] = r_yaw
    T2_yaw_only[:3, 3] = p2  # 使用完整平移

    # 计算相对变换
    T = T2_yaw_only @ np.linalg.inv(T1)
    return T

def apply_transform(pose, T):
    """
    对一个 7维 Pose 应用 4x4 变换矩阵 T
    pose: [x, y, z, qx, qy, qz, qw]
    返回变换后的 Pose（也是7维）
    """
    pos = pose[:3]
    quat = pose[3:]
    
    # 位置变换
    pos_hom = np.ones(4)
    pos_hom[:3] = pos
    new_pos = T @ pos_hom
    new_pos = new_pos[:3]
    
    # 姿态变换
    rot = R.from_quat(quat)
    T_rot = R.from_matrix(T[:3, :3])
    new_rot = T_rot * rot  # 先应用 T 的旋转，再应用原始旋转
    new_quat = new_rot.as_quat()  # xyzw

    return np.concatenate([new_pos, new_quat])

class RootStateListener:
    def __init__(self, ip: str, port: int = 5555, FAKE: bool = True):
        self.ip = ip
        self.port = port
        self.latest_transform = np.array([0,0,0, 0,0,0,1],dtype=float)
        self.running = True
        self.lock = threading.Lock()
        self.align_transform = np.eye(4)
        if FAKE:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://{self.ip}:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            # print(f"线程已创建，是否存活: {self.thread.is_alive()}")
            self.thread.start()
            # print(f"线程启动后是否存活: {self.thread.is_alive()}")

    def _receive_loop(self):
        while self.running:
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
                # print("message:", message)
                transforms = message.get("root_state_tf", [])
                for tf in transforms:
                    with self.lock:
                        self.latest_transform[0] = tf["translation"]["x"]
                        self.latest_transform[1] = tf["translation"]["y"]
                        self.latest_transform[2] = tf["translation"]["z"]
                        self.latest_transform[3] = tf["rotation"]["x"]
                        self.latest_transform[4] = tf["rotation"]["y"]
                        self.latest_transform[5] = tf["rotation"]["z"]
                        self.latest_transform[6] = tf["rotation"]["w"]
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                print(f"[RootStateListener] Error receiving data: {e}")
                time.sleep(0.1)

    def get_latest_transform(self):
        with self.lock:
            return apply_transform(self.latest_transform, self.align_transform)
    
    def set_align_pose(self, align_target):
        with self.lock:
            self.align_transform  = compute_alignment_transform_yaw(self.latest_transform, align_target)
        
    def shutdown(self):
        self.running = False
        self.thread.join()
        self.socket.close()
        self.context.term()

# 示例用法
if __name__ == "__main__":
    listener = RootStateListener("192.168.123.164")

    try:
        while True:
            tf = listener.get_latest_transform()
            print("Latest Transform:")
            print(tf)
            # time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping listener...")
        listener.shutdown()
