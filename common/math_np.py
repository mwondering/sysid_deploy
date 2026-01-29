import numpy as np

def quat_conjugate(q):
    """Computes the conjugate of a quaternion.
    Args:
        q: (..., 4) array in (w, x, y, z) format
    Returns:
        (..., 4) array
    """
    q = np.asarray(q)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)

def quat_inv(q, eps=1e-9):
    """Computes the inverse of a quaternion.
    Args:
        q: (..., 4) array
        eps: small value to avoid division by zero
    Returns:
        (..., 4) array
    """
    q = np.asarray(q)
    norm_sq = np.sum(q**2, axis=-1, keepdims=True)
    return quat_conjugate(q) / np.clip(norm_sq, eps, None)

def quat_mul(q1, q2):
    """Multiply two quaternions together.
    Args:
        q1: (..., 4) array
        q2: (..., 4) array
    Returns:
        (..., 4) array
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def matrix_from_quat(q):
    """Convert quaternion to rotation matrix.
    Args:
        q: (..., 4) array
    Returns:
        (..., 3, 3) array
    """
    q = np.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    mat[..., 0, 0] = 1 - 2*(yy + zz)
    mat[..., 0, 1] = 2*(xy - wz)
    mat[..., 0, 2] = 2*(xz + wy)
    mat[..., 1, 0] = 2*(xy + wz)
    mat[..., 1, 1] = 1 - 2*(xx + zz)
    mat[..., 1, 2] = 2*(yz - wx)
    mat[..., 2, 0] = 2*(xz - wy)
    mat[..., 2, 1] = 2*(yz + wx)
    mat[..., 2, 2] = 1 - 2*(xx + yy)
    return mat

def quat_apply(q, v):
    """Apply quaternion rotation to vector(s).
    Args:
        q: (..., 4) array
        v: (..., 3) array
    Returns:
        (..., 3) array
    """
    q = np.asarray(q)
    v = np.asarray(v)
    rot = matrix_from_quat(q)
    return np.einsum('...ij,...j->...i', rot, v)

def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    """Subtract transformations between two reference frames into a stationary frame.
    Args:
        t01: (..., 3) array
        q01: (..., 4) array
        t02: (..., 3) array or None
        q02: (..., 4) array or None
    Returns:
        t12: (..., 3) array
        q12: (..., 4) array
    """
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

def compute_z_rotation_align(q1, q2):
    """
    计算使 q1 的朝向绕 Z 轴对齐到 q2 的旋转（仅 yaw 差）
    四元数格式: (w, x, y, z)
    返回: 对应绕 Z 轴的旋转四元数 (w, x, y, z)
    """
    # 提取 yaw 角（绕Z轴）
    # 参考欧拉角 ZYX：yaw = atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    def quat_to_yaw(q):
        w, x, y, z = q
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    yaw1 = quat_to_yaw(q1)
    yaw2 = quat_to_yaw(q2)
    delta_yaw = yaw2 - yaw1

    # 构造绕Z轴旋转的四元数
    half_dyaw = delta_yaw / 2
    qz = np.array([np.cos(half_dyaw), 0.0, 0.0, np.sin(half_dyaw)])
    return qz


def compute_transform_with_z_rotation(src_pose, tgt_pose):
    """
    给定两个姿态 [x, y, z, w, x, y, z]，计算仅基于 Z 轴对齐的变换矩阵 (4x4)
    """
    p1 = np.array(src_pose[:3])
    q1 = np.array(src_pose[3:])
    p2 = np.array(tgt_pose[:3])
    q2 = np.array(tgt_pose[3:])

    # 计算 Z 轴对齐旋转
    qz_align = compute_z_rotation_align(q1, q2)

    # 旋转 p1
    p1_rotated = quat_apply(qz_align, p1)
    translation = p2 - p1_rotated

    return translation, qz_align


def apply_pose_transform(pose, t, q_T):
    """
    应用位姿变换到一个姿态 (位置 + 四元数)
    参数:
        pose: [x, y, z, w, x, y, z]  —— 输入姿态
        t: (3,) 平移向量
        q_T: (4,) 旋转四元数 (w, x, y, z)
    返回:
        变换后的姿态 [x, y, z, w, x, y, z]
    """
    p = np.array(pose[:3])
    q = np.array(pose[3:])
    t = np.array(t)
    q_T = np.array(q_T)

    # 位置变换：先旋转 p，再加平移
    p_new = quat_apply(q_T, p) + t

    # 姿态变换：组合旋转（注意右乘顺序）
    q_new = quat_mul(q_T, q)

    return np.concatenate([p_new, q_new])




def yaw_quat(quat):
    quat_yaw = quat.copy()
    qw = quat_yaw[0]
    qx = quat_yaw[1]
    qy = quat_yaw[2]
    qz = quat_yaw[3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[3] = np.sin(yaw / 2)
    quat_yaw[0] = np.cos(yaw / 2)
    # quat_yaw = normalize(quat_yaw)    # TODO is necessary ?
    return quat_yaw
