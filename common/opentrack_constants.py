
from ml_collections import config_dict
ENABLE_RANDOMIZE = True
EPISODE_LENGTH = 1000

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for G1."""

from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "xmls" / "unitree_g1"
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_wholebody_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_wholebody_rough_terrain.xml"
DEPLOY_XML =ROOT_PATH /"g1.xml"
NUM_JOINT = 29


def task_to_xml(task_name: str) -> Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
        # "flat_terrain": DEPLOY_XML,
        # "rough_terrain": DEPLOY_XML,
    }[task_name]

OBS_SIZE_DICT = {
    "dif_joint_pos":29,
    "dif_joint_vel":29,
    "gyro_pelvis":3,
    "gvec_pelvis":3,
    "joint_pos":29,
    "joint_vel":29,
    "last_motor_targets":29,
    "ref_feet_height":4,
    "ref_root_angvel":3,
    "ref_root_height":1,
    "ref_root_linvel":3,
}

FEET_SITES = [
    "left_foot",
    "right_foot",
]

FEET_ALL_SITES = [
    "left_foot",
    "right_foot",
    "left_foot_top",
    "right_foot_top",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
# LEFT_FEET_GEOMS = ["left_foot4_collision"]
# RIGHT_FEET_GEOMS = ["right_foot4_collision"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg.
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
)

DOF_VEL_LIMITS = [
    32.0,
    32.0,
    32.0,
    20.0,
    37.0,
    37.0,
    32.0,
    32.0,
    32.0,
    20.0,
    37.0,
    37.0,
    32.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
]

ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# fmt: off
TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])

DEFAULT_QPOS = np.float32([
    0, 0, 0.8,
    1, 0, 0, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

KPs = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])

UPPER_BODY_LINKs = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

LOWER_BODY_LINKs = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
]

UPPER_BODY_JOINTs = [
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


FEET_LINKs = ["left_ankle_roll_link", "right_ankle_roll_link"]

SHOULDER_LINKs = ["right_shoulder_pitch_link", "left_shoulder_pitch_link"]


LAFAN1_DATASETS = [
    # 'dance1_subject1',
    'dance1_subject2',
    # 'dance1_subject3',
    # 'dance2_subject1',
    # 'dance2_subject2',
    # 'dance2_subject3',
    # 'dance2_subject4',
    # 'dance2_subject5',
    # 'fallAndGetUp1_subject1',
    # 'fallAndGetUp1_subject4',
    # 'fallAndGetUp1_subject5',
    # 'fallAndGetUp2_subject2',
    # 'fallAndGetUp2_subject3',
    # 'fallAndGetUp3_subject1',
    # 'fight1_subject2',
    # 'fight1_subject3',
    # 'fight1_subject5',
    # 'fightAndSports1_subject1',
    # 'fightAndSports1_subject4',
    # 'jumps1_subject1',
    # 'jumps1_subject2',
    # 'jumps1_subject5',
    # 'run1_subject2',
    # 'run1_subject5',
    # 'run2_subject1',
    # 'run2_subject4',
    # 'sprint1_subject2',
    # 'sprint1_subject4',
    # 'walk1_subject1',
    # 'walk1_subject2',
    # 'walk1_subject5',
    # 'walk2_subject1',
    # 'walk2_subject3',
    # 'walk2_subject4',
    # 'walk3_subject1',
    # 'walk3_subject2',
    # 'walk3_subject3',
    # 'walk3_subject4',
    # 'walk3_subject5',
    # 'walk4_subject1',
]

LAFAN1_SPECIALIST_DATASETS_1 = [
    'dance1_subject1',
    'dance1_subject2',
    'dance1_subject3',
    'dance2_subject1',
    'dance2_subject2',
    'dance2_subject3',
    'dance2_subject4',
    'dance2_subject5',
]

LAFAN1_SPECIALIST_DATASETS_2 = [
    'fallAndGetUp1_subject1',
    'fallAndGetUp1_subject4',
    'fallAndGetUp1_subject5',
    'fallAndGetUp2_subject2',
    'fallAndGetUp2_subject3',
    'fallAndGetUp3_subject1',
]

def default_config() -> config_dict.ConfigDict:

    env_config = config_dict.create(
        terrain_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        action_scale=1.0,
        recalculate_velocity=True,
        history_len=79,
        enable_randomize=ENABLE_RANDOMIZE,
        soft_joint_pos_limit_factor=0.95,
        reference_traj_config=config_dict.create(
            name={"lafan1": LAFAN1_DATASETS},
            random_start=True,
            fixed_start_frame=0,  # only works if random_start is False
            add_pertubation=False,  # only for test
        ),
        termination_config=config_dict.create(
            root_height_threshold=0.3,
            rigid_body_dif_threshold=0.5,
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking related rewards.
                rigid_body_pos_tracking_upper=1.0,
                rigid_body_pos_tracking_lower=0.5,
                rigid_body_rot_tracking=0.5,
                rigid_body_linvel_tracking=0.5,
                rigid_body_angvel_tracking=0.5,
                joint_pos_tracking=0.75,
                joint_vel_tracking=0.5,
                roll_pitch_tracking=1.0,
                penalty_action_rate=-0.5,
                penalty_torque=-0.00002,
                dof_pos_limit=-10,
                dof_vel_limit=-5,
                # collision=-10,
                termination=-200,
                # root vel
                root_linvel_tracking=1.0,
                root_angvel_tracking=1.0,
                root_height_tracking=1.0,
                feet_height_tracking=1.0,
                feet_pos_tracking=2.1,
                smoothness_joint=-1e-6,
            ),
            auxiliary=config_dict.create(
                upper_body_sigma=1.0,
                lower_body_sigma=1.0,
                feet_pos_sigma=1.0,
                body_rot_sigma=1.0,
                feet_rot_sigma=1.0,
                body_linvel_sigma=5.0,
                feet_linvel_sigma=1.0,
                body_angvel_sigma=50.0,
                feet_angvel_sigma=1.0,
                joint_pos_sigma=10.0,
                joint_vel_sigma=1.0,
                root_pos_sigma=0.5,
                root_rot_sigma=1.0,
                root_linvel_sigma=1.0,
                root_angvel_sigma=10.0,
                roll_pitch_sigma=0.2,
                # aux height and contact
                root_height_sigma=0.1,
                feet_height_sigma=0.1,
                global_feet_vel_threshold=0.5,
                global_feet_height_threshold=0.04,
                feet_linvel_threshold=0.1,
                feet_angvel_threshold=0.1,
                feet_slipping_sigma=2.0,
            ),
            penalize_collision_on=[
                ["left_hand_collision", "left_thigh"],
                ["right_hand_collision", "right_thigh"],
                ["left_hand_collision", "right_hand_collision"],
                ["left_hand_collision", "right_wrist_pitch_collision"],
                ["right_hand_collision", "left_wrist_pitch_collision"],
            ],
        ),
        push_config=config_dict.create(
            enable=ENABLE_RANDOMIZE,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        obs_scales_config=config_dict.create(joint_vel=0.05),
        obs_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
            "dif_joint_pos",
            "dif_joint_vel",
            "ref_root_linvel",
            "ref_root_angvel",
            "ref_root_height",
            "ref_feet_height",
        ],
        privileged_obs_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "linvel_pelvis",
            "dif_torso_rp",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
            "dif_joint_pos",
            "dif_joint_vel",
            "feet_contact",
            "dif_feet_height",
            "dif_root_height",
            "dif_root_linvel",
            "dif_root_angvel",
            "dif_rigid_body_pos_local",
            "dif_rigid_body_rot_local",
            "dif_rigid_body_linvel_local",
            "dif_rigid_body_angvel_local",
        ],
        history_keys=[
            "gyro_pelvis",
            "gvec_pelvis",
            "joint_pos",
            "joint_vel",
        ],
    )

    policy_config = config_dict.create(
        num_timesteps=3_000_000_000,
        max_devices_per_host=8,
        # high-level control flow
        wrap_env=True,
        # environment wrapper
        num_envs=8192,  # 8192(256*32), 16384(512*32), 32768(1024*32)
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        # ppo params
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=256,  # 256, 512, 1024
        num_minibatches=32,  # 8, 16, 32
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 512, 256, 256, 128),
            value_hidden_layer_sizes=(512, 512, 256, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        # eval
        num_evals=5,
        # training metrics
        log_training_metrics=True,
        training_metrics_steps=int(1e6),  # 1M
        # callbacks
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
    )
    return config