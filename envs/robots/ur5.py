import os
import enum
import numpy as np
from dm_control import mjcf
import xml.etree.ElementTree as et

from ..utils import get_mjcf_model
from .robot import Robot

class UR5WithGripper(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0]
        }
    }

    _ARM_JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    _GRIPPER_JOINT_NAMES = [
        "right_driver_joint", "right_spring_link_joint", "right_follower_joint",
        "left_driver_joint", "left_spring_link_joint", "left_follower_joint",
    ]

    _GRIPPER_JOINT_NUM = 6
    _ARM_JOINT_NUM = 6


class UR5WithDex(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    }

    _ARM_JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    _GRIPPER_JOINT_NAMES = [
        "ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2", "mfj3",
        "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1", "thj2", "thj3",
    ]

    _GRIPPER_JOINT_NUM = 16
    _ARM_JOINT_NUM = 6
