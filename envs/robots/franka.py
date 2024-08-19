import os
import enum
import numpy as np
from dm_control import mjcf
import xml.etree.ElementTree as et

from ..utils import get_mjcf_model
from .robot import Robot

import os
import enum
import numpy as np
from dm_control import mjcf
import xml.etree.ElementTree as et

from ..utils import get_mjcf_model
from .robot import Robot


class FrankaWithGripper(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0],
            'ctrl': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0],
            'ctrl': [0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]
    _GRIPPER_JOINT_NAMES = [
        "finger_joint1", "finger_joint2",
    ]

    _GRIPPER_JOINT_NUM = 2
    _ARM_JOINT_NUM = 7


class FrankaWithRobotiqGripper(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0],
            'ctrl': [0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]
    _GRIPPER_JOINT_NAMES = [
        "right_driver_joint", "right_spring_link_joint", "right_follower_joint",
        "left_driver_joint", "left_spring_link_joint", "left_follower_joint",
    ]

    _GRIPPER_JOINT_NUM = 6
    _ARM_JOINT_NUM = 7


class FrankaWithDex(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]
    _GRIPPER_JOINT_NAMES = [
        "ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2", "mfj3",
        "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1", "thj2", "thj3",
    ]

    _GRIPPER_JOINT_NUM = 16
    _ARM_JOINT_NUM = 7


class DualRobot(Robot):
    
    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0],
            'ctrl': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.05, 0, 2.08, 0.7854, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.5, 0, -2, -1.57, 1.57, 1.8],
            'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
        "joint1_dual", "joint2_dual", "joint3_dual", "joint4_dual", "joint5_dual", "joint6_dual", "joint7_dual",
    ]
    _GRIPPER_JOINT_NAMES = [
        "ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2", "mfj3",
        "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1", "thj2", "thj3",
        "finger_joint1", "finger_joint2"
    ]

    _GRIPPER_JOINT_NUM = 16 + 1
    _ARM_JOINT_NUM = 7 + 7

