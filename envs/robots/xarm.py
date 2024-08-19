from .robot import Robot

class XArm6WithDex(Robot):

    _INIT_POSE = {
        Robot.ControlMode.ACTUATOR: {
            'qpos': [0, 0.5, -1.76, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0.5, -1.76, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        Robot.ControlMode.MOCAP: {
            'qpos': [0, 0.5, -1.76, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    }

    _ARM_JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ]
    _GRIPPER_JOINT_NAMES = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"
    ]

    _GRIPPER_JOINT_NUM = 16
    _ARM_JOINT_NUM = 6