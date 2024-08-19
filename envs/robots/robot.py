import os
import enum
import numpy as np
from dm_control import mjcf
import xml.etree.ElementTree as et

from ..utils import get_mjcf_model
from ..randomize.wrapper import RandMJCFWrapper

class Robot(RandMJCFWrapper):

    class ControlMode(enum.Enum):
        NO_CONTROL = 0
        ACTUATOR = 1
        MOCAP = 2

    # _INIT_POSE = {
    #     ControlMode.ACTUATOR: {
    #         'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0],
    #         'ctrl': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0]
    #     },
    #     ControlMode.MOCAP: {
    #         'qpos': [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0],
    #         'ctrl': [0]
    #     }
    # }

    # _JOINT_NAMES = [
    #     "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
    #     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    # ]

    def __init__(
        self,
        model: mjcf.RootElement,
        actuator: mjcf.RootElement=None,
        mocap: mjcf.RootElement=None,
        config: dict=None,
        init_pose: dict=None
    ):
        """Initialize a UR5 Arm instance.

        Args:
            model (mjcf.RootElement): mjcf model of the robot
            actuator (mjcf.RootElement, optional): mjcf model of the actuators
            mocap (mjcf.RootElement, optional): mjcf model of the mocap body and weld connect
            config (dict): domain randomization configurations
            init_pose (dict): initial robot pose

        Raises:
            ValueError: if neither mjcf model of the two modes is provided 
        """
        super().__init__(model, config)
        if actuator is None and mocap is None:
            raise ValueError('At least MJCF model of one control mode is needed.')
        self._actuator_model = actuator
        self._mocap_model = mocap
        self._control_mode = Robot.ControlMode.NO_CONTROL
        self._init_pose = init_pose
        # if control_mode == Robot.ControlMode.ACTUATOR:
        #     self.enable_actuator()
        # elif control_mode == Robot.ControlMode.MOCAP:
        #     self.enable_mocap()

    @classmethod
    def from_file_path(
        cls,
        xml_path: str,
        asset_paths: list,
        actuator_path: str=None,
        mocap_path: str=None,
        config: dict=None,
        init_pose: dict=None
    ) -> 'Robot':
        """Construct a Robot Arm instance.

        Args:
            xml_path (str): path of xml file of the base model 
            asset_paths (list): a list of xml file paths, containing model assets
            actuator_path (str): path of xml file containing actuators
            mocap_path (str): path of xml file containing mocap body and connects
            config (dict): domain randomization configurations
            init_pose (dict): initial robot pose

        Returns:
            UR5Arm: an instance of UR5Arm
        """
        model = get_mjcf_model(xml_path, asset_paths)
        actuator = mjcf.from_path(actuator_path) if actuator_path else None
        mocap = mjcf.from_path(mocap_path) if mocap_path else None
        return cls(model, actuator, mocap, config, init_pose)

    def enable_actuator(self):
        """Switch to actuator control mode.
        """
        if self._actuator_model is None:
            raise RuntimeError('Actuator MJCF model is not provided.')
        
        root = self._actuator_model.to_xml()
        self._attach(root.find('.//actuator'), self._model.actuator)
        # if self._mocap_model and self._mocap_model.parent_model is not None:
        #     self._mocap_model.detach()
        # if self._actuator_model.parent_model is None:
        #     self._model.attach(self._actuator_model)

        self._control_mode = Robot.ControlMode.ACTUATOR

    def enable_mocap(self):
        """Switch to mocap control mode.
        """
        if self._mocap_model is None:
            raise RuntimeError('Mocap MJCF model is not provided.')
        
        root = self._mocap_model.to_xml()
        self._attach(root.find('.//worldbody'), self._model.worldbody)
        self._attach(root.find('.//actuator'), self._model.actuator)
        self._attach(root.find('.//equality'), self._model.equality)

        self._control_mode = Robot.ControlMode.MOCAP

    @property
    def arm_joint_names(self):
        if hasattr(self, '_ARM_JOINT_NAMES'):
            return self._ARM_JOINT_NAMES
        else:
            raise NotImplementedError
        
    @property
    def gripper_joint_names(self):
        if hasattr(self, '_GRIPPER_JOINT_NAMES'):
            return self._GRIPPER_JOINT_NAMES
        else:
            raise NotImplementedError

    @property
    def init_pose(self):
        if self._init_pose:
            return self._init_pose[self._control_mode]
        elif hasattr(self, '_INIT_POSE'):
            return self._INIT_POSE[self._control_mode]
        else:
            raise NotImplementedError
    
    @property
    def arm_joint_num(self):
        return self._ARM_JOINT_NUM
    
    @property
    def gripper_joint_num(self):
        return self._GRIPPER_JOINT_NUM
    
    @property
    def dual_arm_joint_num(self):
        if hasattr(self, '_DUAL_ARM_JOINT_NUM'):
            return self._DUAL_ARM_JOINT_NUM
        else:
            return 0
    
    @property
    def dual_gripper_joint_num(self):
        if hasattr(self, '_DUAL_GRIPPER_JOINT_NUM'):
            return self._DUAL_GRIPPER_JOINT_NUM
        else:
            return 0
    
    @property
    def actuators(self):
        return self._find_all_elem('actuator')
    
    def _attach(self, root: et.Element, father: mjcf.Element):
        if root is None:
            return
        for child in root:
            if 'name' in child.attrib:
                child.attrib['name'] = child.attrib['name'].split('/')[-1]
            son = father.add(child.tag, **child.attrib)
            self._attach(child, son)
