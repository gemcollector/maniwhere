import json
import numpy as np
from dm_control import mjcf
from dm_control.rl import control

from .randattr import RandAttr, ExpScheduler, ConstantScheduler, SigmoidScheduler

class RandMJCFWrapper(object):
    def __init__(self, model: mjcf.RootElement, config: dict=None):
        self._model = model
        if config is not None:
            self._locate_rand_params(config)
            self._load_scheduler(config)
            # `default_val` in config file will override corresponding value in xml file
            self.reset_param()

    @classmethod
    def from_rand_config_path(cls, model, config_path):
        """Construct a randomization physics instance.
        
        Args:
            model (mjcf.RootElement)
            config_path (string): path to config file
        """
        try:
            with open(config_path, mode='r') as f:
                config = json.load(f)
        except Exception as e:
            print('Error occurred while loading configuration file: ', e)
        
        return cls(model, config)

    def randomize(self):
        """Randomize the parameters.
        """
        for item in self._rand_params:
            attr, elem_list = item.values()
            if isinstance(attr, list):
                for i, elem in enumerate(elem_list):
                    elem.set_attributes(**{attr[i].name: attr[i].randomize()})
            else:
                attr.randomize()
                for elem in elem_list:
                    elem.set_attributes(**{attr.name: attr.cur_val})

        # adjust DR range
        self._scheduler.step()

    def reset_param(self):
        """Reset the parameters.
        """
        for item in self._rand_params:
            attr, elem_list = item.values()
            if isinstance(attr, list):
                for i, elem in enumerate(elem_list):
                    elem.set_attributes(**{attr[i].name: attr[i].reset()})
            else:
                attr.reset()
                for elem in elem_list:
                    elem.set_attributes(**{attr.name: attr.cur_val})
    
    @property
    def model(self):
        return self._model

    def _locate_rand_params(self, config):
        """Locate the randomization parameters and nodes in the mjcf model tree.
        """
        self._rand_params = []
        for namespace, entrys in config['physics_params'].items():
            for entry in entrys:
                if not entry.get('enable', True):
                    continue

                if 'identifier' in entry:
                    elements = []
                    if isinstance(entry['identifier'], list):
                        for name in entry['identifier']:
                            elements.append(self._find_elem(namespace, name))
                    else:
                        elements.append(self._find_elem(namespace, entry['identifier']))
                else:
                    elements = self._find_all_elem(namespace)

                if elements:
                    #  TODO: Add a warning if both are missing
                    attr_dict = {}
                    if 'discrete_vals' in entry:
                        attr_dict['discrete_vals'] = np.array(entry['discrete_vals'])
                    else:
                        if 'min_val' in entry:
                            attr_dict['min_val'] = np.array(entry['min_val'])
                        elif 'min_coff' in entry:
                            attr_dict['min_val'] = np.array(entry['default_val']) * entry['min_coff']
                        else:
                            raise ValueError("Either min_val or min_coff is needed.")
                        
                        if 'max_val' in entry:
                            attr_dict['max_val'] = np.array(entry['max_val'])
                        elif 'max_coff' in entry:
                            attr_dict['max_val'] = np.array(entry['default_val']) * entry['max_coff']
                        else:
                            raise ValueError("Either max_val or max_coff is needed.")

                    if 'initial_point' in entry:
                        attr_dict['initial_point'] = np.array(entry['initial_point'])
                    if 'circle_center' in entry:
                        attr_dict['circle_center'] = np.array(entry['circle_center'])
                    
                    attr_dict['schedule'] = entry.get('schedule', True)

                    if entry.get('consistent', False):
                        self._rand_params.append({
                            'attr': RandAttr(name=entry['attr'],
                                            default_val=entry['default_val'],
                                            mode=entry.get('mode', 'uniform'),
                                            **attr_dict),
                            'elems': elements
                        })
                    else:
                        self._rand_params.append({
                            'attr': [RandAttr(name=entry['attr'],
                                              default_val=entry['default_val'],
                                              mode=entry.get('mode', 'uniform'),
                                              **attr_dict)
                                              for _ in range(len(elements))],
                            'elems': elements
                        })

    def _load_scheduler(self, config):
        """Load DR range scheduler.
        """
        # TODO: error messages
        if 'scheduler' not in config:
            raise ValueError(f'Please provide a scheduler configuration.')
        
        scheduler_dict = config['scheduler']
        SchedulerClass = globals()[scheduler_dict['type']]
        del scheduler_dict['type']
        self._scheduler = SchedulerClass([param['attr'] for param in self._rand_params],
                                         **scheduler_dict)

    def _find_elem(self, namespace, name):
        """Find single element in mjcf model.
        """
        elem = self._model.find(namespace, name)
        if elem:
            return elem
        else:
            raise ValueError(f'Element {namespace}/{name} not found in mjcf model.')
        
    def _find_all_elem(self, namespace):
        """Find all elements in mjcf model.
        """
        elems = self._model.find_all(namespace)
        if elems:
            return elems
        else:
            raise ValueError(f'Element {namespace} not found in mjcf model.')

class RandPhysics(mjcf.Physics):
    """A physics wrapper for domain randomization.
    """
    @classmethod
    def from_rand_mjcf(cls, mjcf_wrapper: RandMJCFWrapper):
        """Construct a randomization physics instance.
        
        Args:
            model (mjcf.RootElement)
            config (dict): configuration dict
        """
        physics = cls.from_mjcf_model(mjcf_wrapper.model)
        physics._mjcf_wrapper = mjcf_wrapper
        return physics

    def randomize(self):
        """Randomize the parameters.
        """
        self._mjcf_wrapper.randomize()
        self.reload_from_mjcf_model(self._mjcf_wrapper.model)

    def reset_param(self):
        """Reset the parameters.
        """
        self._mjcf_wrapper.reset_param()
        self.reload_from_mjcf_model(self._mjcf_wrapper.model)

    def enable_actuator(self):
        self._mjcf_wrapper.enable_actuator()
        self.reload_from_mjcf_model(self._mjcf_wrapper.model)

    def enable_mocap(self):
        self._mjcf_wrapper.enable_mocap()
        self.reload_from_mjcf_model(self._mjcf_wrapper.model)

    def reset_pose(self, target_pose=None):
        """Reset robot to given pose, default to the predefined initial pose.

        Args:
            target_pose (dict, optional): should contain `qpos`, `ctrl` or both.
        """
        init_pose = self._mjcf_wrapper.init_pose if hasattr(self, '_mjcf_wrapper') else {}
        pose = target_pose or init_pose
        qpos = pose.get('qpos')
        ctrl = pose.get('ctrl')
        if qpos is not None:
            self.named.data.qpos[-len(qpos):] = qpos
        if ctrl is not None:
            actuators = self._mjcf_wrapper.actuators
            self.bind(actuators[-len(ctrl):]).ctrl = ctrl

    def get_site_xpos(self, name):
        """Utility function to get position of <site/>.
        """
        return self.named.data.site_xpos[name]
    
    def get_geom_xpos(self, name):
        """Utility function to get position of <geom/>.
        """
        return self.named.data.geom_xpos[name]

    def set_freejoint_pos(self, name, pos=None, quat=None):
        """Utility function to set position of objects with a <freejoint/>.
        """
        if pos is not None:
            self.named.data.qpos[name][:3] = np.asarray(pos)
        if quat is not None:
            self.named.data.qpos[name][3:] = np.asarray(quat)

    def set_body_pos(self, name, pos=None, quat=None):
        """Utility function to set position of a body.
        """
        if pos is not None:
            self.named.model.body_pos[name] = pos
        if quat is not None:
            self.named.model.body_quat[name] = quat

    def set_joint_pos(self, name, pos):
        """Utility function to set qpos of a joint.
        """
        self.named.data.qpos[name] = np.asarray(pos)

    def check_contact(self, geom1, geom2):
        """Successful when two geom contact.
        """
        contacts = self.data.contact
        for i in range(self.data.ncon):
            contact1 = self.model.id2name(contacts[i].geom1, 'geom')
            contact2 = self.model.id2name(contacts[i].geom2, 'geom')
            if (contact1 == geom1 and contact2 == geom2) or \
                (contact1 == geom2 and contact2 == geom1):
                return True
        return False
    
    def check_contact_group(self, geoms1, geoms2):
        """Successful when some geom in the two group contacts.
        """
        contacts = self.data.contact
        for i in range(self.data.ncon):
            contact1 = self.model.id2name(contacts[i].geom1, 'geom')
            contact2 = self.model.id2name(contacts[i].geom2, 'geom')
            c1_in_g1 = contact1 in geoms1
            c2_in_g2 = contact2 in geoms2
            c2_in_g1 = contact2 in geoms1
            c1_in_g2 = contact1 in geoms2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                return True
        return False

    @property
    def robot(self):
        return self._mjcf_wrapper

class RandEnvironment(control.Environment):
    """An environment wrapper for domain randomization.
    """
    def __init__(
        self,
        physics, 
        task,
        config, 
        time_limit=float('inf'), 
        control_timestep=None, 
        environment_kwargs=None
    ):
        self._rand_params = config['control_params']
        
        environment_kwargs = environment_kwargs or {}
        super().__init__(physics, task, time_limit=time_limit, 
                         control_timestep=control_timestep, **environment_kwargs)

    @property
    def gripper_joint_num(self):
        return self.physics.robot.gripper_joint_num
    
    @property
    def arm_joint_num(self):
        return self.physics.robot.arm_joint_num

    @property
    def dual_arm_joint_num(self):
        if hasattr(self.physics.robot, 'dual_arm_joint_num'):
            return self.physics.robot.dual_arm_joint_num
        else:
            return 0
        
    @property
    def dual_gripper_joint_num(self):
        if hasattr(self.physics.robot, 'dual_gripper_joint_num'):
            return self.physics.robot.dual_gripper_joint_num
        else:
            return 0

    def randomize(self):
        contorl_timestep = self._rand_params.get('control_timestep', None)
        action_delay = self._rand_params.get('action_delay', None)
        # randomize physics and visual parameters
        if isinstance(self._physics, RandPhysics):
            self._physics.randomize()
        else:
            print("Warning: `RandPhysics` class should be derived to perform randomization.")
        # randomize action delay
        if action_delay and action_delay.get('enable', True):
            if hasattr(self._task, 'randomize'):
                self._task.randomize(action_delay['min_val'], action_delay['max_val'])
            else:
                print("Warning: `BaseTask` class should be derived to perform randomization.")
        # random control_timestep
        if contorl_timestep and contorl_timestep.get('enable', True):
            self._n_sub_steps = int(round(
                np.random.uniform(contorl_timestep['min_val'], contorl_timestep['max_val']) / self._physics.timestep()))


    @property
    def state_num(self):
        return self.physics.data.qpos[-self.gripper_joint_num-self.arm_joint_num:].shape