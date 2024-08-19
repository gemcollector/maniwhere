import os
import json
import numpy as np
from dm_control.utils import io as resources
from dm_control import mjcf
from . import _SUITE_DIR

def get_mjcf_model(xml_path, asset_paths):
    """Returns a mjcf model for the given configuration.
    """
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(_SUITE_DIR, xml_path)
    xml = resources.GetResource(xml_path)

    assets = {}
    for asset_path in asset_paths:
        if not os.path.isabs(asset_path):
            asset_path = os.path.join(_SUITE_DIR, asset_path)
        if os.path.isdir(asset_path):
            file_list = [entry.name for entry in os.scandir(asset_path) if entry.is_file()]
            for file in file_list:
                assets[file] = resources.GetResource(os.path.join(asset_path, file))
        else:
            assets[os.path.basename(asset_path)] = resources.GetResource(asset_path)
    
    return mjcf.from_xml_string(
        xml_string=xml,
        assets=assets,
        model_dir=os.path.dirname(os.path.abspath(xml_path))
    )

def get_mjcf_model_from_config(config_path):
    """Returns a mjcf model for the given configuration.
    """
    with open(config_path, mode='r') as f:
        config = json.load(f)
    xml_path = os.path.join(os.path.dirname(config_path), config['xml'])
    xml = resources.GetResource(xml_path)
    assets = {}
    for asset_path in config['assets']:
        asset_path = os.path.join(os.path.dirname(config_path), asset_path)
        if os.path.isdir(asset_path):
            file_list = [entry.name for entry in os.scandir(asset_path) if entry.is_file()]
            for file in file_list:
                assets[file] = resources.GetResource(os.path.join(asset_path, file))
        else:
            assets[os.path.basename(asset_path)] = resources.GetResource(asset_path)
    
    return mjcf.from_xml_string(
        xml_string=xml,
        assets=assets,
        model_dir=os.path.dirname(os.path.abspath(config['xml']))
    )

def concat_mjcf_model(base_mjcf, new_mjcf: mjcf.RootElement, attach_site_name='attachment_site'):
    """Attaches an mjcf model to another mjcf model.

    The base mjcf must have a site to attach, the name of which should be 
    specified by `attach_site_name`.

    Args:
        base_mjcf: The mjcf.RootElement of the arm.
        attach_mjcf: The mjcf.RootElement of the hand.

    Raises:
        ValueError: If base mjcf does not have an attachment site.

    Developing...
    """
    attachment_site = base_mjcf.find('site', attach_site_name)
    if attachment_site is None:
        raise ValueError('No attachment site found in the base model.')

    # Expand the ctrl and qpos keyframes to account for the new DoFs.
    base_keys = base_mjcf.find_all('key')
    new_keys = new_mjcf.find_all('key')
    delta = len(base_keys) - len(new_keys)
    if delta > 0:
        new_physics = mjcf.Physics.from_mjcf_model(new_mjcf)
        for _ in range(delta):
            new_keys.append(new_mjcf.keyframe.add(
                'key', 
                ctrl=np.zeros(new_physics.model.nu),
                qpos=np.zeros(new_physics.model.nq)
            ))
    elif delta < 0:
        base_physics = mjcf.Physics.from_mjcf_model(base_mjcf)
        for _ in range(-delta):
            base_keys.append(base_mjcf.keyframe.add(
                'key',
                ctrl=np.zeros(base_physics.model.nu), 
                qpos=np.zeros(base_physics.model.nq)
            ))
    
    # print("1111111111111111111111111")

    # print("base: ", base_keys)
    # print("new: ", new_keys)

    for base_key, new_key in zip(base_keys, new_keys):
        base_key.ctrl = np.concatenate([base_key.ctrl, new_key.ctrl])
        base_key.qpos = np.concatenate([base_key.qpos, new_key.qpos])
    
    # print("22222222222222222222222222")

    # print("before attach: ", base_mjcf.find_all('key'))

    # import pprint
    # np.set_printoptions(threshold=np.inf)
    # print(new_mjcf)
    new_mjcf.keyframe.remove(affect_attachments=True)
    # print(new_mjcf.all_children())

    attachment_site.attach(new_mjcf)

    # print("after attach: ", base_mjcf.find_all('key'))
    # print("====================")

def scene_arm_hand(config_path):
    """Generate a customed mjcf model, which includes an robot arm 
        with a hand, attached to a certain task scene.
    Developing...
    """
    with open(config_path, mode='r') as f:
        config = json.load(f)
    scene_path = os.path.join(os.path.dirname(config_path), config['scene_xml'])
    arm_path = os.path.join(os.path.dirname(config_path), config['arm_xml'])
    hand_path = os.path.join(os.path.dirname(config_path), config['hand_xml'])

    scene = mjcf.from_path(scene_path)
    arm = mjcf.from_path(arm_path)
    hand = mjcf.from_path(hand_path)

    # print("scene: ", scene)
    # print("arm: ", arm)
    # print("hand: ", hand)

    concat_mjcf_model(arm, hand)
    concat_mjcf_model(scene, arm)
    
    return scene