import numpy as np
import time
import cv2
import pickle
import matplotlib.pyplot as plt

from dmc2gym import make
from envs.ur5e.tasks import ur5e_reach, ur5e_push, ur5e_stack

type = 'check'

def make_video(imgs, timestamps=None, video_name='./test.mp4'):
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    fps = 30  # 设置帧率为1
    width, height = imgs[0].shape[:2]  # 获取图片大小
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    # 将所有图片添加到视频中
    # start_time = cv2.getTickCount()
    for img in imgs:
        # img = imgs[i]
        # timestamp = timestamps[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img)  # 添加图片到视频中
        # 计算当前时间与开始时间的时间差，并等待差值的时间
        # elapsed_time = cv2.getTickCount() - start_time
        # elapsed_time_in_milliseconds = elapsed_time / cv2.getTickFrequency() * 1000
        # wait_time = timestamp - elapsed_time_in_milliseconds
        # if wait_time > 0:
        #     cv2.waitKey(int(wait_time))
    # 关闭视频编写器和所有窗口
    video_writer.release()

def get_action(action):
    if type == 'line':
    # action = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0]
        for _ in range(20):
            action = [a + 0.02 for a in action]
            yield action
        for _ in range(30):
            action = [a - 0.02 for a in action]
            yield action
        for _ in range(60):
            yield action
        for _ in range(10):
            action = [a + 0.2 for a in action]
            yield action
        for _ in range(10):
            action = [a - 0.2 for a in action]
            yield action
        for _ in range(60):
            yield action
        for _ in range(15):
            action = [a + 0.05 for a in action]
            yield action
        for _ in range(15):
            action = [a - 0.05 for a in action]
            yield action
        for _ in range(60):
            yield action
    elif type == 'rand':
        with open('./realworld/actions.pickle', 'rb') as f:
            actions = pickle.load(f)
            for action in actions:
                yield action + st_action
    elif type == 'check':
        with open('./realworld/actions_check.pickle', 'rb') as f:
            actions = pickle.load(f)
            for action in actions:
                yield action + st_action

def plot(xs, ys, str_x, str_y, title):
    _, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'yellow', 'orange']
    for i in range(len(xs)):
        ax.plot(xs[i], ys[i], color=colors[i])
    ax.set_xlabel(str_x)
    ax.set_ylabel(str_y)
    ax.set_title(title)
    # ax.legend()
    plt.savefig(f'./assets/{title}.png')
    plt.show()

def set_param(physics, name, param_name, value):
    elem = physics._find_elem('actuator', name)
    elem.set_attributes(**{param_name: value})

if __name__ == '__main__':
    task_name = 'reach'
    env = ur5e_reach()
    
    physics = env.physics
    # gainprm = 8800
    # kd = 1900
    # force = 650
    # biasprm = [0, -gainprm, -kd]
    # forcerange = [-force, force]
    # set_param(physics, 'shoulder_pan', 'gainprm', [gainprm])
    # set_param(physics, 'shoulder_pan', 'biasprm', biasprm)
    # set_param(physics, 'shoulder_pan', 'forcerange', forcerange)
    # physics.reload_from_mjcf_model(physics._root)

    qpos = [[], [], [], [], [], []]
    acts = [[], [], [], [], [], []]
    steps = []
    imgs = []
    timestamps = []
    timevals = []

    start_time = time.time()
    
    # action = physics.data.ctrl
    
    # print("initial:", physics.data.qpos[:7])
    # action[-1] = 255

    for _ in range(1):
        # env.randomize()
        env.reset()
        steps.append([])
        for p in qpos:
            p.append([])
        for a in acts:
            a.append([])

        st_action = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0]
        for step, act in enumerate(get_action(st_action)):

            # for i in range(len(act)):
            #     if i > -1: act[i] = 0
            # action[2] -= 0.2
            # action = np.zeros(env.action_space.shape, dtype=np.float32)

            steps[-1].append(step)
            for i, p in enumerate(qpos):
                p[-1].append(physics.data.qpos[i] - st_action[i])
                # acts[i][-1].append(act[i] - st_action[i])
                acts[i][-1].append(act[i])
            # timestep[0]: next_obs, timestep[1]: reward, timestep[2]: done, timestep[3]: info
            timestep = env.step(act)
            # print(act)

            img = physics.render(256, 256, camera_id=physics.model.name2id('angled_cam', 'camera'))
            imgs.append(img)
            # print("--: ", physics.data.qpos[-1])

            # timestamps.append(time.time() - start_time)
            # timevals.append(physics.data.time)

    with open(f'./realworld/real_qpos_{type}_0.08_10_0.001.pickle', 'rb') as f:
        real_qpos = pickle.load(f)

        for i in range(len(real_qpos)):
            _, ax = plt.subplots()
            colors = ['red', 'green', 'blue', 'yellow', 'orange']
            ax.plot(steps[0], qpos[i][0], color=colors[0], label='simulation')
            ax.plot(steps[0], real_qpos[i], color=colors[1], label='real')
            # ax.plot(steps[0], acts[i][0], color=colors[2], label='action')
            ax.set_xlabel('steps')
            ax.set_ylabel('joint')
            ax.set_title(f'sim&real{i}')
            ax.legend()
            plt.savefig(f'./assets/sim2real_align_{type}_{i}.png')
            # plt.show()


    # print("stamp:", timestamps)
    # print("tvals:", timevals)

    # print("end:", physics.data.qpos[:7])

    # print(physics.data.qpos[2], physics.data.qpos[2] - 1.5708)

    make_video(imgs)

 