import os
import copy
import random
import numpy as np
import torch
from config import Config
from agent import Agent
from environment import Environment
from heuristics import Heuristic
import matplotlib.pyplot as plt
import seaborn as sns


def train(cfg, env, agent):
    """训练"""
    print('开始训练！')
    rewards = []  # the reward of each episode
    steps = []
    best_ep_reward = 0  # the max reward obtained so far
    output_agent = None

    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # the reward of this eps
        ep_step = 0  # the action step of this eps
        feature_vector, state = env.reset()
        done = False

        while not done:
            ep_step += 1
            action = agent.sample_action(feature_vector)
            heuristic = Heuristic(action, state)
            server = heuristic.action
            feature_vector_, reward, done, state_ = env.step(server, state)
            agent.memory.push((feature_vector, action, agent.log_probs.detach().cpu().numpy().item(), reward, done))
            state = state_
            feature_vector = feature_vector_
            agent.update()
            ep_reward += reward

        # evaluate the model during training
        if (i_ep+1) % cfg.eval_per_episode == 0:
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                feature, state = env.reset()
                done = False
                while not done:
                    action = agent.predict_action(feature)
                    heu = Heuristic(action, state)
                    server = heu.action
                    feature_, reward, done, state_ = env.step(server, state)
                    state = state_
                    feature = feature_
                    eval_ep_reward += reward

                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward / cfg.eval_eps
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)  # snapshot of the best performance of the agent
            print(f'回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}')

        steps.append(ep_step)
        rewards.append(ep_reward)
    print('完成训练！')
    return output_agent, {'rewards':rewards}


def test(cfg, env, agent):
    print('开始测试！')
    rewards = []
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0
        ep_step = 0
        feature, state = env.reset()
        done = False
        while not done:
            ep_step += 1
            action = agent.predict_action(feature)
            heu = Heuristic(action, state)
            server = heu.action
            feature_, reward, done, state_ = env.step(server, state)
            state = state_
            feature = feature_
            ep_reward += reward
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}')
    print('完成测试！')
    return {'rewards': rewards}


def all_seed(env, seed=1):
    """万能的seed函数"""
    if seed == 0:
        return
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python
    # config for cudnn GPU相关
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def env_agent_config(cfg):
    env = Environment(cfg.patient_path, cfg.server_path, cfg.avg_arrive_time)
    all_seed(env, seed=cfg.seed)
    agent = Agent(cfg)
    return env, agent


def smooth(data, weight=0.9):
    """用于平滑曲线，类似于Tensorboard中的smooth曲线"""
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards,cfg, tag='train'):
    """画图"""
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 获取参数
    cfg = Config(path='./config/05081341.json')
    # 训练
    env, agent = env_agent_config(cfg)
    best_agent, res_dic = train(cfg, env, agent)

    plot_rewards(res_dic['rewards'], cfg, tag='train')

    # 测试
    res_dic = test(cfg, env, best_agent)
    plot_rewards(res_dic['rewards'], cfg, tag='test')
