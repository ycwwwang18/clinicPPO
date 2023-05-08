import json


class Config:
    def __init__(self, path=None) -> None:
        if path is None:
            self.env_name = "Clinic"  # 环境名字
            self.new_step_api = False  # 是否用gym的新api
            self.algo_name = "PPO"  # 算法名字
            self.mode = "train"  # train or test
            self.seed = 2  # 随机种子
            self.device = "cpu"  # device to use
            self.train_eps = 2000  # 训练的回合数
            self.test_eps = 20  # 测试的回合数
            # self.max_steps = 200 # 每个回合的最大步数
            self.eval_eps = 5  # 评估的回合数
            self.eval_per_episode = 10  # 评估的频率

            self.gamma = 0.99  # 折扣因子
            self.k_epochs = 4  # 更新策略网络的次数
            self.actor_lr = 0.0003  # actor网络的学习率
            self.critic_lr = 0.0003  # critic网络的学习率
            self.eps_clip = 0.2  # epsilon-clip
            self.entropy_coef = 0.01  # entropy的系数
            self.update_freq = 100  # 更新频率
            self.actor_hidden_dim = 256  # actor网络的隐藏层维度
            self.critic_hidden_dim = 256  # critic网络的隐藏层维度

            self.n_states = 6  # 状态的维度
            self.n_actions = 7  # 动作的数量
            self.patient_path = './data/patient_20_same_path.xlsx'  # 患者文件路径
            self.server_path = './data/server information1.xlsx'  # 服务台文件路径
            self.avg_arrive_time = 3  # 患者平均到达时间间隔
        else:
            self.import_json(path)

    def export_json(self, path):
        """将参数配置导出为json文件"""
        config = {
            "env_name": self.env_name,
            "new_step_api": self.new_step_api,
            "algo_name": self.algo_name,
            "mode": self.mode,
            "seed": self.seed,
            "device": self.device,
            "train_eps": self.train_eps,
            "test_eps": self.test_eps,
            "eval_eps": self.eval_eps,
            "eval_per_episode": self.eval_per_episode,
            "gamma": self.gamma,
            "k_epochs": self.k_epochs,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "eps_clip": self.eps_clip,
            "entropy_coef": self.entropy_coef,
            "update_freq": self.update_freq,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "patient_path": self.patient_path,
            "server_path": self.server_path,
            "avg_arrive_time": self.avg_arrive_time
        }
        f = open(path, 'w')
        json.dump(config, f, indent=2)

    def import_json(self, path):
        f = open(path, 'r')
        config = json.load(f)
        self.env_name = config['env_name']
        self.new_step_api = config['new_step_api']
        self.algo_name = config['algo_name']
        self.mode = config['mode']
        self.seed = config['seed']
        self.device = config['device']
        self.train_eps = config['train_eps']
        self.test_eps = config['test_eps']
        self.eval_eps = config['eval_eps']
        self.eval_per_episode = config['eval_per_episode']

        self.gamma = config['gamma']
        self.k_epochs = config['k_epochs']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.eps_clip = config['eps_clip']
        self.entropy_coef = config['entropy_coef']
        self.update_freq = config['update_freq']
        self.actor_hidden_dim = config['actor_hidden_dim']
        self.critic_hidden_dim = config['critic_hidden_dim']

        self.n_states = config['n_states']
        self.n_actions = config['n_actions']
        self.patient_path = config['patient_path']
        self.server_path = config['server_path']
        self.avg_arrive_time = config['avg_arrive_time']


if __name__ == "__main__":
    cfg = Config(path='./config/05081341.json')
    cfg.export_json("05081341.json")
