# utils.py


# region general
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import os
import time
import copy
import pdb

class Config():
    def __init__(self, DBM = True, continue_training = False, envs_NK=[[2,2],[2,3],[3,2],[3,3]], env_config={"ob_dim":384 - 1}, use_pretrained_model=False, pretrained_model_id = None, new_ineq_num_factor = 0.3, num_worker = 3):
        self.DBM = DBM
        self.continue_training = continue_training
        self.set_task_id() # 基于是否继续训练设置一个合理的id
        self.envs_NK = envs_NK
        self.env_config = env_config
        self.use_pretrained_model = use_pretrained_model
        self.pretrained_model_id = pretrained_model_id
        assert self.use_pretrained_model == False or self.pretrained_model_id != None
        self.new_ineq_num_factor = new_ineq_num_factor
        self.log_dir = f"02all_data/{self.task_id}"
        self.num_worker = num_worker
        if self.DBM:
            print("finished general settings.")

    def set_task_id(self):
        # 设置数据目录路径（根据实际情况调整）
        data_dir = "02all_data"
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 获取目录中所有条目并筛选出三位数字格式的ID
        task_ids = []
        for entry in os.listdir(data_dir):
            if entry.isdigit() and len(entry) == 3:  # 假设ID是三位数字格式
                task_ids.append(int(entry))

        # 生成新ID（如果没有现有ID则从000开始，否则取最大ID+1）
        new_id = max(task_ids) + 1 if task_ids else 0
        if self.continue_training:
            self.task_id = f"{new_id - 1:03d}"  # 格式化为三位数字字符串
        else:
            self.task_id = f"{new_id:03d}"  # 格式化为三位数字字符串
    
    def set_policy_config(self, max_epoch_num = 1000, n_directions = 4, evaluate_time = 3, learning_rate = 0.01, gamma = 0.97, std = 0.02):
        """
        :param: max_epoch_num: 最大epoch数，也就是最大进行多少次全部envs的rollout；取代了max_step
        """
        self.max_epoch_num = max_epoch_num
        self.n_directions = n_directions
        assert n_directions%2 == 0, "n_directions must be even"
        self.evaluate_time = evaluate_time
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.std = std

    def print_config(self):
        print('-'*3,"config",'-'*3)
        for key, value in self.__dict__.items():
            print(f"* {key}: {value}")
        print('-'*3,"------",'-'*3)

    def set_attention_network_config(self,device = "cpu"):
        """
        directly from the original value.
        """
        self.network_param = {}
        self.network_param['hsize'] = 64
        self.network_param['numlayers'] = 2
        self.network_param['embed'] = 100
        #self.network_param['rowembed'] = 200
        self.device = device

def set_seed(args):
    """
    注意给env也要设置seed啊：
    state, _ = env.reset(seed=args.seed)
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# region envs
from converseenv import ConverseEnv
class ConverseEnvWrapper(ConverseEnv):
    def __init__(self, N=3,K=3,RENDER = "dont_render",epsilon_id=0):
        super().__init__(N=N,K=K,RENDER=RENDER)
        self.epsilon_id = epsilon_id
        
def make_same_epsilon_env(config,epsilon_id):
    envs = []
    for NK in config.envs_NK:
        env = ConverseEnvWrapper(N = NK[0], K = NK[1],epsilon_id=epsilon_id)
        envs.append(env)
    return envs

def make_converse_env(config = Config()):
    envs = []
    total_epsilon = config.n_directions + config.evaluate_time
    for epsilon_id in range(total_epsilon):
        envs.append(make_same_epsilon_env(config,epsilon_id))
    if config.DBM:
        print(f"total envs: {[[[env.N,env.K,env.epsilon_id] for env in env_s] for env_s in envs]}")
    return envs

# region policy
class AttentionPolicy(nn.Module):
    """
    这一部分代码由ai直接使用pytorch重构。嗯，deepseek-R1。
    因为不是自己写的，所以这里对config的使用不足，就这样吧，懒得改了。
    对代码进行检查。
    """
    def __init__(self, policy_params):
        super().__init__()
        self.numvars = policy_params['numvars']
        hsize = policy_params['hsize']
        numlayers = policy_params['numlayers']
        embeddeddim = policy_params['embed']

        # 参数（NK）编码器
        self.param_embed_dim = 4
        self.param_encoder = nn.Sequential(
            nn.Linear(2, self.param_embed_dim),
            nn.Tanh()
        )

        # 构建MLP和FiLM生成器
        self.layers = nn.ModuleList() # 这个是映射层哎。
        self.film_generators = nn.ModuleList()# 这个是FiLM生成器，用于生成每个层的scale和bias。
        
        input_dim = self.numvars + 1 # 输入维度处理
        for i in range(numlayers):
            if i == 0:
                layer = nn.Linear(input_dim, hsize)
            else:
                layer = nn.Linear(hsize, hsize)
            self.layers.append(layer)
            
            # FiML生成器
            film_gen = nn.Sequential(
                nn.Linear(self.param_embed_dim, 2*hsize),
                nn.Tanh()
            )
            self.film_generators.append(film_gen)
        
        # 输出层
        self.final_layer = nn.Linear(hsize, embeddeddim)
        
        # 初始化权重
        self._init_weights()
        
        # 保持filter兼容，不使用filter。
        #self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.numvars+1,))
        
        self.t = 0

    def _init_weights(self):
        """
        对全部的非重复的nn.Linear进行初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def apply_film(self, x, param_embed, layer_idx):
        film_params = self.film_generators[layer_idx](param_embed)
        scale, bias = torch.chunk(film_params, 2, dim=-1)
        return scale * x + bias

    def forward(self, ob, update=True, num=1, n_value=None, k_value=None) -> list:
        t1 = time.time()
        A, b, c0, cutsa, cutsb = ob
        
        # 数据预处理
        baseob_original = torch.from_numpy(np.column_stack((A, b))).float()
        ob_original = torch.from_numpy(np.column_stack((cutsa, cutsb))).float()
        
        try:
            totalob_original = torch.cat([baseob_original, ob_original], dim=0)
        except Exception as e:
            print(f"Error concatenating: {e}")
            return []
        
        # 数据归一化
        totalob_original = (totalob_original - totalob_original.min()) / (totalob_original.max() - totalob_original.min() + 1e-8)
        
        # 应用filter
        totalob_np = self.observation_filter(totalob_original.numpy(), update=update)
        totalob = torch.from_numpy(totalob_np).float()
        
        # 分割数据
        baseob = totalob[:A.shape[0]]
        ob = totalob[A.shape[0]:]
        
        # 参数编码
        NK = torch.tensor([[n_value, k_value]], dtype=torch.float)
        NK_embed = self.param_encoder(NK)
        
        # 前向传播
        x_base = baseob
        for i, layer in enumerate(self.layers):
            x_base = layer(x_base)
            x_base = F.tanh(x_base)
            x_base = self.apply_film(x_base, NK_embed, i)
        base_embed = self.final_layer(x_base)
        
        x_ob = ob
        for i, layer in enumerate(self.layers):
            x_ob = layer(x_ob)
            x_ob = F.tanh(x_ob)
            x_ob = self.apply_film(x_ob, NK_embed, i)
        cut_embed = self.final_layer(x_ob)
        
        # 计算attention
        attention_map = torch.mm(cut_embed, base_embed.t())
        score = attention_map.mean(dim=1)
        
        # 概率计算
        score = score - score.max()
        prob = F.softmax(score, dim=0)
        
        # 动作选择
        if num == 1:
            action = torch.multinomial(prob, 1).item()
        else:
            _, indices = torch.topk(prob, num)
            action = indices.tolist()
        
        self.t += time.time() - t1
        return action

    def get_weights(self):
        return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

    def update_weights(self, weights_dict):
        self.load_state_dict({k: torch.from_numpy(v) for k, v in weights_dict.items()})

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        return self.get_weights(), mu, std