# train.py
# 开始写代码于20250731
# 重构的原因是：
#    1. 原代码无法实现multipreocessing
#    2. 原代码无法实现GPU训练，改用torch实现目标
#    3. 原代码网络通信存在问题，长时间处理会导致问题
#    4. 原代码结构十分混乱，具有众多冗余参数，决定对代码进行重构，主要目标是实现multiprecessing
#    5. 环境本身也存在问题，但是因为网络通信的原因，通信困难，所以决定直接修改。
# 预计时间：本周五之前完成任务。
# task:
#   1.log——dir还没有建立
#   2.converseenv对应的一些全局变量可能存在相互干扰的现象。首先先研究一下是否会干扰，其次需要修改为类的变量。
#   3.修改render为"linux"和"windows"，对应的分别在linux和windows上面运行啊。
#   4.直接将numvars修改为ob_dim比较好，免去了来自原始的问题的联合的影响哎。现在没有修改，因为有点害怕改错了。

from utils import *
import os
import sys
# 添加llama_factory环境根目录到PATH（替换为你的环境实际路径）
env_root = "e:\\anaconda\\envs\\llama_factory"
torch_lib_path = "e:\\anaconda\\envs\\llama_factory\\Lib\\site-packages\\torch\\lib"
os.environ["PATH"] = torch_lib_path + ";" + os.environ["PATH"]
if env_root not in os.environ["PATH"]:
    os.environ["PATH"] = env_root + ";" + os.environ["PATH"]  # Windows用分号分隔

config = Config(DBM=True)
config.set_policy_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.set_attention_network_config(device = device)
print("device: ",device)
if config.DBM:
    config.print_config()
# 建立envs：
# envs = make_converse_env(config)
# 建立policy：

import os
print("Python文件 PATH:\n", os.environ["PATH"].split(";"))