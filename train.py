# train.py
# 开始写代码于20250731
# 重构的原因是：
#    1. 原代码无法实现multipreocessing
#    2. 原代码无法实现GPU训练，改用torch实现目标
#    3. 原代码网络通信存在问题，长时间处理会导致问题
#    4. 原代码结构十分混乱，具有众多冗余参数，决定对代码进行重构，主要目标是实现multiprecessing
#    5. 环境本身也存在问题，但是因为网络通信的原因，通信困难，所以决定直接修改。
# 预计时间：本周五之前完成任务。
# 出现的问题：1. （cuda识别问题）https://blog.csdn.net/weixin_74188799/article/details/149826307?fromshare=blogdetail&sharetype=blogdetail&sharerId=149826307&sharerefer=PC&sharesource=weixin_74188799&sharefrom=from_link
# task:
#   1.log——dir还没有建立
#   2.converseenv对应的一些全局变量可能存在相互干扰的现象。首先先研究一下是否会干扰，其次需要修改为类的变量。
#   3.修改render为"linux"和"windows"，对应的分别在linux和windows上面运行啊。
#   4.直接将numvars修改为ob_dim比较好，免去了来自原始的问题的联合的影响哎。现在没有修改，因为有点害怕改错了。
#   5.设置种子嘛？？
#   6.model的pth文件的加载没有测试。
#   7.在config中添加测试说明。
# test done:
#   1.对log的建立、保存、加载测试完成。继续训练的log加载测试完成。
#   

# 初始化config，也就是全部的配置。
from utils import *
config = Config(continue_training=False, DBM=True)
config.set_policy_config()
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.set_attention_network_config(device = device)
print("device: ",device)
if config.DBM:
    config.print_config()
# 建立envs：
envs = make_converse_env(config)
# 建立policy和optimizer：
policy = AttentionPolicy(config.network_param, device = device)
optimizer = torch.optim.Adam(policy.parameters(), lr = config.learning_rate)
# 初始化存储对象
log = create_new_log(config)

# 从pretrained_model_id加载模型
if config.use_pretrained_model:
    policy.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/model.pth"))
    optimizer.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/optimizer.pth"))
# 继续训练
elif config.continue_training:
    policy.load_state_dict(torch.load(f"02all_data/{config.task_id}/model.pth"))
    optimizer.load_state_dict(torch.load(f"02all_data/{config.task_id}/optimizer.pth"))
    log = load_log(config)
    add_new_meta_data(log, config)
    if config.DBM:
        print(log)
# 从头开始训练不需要额外处理就是了
# 开始训练
