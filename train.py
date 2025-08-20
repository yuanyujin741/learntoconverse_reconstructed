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
#   3.修改render为"linux"和"windows"，对应的分别在linux和windows上面运行啊；不同的平台对于torch的要求也不太一样哩，就是expr2vecs的时候需要指定cuda的id。
#   4.直接将numvars修改为ob_dim比较好，免去了来自原始的问题的联合的影响哎。现在没有修改，因为有点害怕改错了。
#   5.设置种子嘛？？
#   6.model的pth文件的加载没有测试。
#   7.在config中添加测试说明。
#   8.rollout_length的效果不知道怎么样子。可能没有效果，也就是没有传递到envs中，可能没有效果。
#   9.实际使用之前需要确定一下expr2vec的可用性。
# test done:
#   1.对log的建立、保存、加载测试完成。继续训练的log加载测试完成。
#   

# 是的，这里是多进程保护哎：
if __name__ == "__main__":
    # 初始化config，也就是全部的配置。
    from utils import *
    multiprocessing.set_start_method('spawn')  # 关键修复。似乎也可以直接使用gpu进行模型训练了。
    config = Config(continue_training=False, use_pretrained_model=True, pretrained_model_id="007",pretrained_model_checkpoint = 0, envs_NK=[[2,2],[2,3]],new_ineq_num_factor=0.3,num_worker=1, DBM=True, rewardtype = "original",
                    test_mode=False, test_subject="without_FiLM", test_task_id="009", test_task_models=[0,50,100,150,200,250,300,350,"latest"], test_num = 2)# 确实我自己是有点担心电脑会炸掉，一直这么高强度训练的话。,[3,2],[3,3],[2,4],[4,2],[4,3],[4,4]
    config.set_policy_config(using_FiLM=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_attention_network_config(device = device)
    print("device: ",device)
    if config.test_mode == False:
        set_seed(config)
    if config.DBM:
        config.print_config()
    # 建立envs：
    envs = make_converse_env(config)
    # 建立policy和optimizer：
    policy = AttentionPolicy(config.network_param, device = device, using_FiLM = config.using_FiLM)
    optimizer = torch.optim.Adam(policy.parameters(), lr = config.learning_rate)
    # 初始化存储对象
    log = create_new_log(config)

    # 从pretrained_model_id加载模型
    if config.use_pretrained_model:
        if config.pretrained_model_checkpoint == "latest":
            policy.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/model.pth"),strict=False)
            #optimizer.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/optimizer.pth"))
        else:
            policy.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/checkpoint/model_{config.pretrained_model_checkpoint}.pth"),strict=False)
            #optimizer.load_state_dict(torch.load(f"02all_data/{config.pretrained_model_id}/checkpoint/optimizer_{config.pretrained_model_checkpoint}.pth"))

    # 继续训练
    elif config.continue_training:
        policy.load_state_dict(torch.load(f"02all_data/{config.task_id}/model.pth"))
        optimizer.load_state_dict(torch.load(f"02all_data/{config.task_id}/optimizer.pth"))
        log = load_log(config)
        add_new_meta_data(log, config)
        if config.DBM:
            print(log)

    if not config.test_mode: # 如果不是test_mode的话就这样做：
        # 从头开始训练不需要额外处理就是了
        # 开始训练
        current_rollout_num = len(log['results']) # 适配了其他的问题哎
        while current_rollout_num < config.max_rollout_num:
            config.set_intime_config(episode_num=current_rollout_num)
            # 做不同epsilon的rollout
            rollout_start_time = time.time()
            epsilon_table = create_epsilon_table(config=config, policy=policy) # create epsilon table
            results = rollout_workers(envs_s=envs, epsilon_table=epsilon_table, original_policy=policy, config=config)
            # 保存处理的结果
            save_results_2_log(results, log, config = config)
            log["rollout_time"].append(time.time() - rollout_start_time)
            print(log)
            # 参数更新，修改为使用optimizer实现参数更新，而不是手动设置之类的。
            grad = get_grad(epsilon_table, config, log)
            optimizer.zero_grad()
            for name, param in policy.named_parameters():
                if name in grad:
                    param.grad = - grad[name].to(param.device)
                else:
                    print("warning! no grad for ", name)
            optimizer.step()
            log["clocktime"].append(time.time() - rollout_start_time)
            # 模型保存以及log保存（注意环境的输出本来就是保存起来的）
            save_log(config=config, log = log)
            torch.save(policy.state_dict(), f"02all_data/{config.task_id}/model.pth")
            torch.save(optimizer.state_dict(), f"02all_data/{config.task_id}/optimizer.pth")
            if current_rollout_num % 50 == 0:
                os.makedirs(f"02all_data/{config.task_id}/checkpoint", exist_ok=True)
                torch.save(policy.state_dict(), f"02all_data/{config.task_id}/checkpoint/model_{current_rollout_num}.pth")
                torch.save(optimizer.state_dict(), f"02all_data/{config.task_id}/checkpoint/optimizer_{current_rollout_num}.pth")
            # 可视化处理
            draw_rewards_20points(log, config)
            draw_times(log, config)
            current_rollout_num += 1
    else:
        envs = make_test_env(config) # test done
        results = rollout_test_workers(envs_s=envs,original_policy=policy,config=config)
        explain_test_results(results=results, config=config)
        print("TEST DONE")