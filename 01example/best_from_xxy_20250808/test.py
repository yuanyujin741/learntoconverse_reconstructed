from pentropy_main import Timer
timer = Timer()
import time
for episode_num in range(10):
    timer.episode_num = episode_num # 调用start之前更新episode_num才可以哦。
    remark = timer.start(timer.SOLVING_TIME)
    time.sleep(episode_num)
    remark["num_exprs"] = 1
    remark["num_vars"] = 2
    timer.end(remark)
print(timer.record)
"""
PS E:\pycharm\python_doc\learntoconverse_reconstructed> & E:/anaconda/envs/llama_factory/python.exe e:/pycharm/python_doc/learntoconverse_reconstructed/01example/best_from_xxy_20250808/test.py
[{'solving_time': {'time': 0.0, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 1.0153720378875732, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 2.0057380199432373, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 3.004077434539795, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 4.000191926956177, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 5.001958608627319, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 6.004347801208496, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 7.0126731395721436, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 8.00496244430542, 'remark': {'num_exprs': 1, 'num_vars': 2}}}, {'solving_time': {'time': 9.007729530334473, 'remark': {'num_exprs': 1, 'num_vars': 2}}}]
"""