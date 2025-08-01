from concurrent.futures import ProcessPoolExecutor
import time
import random

def compute_heavy_task(env_id, action):
    # 模拟一个计算密集型任务
    randomnum = random.randint(1,10)
    time.sleep(randomnum)  # 假设任务需要2秒完成
    print(f"{env_id}", time.time()-starttime)
    return f"Task {env_id}, {action} completed, randomnum: {randomnum}"

starttime = time.time()
if __name__ == "__main__":
    # 提交多个任务到进程池
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compute_heavy_task, 1, "action1"),
            executor.submit(compute_heavy_task, 2, "action2"),
            executor.submit(compute_heavy_task, 3, "action3"),
            executor.submit(compute_heavy_task, 4, "action4"),
        ]
        endtime = time.time()
        print(endtime - starttime)
        for future in futures:
            # 实际上卡在这里了。
            # 顺序打印结果，如果前面一个没有结果，那么即使后面一个有了结果也不会继续运行的。
            print(future.result())
"""
0.3890242576599121
2 2.0032315254211426
4 3.011059522628784
1 7.016260862350464
Task 1, action1 completed, randomnum: 7      
Task 2, action2 completed, randomnum: 2      
3 7.01068639755249
Task 3, action3 completed, randomnum: 7      
Task 4, action4 completed, randomnum: 3 
"""