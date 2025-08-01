from multiprocessing import Pool, cpu_count
import time
import random
import os
import sys

# 任务函数（每个工作进程执行的任务）
def worker_task(task_id, num_workers):
    """
    模拟任务处理，返回处理结果
    :param task_id: 任务ID
    :param num_workers: 工作进程数量
    :return: (任务ID, 处理结果)
    """
    # 获取当前工作进程的PID
    pid = os.getpid()
    logpath = f"{task_id}_log.txt"
    with open(logpath,"w+") as logfile:
        sys.stdout = logfile
        print(f"Worker PID {pid} (index {pid % num_workers}) started task {task_id}")

        # 模拟任务处理时间（0~2秒）
        processing_time = random.uniform(0, 2)
        time.sleep(processing_time)

        # 模拟计算结果
        result = task_id ** 2
    return (task_id, result)

if __name__ == '__main__':
    # 配置参数
    total_tasks = 6      # 总任务数量
    num_workers = 3     # 工作进程数量
    
    print(f"Main: Starting with {num_workers} workers for {total_tasks} tasks")
    
    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 提交所有任务到进程池（异步非阻塞）
        task_ids = range(total_tasks)
        async_results = [pool.apply_async(worker_task, (t, num_workers)) for t in task_ids]
        
        print("Main: All tasks submitted. Waiting for results...")
        
        # 获取并处理结果（按完成顺序）
        for async_result in async_results:
            # 阻塞等待单个任务完成
            task_id, result = async_result.get()
            print(f"Main: Received result from task {task_id}: {result}")
    
    print("Main: All tasks completed")