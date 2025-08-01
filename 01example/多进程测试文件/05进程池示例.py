# 进程池可以实现任务的动态分配，将多个任务分别分配给不同的进程处理
# 直接由池进行进程管理，不太方便进行进程间通信。

from multiprocessing import Process,Pool
import os, time, random

def func(name):
    print("子进程{}开始, pid={}".format(name,os.getpid()))
    start_time = time.time()
    time.sleep(random.randint(1,3))
    end_time = time.time()
    print("子进程{}结束，耗时{}秒".format(name,end_time-start_time))

if __name__ == "__main__":
    pool = Pool(3)
    for i in range(10):
        pool.apply_async(func = func, args = (i,))
    pool.close() # 关闭进程池，不再接收“新”的任务
    pool.join() # 等待所有子进程完成
    print("主进程结束")
"""
子进程0开始, pid=26412
子进程1开始, pid=2308
子进程2开始, pid=25828
子进程0结束，耗时1.003891944885254秒
子进程3开始, pid=26412
子进程1结束，耗时1.012702226638794秒
子进程4开始, pid=2308
子进程4结束，耗时1.0137712955474854秒
子进程5开始, pid=2308
子进程2结束，耗时2.0076863765716553秒
子进程6开始, pid=25828
子进程3结束，耗时2.0077011585235596秒
子进程7开始, pid=26412
子进程5结束，耗时1.0109121799468994秒
子进程8开始, pid=2308
子进程6结束，耗时2.0147831439971924秒
子进程9开始, pid=25828
子进程7结束，耗时3.001163959503174秒
子进程8结束，耗时3.0009765625秒
子进程9结束，耗时2.0130066871643066秒
主进程结束
"""