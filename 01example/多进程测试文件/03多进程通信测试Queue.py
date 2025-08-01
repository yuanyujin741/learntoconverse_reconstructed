# 这里讲的是使用Queue，也就是管道进行通信的方案啊。
# 局限：无法双向通信。

# 每个子进程都可以在自己的进程中将一些数据put进入Queue中，数据是任意的，所以可以实现标识与综合处理。
# 似乎出去之后无法再次获取，所以只能由主进程获得子进程的信息。

from multiprocessing import Process,Queue
import numpy as np

def fun(q,i):
    print("子进程{}开始put数据".format(i))
    q.put({"subprocess_name":i,"value":np.array([i,i**2])})

if __name__ == '__main__':
    q = Queue()
    process_list = []
    for i in range(3):
        p = Process(target=fun,args=(q,i),name=f"{i}")
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    print("主进程获得queue数据")
    q_data = []
    for _ in range(3):
        q_data.append(q.get())
    print(q_data)
    print("主进程结束")