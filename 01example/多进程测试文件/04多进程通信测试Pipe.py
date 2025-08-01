# 最好的地方是实现了双向通信。
# 每个子进程和主进程有一个pipe实现双向通信
# linux上测试通过。

import numpy as np
from multiprocessing import Process,Pipe
import select
import time
END_FLAG = {"end": True}

def func(conn,i):
    j = 0
    while True:
        print("子进程{}发送消息".format(i))
        if np.random.rand() < 0.3:
            conn.send(END_FLAG)
        else:
            conn.send({"subprocess_name":i,"value":np.array([i,i**2]),"message_j":j})
        print("子进程{}接收消息".format(i))
        recv_data_atsub = conn.recv()  # 因为没有接收到消息，所以子进程会在这里等候信息到来。
        if recv_data_atsub == END_FLAG:
            print("子进程{}结束".format(i))
            break
        print(recv_data_atsub)
        time.sleep((5-i)*30) # 判断是否具有多监听的能力。
        j += 1
    conn.close()

if __name__ == "__main__":
    process_list = []
    conn1_list = []
    conn2_list = []
    for i in range(3):
        conn1,conn2 = Pipe()
        p = Process(target = func, args = (conn2,i))
        p.start()
        process_list.append(p)
        conn1_list.append(conn1)
        conn2_list.append(conn2)
    active_connections = {conn1:True for conn1 in conn1_list} # 用来判断何时结束通信，从而结束主进程
    while any(active_connections.values()):
        active_conn_list = [conn1 for conn1,value in active_connections.items() if value]
        if active_conn_list == []:
            break
        # 两个空列表 []：分别忽略对可写状态和异常状态的监控；返回值 readable：包含所有有数据待读取的管道连接（子进程发来了数据）
        readable, _, _ = select.select(active_conn_list,[],[])
        for conn1 in readable:
            recv_data_atmain = conn1.recv()
            if recv_data_atmain == END_FLAG:
                conn1.send(END_FLAG)
                active_connections[conn1] = False
            else:
                print(recv_data_atmain)
                conn1.send({"subprocess_name":i,"value":np.array([i,i**2]),"message_j":recv_data_atmain["message_j"]})
    print("主进程结束")


"""
select.select() 默认会阻塞等待，直到以下任一条件满足：
    至少有一个被监控的连接变为可读（即有数据到达）
    被信号中断
    如果设置了超时参数，超时时间到达
"""