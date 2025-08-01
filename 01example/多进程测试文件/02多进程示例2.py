# 基于multiprocessing的多进程示例

from multiprocessing import Process
''' 直接使用Process调用不同的函数。
def fun1(name):
    print("子进程{}开始".format(name))
if __name__ == "__main__":
    process_list = []
    for i in range(5):
        p = Process(target=fun1, args = (f"Python{i}",))# name 的参数就是args
        p.start()
        process_list.append(p)
    for i in process_list: # 实现阻塞操作，在这里啊。
        i.join()
    print("end")

'''

class MyProcess(Process):
    def __init__(self,name):
        super(MyProcess,self).__init__()
        self.name = name
    def run(self):
        print(f"子进程{self.name}开始")

if __name__ == "__main__":
    process_list = []
    for i in range(5):
        p = MyProcess(f"Python{i}")
        p.daemon = True
        p.start() # 唯一的区别在于构建process的时候没有指定target所以说这里直接就是默认执行run
        process_list.append(p)
    for i in process_list:
        i.join()
    print("end")


"""
实例方法：
　　is_alive()：返回进程是否在运行,bool类型。
　　join([timeout])：阻塞当前上下文环境的进程程，直到调用此方法的进程终止或到达指定的timeout（可选参数）。
　　start()：进程准备就绪，等待CPU调度
　　run()：strat()调用run方法，如果实例进程时未制定传入target，这star执行t默认run()方法。
　　terminate()：不管任务是否完成，立即停止工作进程

属性：
　　daemon：守护进程，如果不执行join()方法，那么会在主进程结束之后自动结束执行。
　　name：进程名字
　　pid：进程号
"""
