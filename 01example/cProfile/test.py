import cProfile
import time

def my_function():
    # 需要分析的逻辑
    for i in range(1000000):
        _ = i**2
    done()

def done():
    time.sleep(1)

cProfile.run("my_function()")
