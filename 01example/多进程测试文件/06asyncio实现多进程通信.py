import asyncio
from multiprocessing import Process,Pipe

def producer(pipe):
    asyncio.run(_wait_for_remote(pipe))

async def _wait_for_remote(pipe):
    result = pipe.recv()
    await asyncio.sleep(1)
    pipe.send({"recv":result, "processed":True})

if __name__ == "__main__":
    conn1, conn2 = Pipe()
    p = Process(target=producer, args=(conn2,))
    p.start()
    conn1.send("hello from main")
    result = conn1.recv()
    print(result)
    p.join()
