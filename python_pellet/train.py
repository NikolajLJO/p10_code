import sys
import multiprocessing as mp

from actor import Actor
from memory import ReplayMemory

if __name__ == "__main__":
    mp.set_start_method('spawn')
    thread_count = 4
    replay_memory = ReplayMemory()
    with mp.Pool(processes=thread_count) as pool:
        replay_que = mp.Queue()
        partition_que = mp.Queue()
        process_list = [mp.Process(target=Actor, args=(sys.argv, x, replay_que, partition_que)) for x in range(thread_count)]

        for process in process_list:
            process.start()
