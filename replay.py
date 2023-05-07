import random
from collections import deque  # deque是一种双端队列

'''经验回放'''


class ReplayBufferQue:
    """DQN的经验回放池，每次采样batch_size个样本"""
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        """
        transitions (tuple)
        """
        self.buffer.append(transitions)  # 与list的append()方法类似

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):  # 当前的经验回放池还没有填满
            batch_size = len(self.buffer)
        if sequential:  # sequential sampling, 连续采样
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)  # 返回由transition每一项组成的列表

    def clear(self):
        """clear the buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PGReplay(ReplayBufferQue):
    """
    PG的经验回放池，每次采样所有样本，因此重写sample方法
    """
    def __init__(self):
        self.buffer = deque()

    def sample(self):
        """sample all the transitions"""
        batch = list(self.buffer)
        return zip(*batch)

