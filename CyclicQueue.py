import numpy as np


class CyclicQueue:
    def __init__(self, max_size, min_rat=0.8):
        self.maxSize = max_size
        self.queue = [None] * self.maxSize
        self.minRat = min_rat
        self.lastAccessed = False

    def init_queue(self, elements):
        tmp = elements.copy()
        tmp.reverse()
        for i in tmp:
            self.add(i)

    def add(self, ele):
        del self.queue[-1]
        self.queue.insert(0, ele)
        self.lastAccessed = False

    def to_string(self):
        res = ""
        for ele in self.queue:
            res = f"{res}{str(ele)}"
        return res

    def mean(self):
        tmp = self.queue.copy()
        if tmp.count(None) >= self.minRat * self.maxSize:
            return None
        else:
            tmp = list(filter(None, tmp))
        if isinstance(tmp[0], list):
            return np.mean(tmp, axis=0)
        else:
            temp = list(filter(None, tmp))
            return sum(temp) / len(temp)

    def median(self, init_index=0, final_index=None):
        final_index = self.maxSize if final_index is None else final_index
        tmp = self.queue[init_index:final_index].copy()
        if tmp.count(None) >= self.minRat * self.maxSize:
            return None
        else:
            temp = list(filter(None, tmp))
            temp.sort()
            return temp[(final_index - init_index - 1) // 2 + 1]

    def mode(self):
        temp = self.queue.copy()
        if temp.count(None) >= self.minRat * self.maxSize:
            return None
        mx = 0
        res = None
        for ele in temp:
            curr = temp.count(ele)
            if curr > mx:
                res = ele
                mx = curr
        return res

    def size(self):
        return len(self.queue)

    def get_last(self):
        self.lastAccessed = True
        return self.queue[0]

    def get_list(self):
        return self.queue

    def get_last_accessed(self) -> bool:
        return self.lastAccessed

    def replace_index(self, ele, index):
        self.queue[index] = ele
