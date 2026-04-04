import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer =deque(maxlen=capacity)
    
    def push(self,s,a,r,s_,done):
        self.buffer.append((s,a,r,s_,done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer,batch_size)
        s,a,r,s_,d =zip(*batch)

        return (
            np.array(s, dtype=np.float32),
            np.array(a),
            np.array(r,dtype=np.float32),
            np.array(s_,dtype=np.float32),
            np.array(d,dtype=np.float32),
        )
    def __len__(self):
        return len(self.buffer)
    
