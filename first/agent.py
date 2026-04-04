import torch
import torch.nn as nn
import torch.optim as optim
import random

from model import DQN
from config import *

class DQNAgent:
    def __init__(self,state_dim,action_dim,device):
        self.device=device
        self.action_dim=action_dim
        self.policy_net =DQN(state_dim,action_dim).to(device)
        self.target_net =DQN(state_dim,action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer =optim.Adam(self.policy_net.parameters(),lr=LR)

        self.gamma = GAMMA

    def select_action(self,state,epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        state =torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values).item())
    
    def train(self, replay_buffer,batch_size):
        if len(replay_buffer)<batch_size:
            return None
         
        s,a,r,s_,d =replay_buffer.sample(batch_size)
        
        s=torch.tensor(s,device=self.device)
        a=torch.tensor(a,device=self.device).unsqueeze(1)
        r=torch.tensor(r,device=self.device).unsqueeze(1)
        s_=torch.tensor(s_,device=self.device)
        d=torch.tensor(d,device=self.device).unsqueeze(1)

        q =self.policy_net(s).gather(1,a)

        with torch.no_grad():
            max_q_ = self.target_net(s_).max(1,keepdim=True)[0]
            target =r +self.gamma*max_q_*(1-d)

        loss = nn.MSELoss()(q,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

        

