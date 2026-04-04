import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import panda_gym
from collections import deque
import random
from torch import amp
import matplotlib.pyplot as plt

# ==========================================
# 1. 4060 优化版超参数 (严格对齐论文逻辑)
# ==========================================
ENV_NAME = "PandaPickAndPlace-v3" 
GAMMA = 0.99
TAU = 0.005             # 严格回归论文 0.005
LR = 2e-4              
BATCH_SIZE = 128       
BUFFER_SIZE = 500000   
HER_K = 5              # 论文设定 K=5
MAP_SIZE = 16          
ALPHA_INIT = 0.2
WARMUP_STEPS = 10000 
MAX_STEPS = 50
MAX_EPISODES = 40000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存路径
SAVE_DIR = "GeneWorker_Results"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_FILE = os.path.join(SAVE_DIR, "train_curve.png")

# ==========================================
# 2. GeneWorker 模型
# ==========================================
class GeneWorkerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(GeneWorkerActor, self).__init__()
        self.max_action = max_action
        input_dim = state_dim + goal_dim
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.to_map = nn.Linear(256, MAP_SIZE * MAP_SIZE)
        
        self.worker_cfg = [
            {'in': input_dim, 'out': 128},
            {'in': 128, 'out': 64},
            {'in': 64, 'out': 64},
            {'in': 64, 'out': action_dim * 2}
        ]
        
        self.w_gen = nn.ModuleList()
        self.b_gen = nn.ModuleList()
        for cfg in self.worker_cfg:
            self.w_gen.append(nn.Sequential(
                nn.Conv2d(1, 2, kernel_size=3, padding=1),
                nn.ReLU(), nn.Flatten(),
                nn.Linear(2 * MAP_SIZE * MAP_SIZE, cfg['in'] * cfg['out'])
            ))
            self.b_gen.append(nn.Linear(256, cfg['out']))

    def forward(self, state, goal):
        combined = torch.cat([state, goal], dim=-1)
        f_t_vec = self.generator(combined)
        f_t_map = self.to_map(f_t_vec).view(-1, 1, MAP_SIZE, MAP_SIZE)
        
        x = combined
        for i in range(len(self.worker_cfg)):
            w = self.w_gen[i](f_t_map).view(-1, self.worker_cfg[i]['in'], self.worker_cfg[i]['out'])
            b = self.b_gen[i](f_t_vec).unsqueeze(1)
            x = torch.bmm(x.unsqueeze(1), w) + b
            x = x.squeeze(1)
            if i < len(self.worker_cfg) - 1: x = F.relu(x)
        
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample_action(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        lp = (normal.log_prob(x_t) - torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)).sum(1, keepdim=True)
        return action * self.max_action, lp

class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        input_dim = state_dim + goal_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], dim=-1)
        return self.q1(sa), self.q2(sa)

# ==========================================
# 3. 绘图辅助函数 (新增)
# ==========================================
def save_curve(rewards, success_rates, path):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Episode Reward', color='blue', alpha=0.6)
    plt.axhline(y=-10, color='r', linestyle='--', label='Success Threshold')
    plt.title('GeneWorker Training Progress')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    # 计算滑动平均成功率
    if len(success_rates) > 100:
        ma_success = np.convolve(success_rates, np.ones(100)/100, mode='valid')
        plt.plot(ma_success, label='Success Rate (MA 100)', color='green')
    else:
        plt.plot(success_rates, label='Success Rate', color='green')
    plt.ylabel('Rate')
    plt.xlabel('Episode')
    plt.ylim([0, 1.1])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')

# ==========================================
# 4. 训练主逻辑
# ==========================================
def train():
    torch.backends.cudnn.benchmark = True
    scaler = amp.GradScaler() # 修正 API

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = GeneWorkerActor(state_dim, goal_dim, action_dim, max_action).to(DEVICE)
    critic = Critic(state_dim, goal_dim, action_dim).to(DEVICE)
    critic_target = Critic(state_dim, goal_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    
    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)
    log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=DEVICE)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR)
    
    buffer = deque(maxlen=BUFFER_SIZE // MAX_STEPS) 
    total_steps = 0
    
    # 统计记录 (新增)
    metrics = {'rewards': [], 'success': []}
    best_avg_reward = -float('inf')

    print(f"🚀 4060 复现版启动 | 任务: {ENV_NAME} | 设备: {DEVICE}")

    for ep in range(MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_data = []
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            s_t = torch.FloatTensor(obs_dict['observation']).unsqueeze(0).to(DEVICE)
            g_t = torch.FloatTensor(obs_dict['desired_goal']).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action, _ = actor.sample_action(s_t, g_t)
                    action = action.cpu().numpy()[0]
            
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_data.append({
                'obs': obs_dict['observation'], 'act': action, 'rew': reward,
                'next_obs': next_obs_dict['observation'], 'next_ag': next_obs_dict['achieved_goal'],
                'dg': obs_dict['desired_goal'], 'done': float(terminated or truncated),
                'is_success': info.get('is_success', 0.0)
            })
            obs_dict = next_obs_dict
            episode_reward += reward
            total_steps += 1
            if terminated or truncated: break
        
        buffer.append(episode_data)
        metrics['rewards'].append(episode_reward)
        metrics['success'].append(episode_data[-1]['is_success'])

        if total_steps >= WARMUP_STEPS:
            for _ in range(20): 
                # HER 采样
                batch = []
                for _ in range(BATCH_SIZE):
                    # 修正：将这里原有的变量名 ep 改为 sampled_ep，防止覆盖计数器
                    sampled_ep = random.choice(buffer)
                    t = random.randint(0, len(sampled_ep)-1)
                    trans = sampled_ep[t]
                    if random.random() < HER_K/(HER_K+1):
                        ft = random.randint(t, len(sampled_ep)-1)
                        new_g = sampled_ep[ft]['next_ag']
                        rew = env.unwrapped.compute_reward(trans['next_ag'], new_g, {})
                        batch.append((trans['obs'], trans['act'], rew, trans['next_obs'], new_g, trans['done']))
                    else:
                        batch.append((trans['obs'], trans['act'], trans['rew'], trans['next_obs'], trans['dg'], trans['done']))
                
                s, a, r, ns, g, d = [torch.FloatTensor(np.array(x)).to(DEVICE) for x in zip(*batch)]
                r, d = r.unsqueeze(1), d.unsqueeze(1)
                alpha = log_alpha.exp()

                with amp.autocast('cuda'):
                    with torch.no_grad():
                        next_a, next_lp = actor.sample_action(ns, g)
                        t_q1, t_q2 = critic_target(ns, g, next_a)
                        target_q = r + (1-d) * GAMMA * (torch.min(t_q1, t_q2) - alpha * next_lp)
                    c_q1, c_q2 = critic(s, g, a)
                    c_loss = F.mse_loss(c_q1, target_q) + F.mse_loss(c_q2, target_q)
                
                critic_opt.zero_grad(); scaler.scale(c_loss).backward()
                scaler.unscale_(critic_opt); nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler.step(critic_opt)

                with amp.autocast('cuda'):
                    new_a, lp = actor.sample_action(s, g)
                    q1, q2 = critic(s, g, new_a)
                    a_loss = (alpha * lp - torch.min(q1, q2)).mean()
                
                actor_opt.zero_grad(); scaler.scale(a_loss).backward()
                scaler.unscale_(actor_opt); nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                scaler.step(actor_opt)

                with amp.autocast('cuda'):
                    alpha_loss = -(log_alpha * (lp + (-action_dim)).detach()).mean()
                alpha_opt.zero_grad(); scaler.scale(alpha_loss).backward(); scaler.step(alpha_opt)
                
                scaler.update()
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # 核心：每 50 个 Episode 执行一次保存和绘图
        if ep % 50 == 0:
            avg_rew = np.mean(metrics['rewards'][-50:])
            avg_suc = np.mean(metrics['success'][-50:])
            print(f"Ep: {ep:5d} | Rew: {avg_rew:.1f} | Suc: {avg_suc:.2f} | Alpha: {log_alpha.exp().item():.3f} | Steps: {total_steps}")
            
            # --- 绘图 ---
            save_curve(metrics['rewards'], metrics['success'], PLOT_FILE)
            
            # --- 保存模型 ---
            # 1. 定期保存一个最新模型
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_latest.pth"))
            
            # 2. 如果表现最好，保存一个最佳模型
            if avg_rew > best_avg_reward:
                best_avg_reward = avg_rew
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_best.pth"))
                print(f"🌟 发现更好的模型，已保存至 actor_best.pth")

            torch.cuda.empty_cache()

if __name__ == "__main__":
    train()