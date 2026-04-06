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
import matplotlib.pyplot as plt

# ==========================================
# 1. 4060 优化版超参数 (严格保持不变)
# ==========================================
ENV_NAME = "PandaPickAndPlace-v3" 
GAMMA = 0.99
TAU = 0.005             
LR = 2e-4              
BATCH_SIZE = 128       
BUFFER_SIZE = 500000   
HER_K = 5              
MAP_SIZE = 16          
ALPHA_INIT = 0.2
WARMUP_STEPS = 10000 
MAX_STEPS = 50
MAX_EPISODES = 40000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "GeneWorker_Results"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_FILE = os.path.join(SAVE_DIR, "train_curve.png")

# ==========================================
# 2. GeneWorker 模型 (保持不变)
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
            {'in': input_dim, 'out': 128}, {'in': 128, 'out': 64},
            {'in': 64, 'out': 64}, {'in': 64, 'out': action_dim * 2}
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
# 3. 绘图 (仅 Success Rate)
# ==========================================
def save_curve(success_rates, path):
    if not success_rates: return
    plt.figure(figsize=(8, 5))
    plt.plot(success_rates, color='green', alpha=0.2)
    if len(success_rates) >= 100:
        ma = np.convolve(success_rates, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(success_rates)), ma, color='green', linewidth=2)
    plt.title('Training Success Rate')
    plt.ylabel('Success Rate')
    plt.xlabel('Episode')
    plt.ylim([-0.05, 1.05])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(path)
    plt.close()

# ==========================================
# 4. 训练主逻辑
# ==========================================
def train():
    # 针对 8G 显存优化的配置
    torch.backends.cudnn.benchmark = True
    # 修正 API 警告
    scaler = torch.amp.GradScaler('cuda') 

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
    
    # Buffer 存在内存(RAM)中，不占显存
    buffer = deque(maxlen=BUFFER_SIZE // MAX_STEPS) 
    total_steps = 0
    metrics = {'rewards': [], 'success': []}
    best_avg_reward = -float('inf')

    print(f"🚀 8G 显存优化版启动 | 任务: {ENV_NAME} | 设备: {DEVICE}")

    for ep in range(MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_data = []
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # 推理阶段使用 no_grad 节省显存
            with torch.no_grad():
                s_t = torch.as_tensor(obs_dict['observation'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                g_t = torch.as_tensor(obs_dict['desired_goal'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action, _ = actor.sample_action(s_t, g_t)
                    action = action.cpu().numpy()[0]
            
            next_obs_dict, reward, term, trunc, info = env.step(action)
            episode_data.append({
                'obs': obs_dict['observation'].astype(np.float32), 
                'act': action.astype(np.float32), 
                'rew': float(reward),
                'next_obs': next_obs_dict['observation'].astype(np.float32), 
                'next_ag': next_obs_dict['achieved_goal'].astype(np.float32),
                'dg': obs_dict['desired_goal'].astype(np.float32), 
                'done': float(term or trunc)
            })
            obs_dict = next_obs_dict
            episode_reward += reward
            total_steps += 1
            if term or trunc: break
        
        buffer.append(episode_data)
        metrics['rewards'].append(episode_reward)
        metrics['success'].append(float(info.get('is_success', 0.0)))

        # 训练更新阶段
        if total_steps >= WARMUP_STEPS:
            for i in range(20): 
                # 1. 批量 HER 采样 (在 CPU 完成)
                samples = []
                for _ in range(BATCH_SIZE):
                    sampled_ep = random.choice(buffer)
                    t = random.randint(0, len(sampled_ep)-1)
                    trans = sampled_ep[t]
                    if random.random() < HER_K/(HER_K+1):
                        new_g = sampled_ep[random.randint(t, len(sampled_ep)-1)]['next_ag']
                        rew = env.unwrapped.compute_reward(trans['next_ag'], new_g, {})
                        samples.append((trans['obs'], trans['act'], rew, trans['next_obs'], new_g, trans['done']))
                    else:
                        samples.append((trans['obs'], trans['act'], trans['rew'], trans['next_obs'], trans['dg'], trans['done']))
                
                # 2. 一次性转为 Tensor (显存优化关键)
                s = torch.as_tensor(np.array([x[0] for x in samples]), device=DEVICE)
                a = torch.as_tensor(np.array([x[1] for x in samples]), device=DEVICE)
                r = torch.as_tensor(np.array([x[2] for x in samples]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
                ns = torch.as_tensor(np.array([x[3] for x in samples]), device=DEVICE)
                g = torch.as_tensor(np.array([x[4] for x in samples]), device=DEVICE)
                d = torch.as_tensor(np.array([x[5] for x in samples]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
                
                alpha = log_alpha.exp()

                # 3. 混合精度更新
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        next_a, next_lp = actor.sample_action(ns, g)
                        t_q1, t_q2 = critic_target(ns, g, next_a)
                        target_q = r + (1-d) * GAMMA * (torch.min(t_q1, t_q2) - alpha * next_lp)
                    
                    c_q1, c_q2 = critic(s, g, a)
                    c_loss = F.mse_loss(c_q1, target_q) + F.mse_loss(c_q2, target_q)
                
                critic_opt.zero_grad(set_to_none=True)
                scaler.scale(c_loss).backward()
                scaler.unscale_(critic_opt)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler.step(critic_opt)

                with torch.amp.autocast('cuda'):
                    new_a, lp = actor.sample_action(s, g)
                    q1, q2 = critic(s, g, new_a)
                    a_loss = (alpha * lp - torch.min(q1, q2)).mean()
                
                actor_opt.zero_grad(set_to_none=True)
                scaler.scale(a_loss).backward()
                scaler.unscale_(actor_opt)
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                scaler.step(actor_opt)

                with torch.amp.autocast('cuda'):
                    alpha_loss = -(log_alpha * (lp + (-action_dim)).detach()).mean()
                alpha_opt.zero_grad(set_to_none=True)
                scaler.scale(alpha_loss).backward()
                scaler.step(alpha_opt)
                
                scaler.update()

                # 4. 软更新
                with torch.no_grad():
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
                
                # 5. 显式清理这一步产生的临时变量
                del s, a, r, ns, g, d, target_q, c_loss, a_loss, alpha_loss, samples

            # 每训练完一轮(20次迭代)尝试回收显存碎片
            if ep % 5 == 0:
                torch.cuda.empty_cache()

        # 定期保存与绘图
        if ep % 50 == 0:
            avg_rew = np.mean(metrics['rewards'][-50:])
            avg_suc = np.mean(metrics['success'][-50:])
            print(f"Ep: {ep:5d} | Rew: {avg_rew:.1f} | Suc: {avg_suc:.2f} | Alpha: {log_alpha.exp().item():.3f}")
            save_curve(metrics['success'], PLOT_FILE)
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_latest.pth"))
            if avg_rew > best_avg_reward:
                best_avg_reward = avg_rew
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_best.pth"))

if __name__ == "__main__":
    train()