import gymnasium as gym
import torch
import matplotlib.pyplot as plt


from agent import DQNAgent
from replay_buffer import ReplayBuffer
from config import *

def main():
    rewards=[]
    losses=[]


    env = gym.make("CartPole-v1")

    device= torch.device("cuda")

    state_dim =env.observation_space.shape[0]
    action_dim=env.action_space.n

    agent = DQNAgent(state_dim,action_dim,device)
    buffer =ReplayBuffer(10000)

    epsilon =EPSILON_START



    # 画图
    plt.ion()
    fig,ax =plt.subplots(1,2,figsize=(12,5))




    for episode in range(EPISODES):
        state,_=env.reset()
        done = False
        total_reward=0
        loss_list=[]
        
        while not done:
            action =agent.select_action(state,epsilon)

            next_state,reward,terminated,truncated,_=env.step(action)

            done=terminated or truncated

            buffer.push(state,action,reward,next_state,done)

            if len(buffer) > MIN_BUFFER_SIZE:
                loss = agent.train(buffer, BATCH_SIZE)
                if loss is not None:
                    loss_list.append(loss)
            else:
                loss = None

            state = next_state
            total_reward +=reward

        avg_loss =sum(loss_list)/len(loss_list) if loss_list else None
        rewards.append(total_reward)
        if avg_loss is not None:
            losses.append(avg_loss)



        epsilon = max(EPSILON_MIN,epsilon*EPSILON_DECAY)

        if episode % TARGET_UPDATE==0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        

        # 🔥 更新图像（关键）
        ax[0].clear()
        ax[1].clear()

        ax[0].set_title("Reward")
        ax[0].plot(rewards)

        ax[1].set_title("Loss")
        ax[1].plot(losses)

        plt.tight_layout()
        plt.pause(0.01)

        print(f"Episode {episode+1} | Reward: {total_reward} | Epsilon: {epsilon:.3f} | Loss: {avg_loss}")

    env.close()

    # 🔥 关闭交互模式
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()