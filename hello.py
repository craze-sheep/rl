import gymnasium as gym

# 下面每一行前面都不能有多余空格
env = gym.make('CartPole-v1')

observation = env.reset()

print("初始观察:", observation)

print("动作空间：",env.action_space)

print("观察空间：",env.observation_space)