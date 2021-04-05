import gym
env = gym.make("Assault-ram-v0")

observation = env.reset()

print(f"Observation: {observation}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print("---")

done = False
while not done:
    random_action = env.action_space.sample()
    print(random_action)

    observation, reward, done, info = env.step(random_action)

    env.render()

env.close()
