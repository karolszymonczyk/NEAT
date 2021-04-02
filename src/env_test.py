import gym
env = gym.make("Assault-ram-v0")

observation = env.reset()

print(f"Observation: {observation.size}")
print(f"Observation: {observation}")
print(f"Action space: {env.action_space}")
print("---")

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample())

    env.render()
