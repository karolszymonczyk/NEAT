import os
import sys
import pickle
import neat
import gym
import numpy as np


def load_model(model):
    with open(model, "rb") as f:
        c = pickle.load(f)

    return c


def run():

    model = "checkpoint" if check else "winner"
    c = load_model(model)

    print("Loaded genome:")
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)

    env = gym.make("Assault-ram-v0")
    observation = env.reset()
    fitness = 0

    done = False
    while not done:
        action = np.argmax(net.activate(observation))

        observation, reward, done, _ = env.step(action)
        fitness += reward
        env.render()

    env.close()

    print(f"SCORE: {fitness}")


if __name__ == '__main__':
    check = '-check' in sys.argv
    run()
