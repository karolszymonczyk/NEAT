import multiprocessing
import os
import sys
import pickle
import numpy as np

import neat

import visualize
import gym


curr_best_fitness = 0


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make("Assault-ram-v0")
        observation = env.reset()

        fitness = 0.0
        done = False
        while not done:

            # predict action
            action = np.argmax(net.activate(observation))
            observation, reward, done, _ = env.step(action)

            if(live):
                env.render()

            fitness += reward

        env.close()

        fitnesses.append(fitness)

    net_fitness = np.max(fitnesses)

    global curr_best_fitness
    if check and net_fitness > curr_best_fitness:
        save_model(genome, 'checkpoint')
        curr_best_fitness = net_fitness

    # The genome's fitness is its max performance across all runs.
    return net_fitness


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    save_model(winner, 'winner')

    # print_stats(winner, stats, config)


def print_stats(winner, stats, config):
    print(f"Winner: {winner}")

    visualize.plot_stats(stats, ylog=True,   view=True,
                         filename="vis/feedforward-fitness.svg")
    visualize.plot_species(
        stats, view=True, filename="vis/feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="vis/winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="vis/winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="vis/winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    live = '-live' in sys.argv
    check = '-check' in sys.argv
    runs_per_net = 2

    run()
