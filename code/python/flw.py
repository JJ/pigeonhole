import copy

import numpy as np
import random
import operator
import math
from deap import tools
from deap.benchmarks import rastrigin


class Agent(object):
    def __init__(self, **kwargs):
        self.value = kwargs.get('value', None)
        self.position = kwargs.get('position', [])
        self.leader = kwargs.get('leader', None)
        self.min = kwargs.get('min', -1)
        self.max = kwargs.get('max', 1)
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.value) + ":" + str(self.position)

    def __repr__(self):
        return str(self.value) + ":" + str(self.position)

    def as_dict(self):
        return self.__dict__


def init_location(size, pmin, pmax, smin=None, smax=None):
    # random uniform locations
    agent = Agent(position=list(random.uniform(pmin, pmax)
                                for _ in range(size)))
    # Other locations
    # Agents don't have speed
    # agent.speed = [random.uniform(smin, smax) for _ in range(size)]
    # agent.smin = smin
    # agent.smax = smax
    return agent


def create_agents(config):
    pop = []  # New population
    leaders = {}  # Leaders
    n_walkers = int(config['pool_size'] * config['walker_rate'])
    n_followers = (config['pool_size'] - n_walkers -
                   config['n_leaders'])//config['n_leaders']

    # create leader agents and their followers
    for id in range(config['n_leaders']):
        new_leader = init_location(config['dimension'], config['a'],
                                   config['b'])
        new_leader.leader = id
        leaders[id] = {'actor': new_leader, 'followers': []}
        pop.append(new_leader)
        # create an equal number of followers/leader (NB. round-off may yield
        # fewer agents than pop-size)
        # for small pool sizes n_followers can be zero !
        for _ in range(n_followers):
            new_follower = init_location(config['dimension'], config['a'],
                                         config['b'])
            new_follower.leader = id
            leaders[id]['followers'].append(new_follower)
            pop.append(new_follower)
    # create walker agents
    for _ in range(n_followers):
        new_walker = init_location(config['dimension'], config['a'],
                                   config['b'])
        new_walker.leader = None
        pop.append(new_walker)

    return pop, leaders


def update(agent, leaders, phi1=2, phi2=1, phi3=20):
    if agent.leader and agent == leaders[agent.leader]['actor']:  # leader?
        u = (random.uniform(0, phi2) for _ in range(len(agent.position)))
        agent.position[:] = list(map(operator.add, agent.position, u))
    elif agent.leader:  # follower?
        e = (random.uniform(0, phi1) for _ in range(len(agent.position)))
        v = (random.uniform(0, phi2) for _ in range(len(agent.position)))
        # e (xl - xi)
        v_e = map(operator.mul, e, map(operator.sub,
                                       leaders[agent.leader]['actor'].position,
                                       agent.position))
        # xi + e(xl - xi) + v
        agent.speed = list(map(operator.add, agent.position,
                           map(operator.add, v_e, v)))

        for i, speed in enumerate(agent.speed):
            if abs(speed) < agent.min:
                agent.speed[i] = math.copysign(agent.min, speed)
            elif abs(speed) > agent.max:
                agent.speed[i] = math.copysign(agent.max, speed)
        agent.position[:] = list(agent.speed)
    else:  # walker
        w = (random.uniform(0, phi3) for _ in range(len(agent.position)))
        agent.position[:] = list(map(operator.add, agent.position, w))


evaluate = rastrigin  # This can be an import or a dictionary like in NetLogo


def main(config):
    stats = tools.Statistics(lambda ind: ind.value)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["steps", "evals"] + stats.fields

    pool, leaders = create_agents(config)
    best_leader = None

    for g in range(config['n_gens']):
        for agent in pool:
            agent.value, = evaluate(agent.position)
            # update best_leader
            if not best_leader or best_leader.value > agent.value:
                best_leader = copy.copy(agent)

        for agent in pool:
            update(agent, leaders)

        # replace leader with best follower if any
        for id in leaders:
            best_follower = min(leaders[id]['followers'],
                                key=lambda a: a.value)
            if leaders[id]['actor'].value > best_follower.value:
                old_leader = leaders[id]['actor']

                leaders[id]['actor'] = best_follower
                leaders[id]['followers'].remove(best_follower)
                leaders[id]['followers'].append(old_leader)

        print(g, best_leader)
        # Gather all the fitnesses in one list and print the stats
        # logbook.record(gen=g, evals=len(pool), **stats.compile(pool))
        # print(logbook.stream)
        # config['Tiempo_Total'] = time.time() - inicio_tiempo

    print(logbook.chapters)

    return config


if __name__ == "__main__":
    # config = None
    # with open(r"..\config.json", "r") as conf_file:
    #     config = json.load(conf_file)
    config = {'list_size': 2,
              'pool_size': 100,
              'walker_rate': 0.2,
              'dimension': 2,
              'a': -5.0,
              'b': 5.0,
              'n_leaders': 4,
              'n_gens': 100
              }
    results = main(config)
    # print(results)
