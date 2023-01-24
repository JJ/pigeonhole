import copy

import numpy as np
import random
import operator
import math
from deap import tools
from deap.benchmarks import rastrigin

step_sigmas = [10**(1-i) for i in range(6)]


class Agent(object):
    def __init__(self, **kwargs):
        self.value = kwargs.get('value', None)
        self.position = kwargs.get('position', [])
        self.leader = kwargs.get('leader', None)
        self.min = kwargs.get('min', -5)
        self.max = kwargs.get('max', 5)
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.value) + ":" + str(self.position) + ":"+ str(self.leader)

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

# def update_min_max(agent):


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
        leaders[id] = {'agent': new_leader, 'followers': []}
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

    if agent.leader is not None and agent == leaders[agent.leader]['agent']:  # leader?
    #    u = (random.uniform(0, phi2) for _ in range(len(agent.position)))
    #   new_position = list(map(operator.add, agent.position, u))
    #   agent.position[:] = new_position
        pass
    elif agent.leader is not None:  # follower?
        e = list(random.uniform(0, phi1) for _ in range(len(agent.position)))
        v = list(random.uniform(0, phi2) for _ in range(len(agent.position)))
        # e (xl - xi)
        v_e = list(map(operator.mul, e, map(operator.sub,
                                       leaders[agent.leader]['agent'].position,
                                       agent.position)))
        new_position = list(map(operator.add, agent.position,
                                v_e))
        # xi + e(xl - xi) + v
        # new_position = list(map(operator.add, agent.position,
        #                   map(operator.add, v_e, v)))

        agent.position[:] = new_position

    elif agent.leader is None:  # walker
        w = list(random.uniform(0, random.choice(step_sigmas)) for _ in range(len(agent.position)))
        new_position = list(map(operator.add, agent.position, w))
        agent.position[:] = new_position

    for i, position in enumerate(agent.position):
        if abs(position) < agent.min:
            agent.position[i] = math.copysign(agent.min, position)
        elif abs(position) > agent.max:
            agent.position[i] = math.copysign(agent.max, position)

def print_pool(g, pool, leaders):
    print("Gen :", g)
    for agent_id in leaders:
        print("leader:", agent_id, leaders[agent_id]['agent'])
        for f in leaders[agent_id]['followers']:
            print(f)
    print("Walkers:")
    for walker in [agent for agent in pool if agent.leader is None]:
        print(walker)
            

    print("#####")


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
    best_solution = None

    for g in range(config['n_gens']):
        for agent in pool:
            agent.value, = evaluate(agent.position)
            # Best Solution
            if not best_solution or best_solution.value > agent.value:
                best_solution = copy.copy(agent)
        # replace leader with best follower if any
        for id in leaders:
            best_follower = min(leaders[id]['followers'],
                                key=lambda a: a.value)
            if leaders[id]['agent'].value > best_follower.value:
                old_leader = leaders[id]['agent']

                leaders[id]['agent'] = best_follower
                leaders[id]['followers'].remove(best_follower)
                leaders[id]['followers'].append(old_leader)

        # update best_leader
        best_leader_id = min(leaders,
                             key=lambda id: leaders[id]['agent'].value)
        best_leader = leaders[best_leader_id]

        walkers = [agent for agent in pool if agent.leader is None]
        best_walker = min(walkers, key=lambda a: a.value)

        if best_walker.value < best_leader['agent'].value:
            best_leader['agent'].position = best_walker.position[:]

        for agent in pool:
            update(agent, leaders)

        print_pool(g, pool, leaders)


        walkers[0].position = best_solution.position[:]

        print(g, best_solution)
        # Gather all the fitnesses in one list and print the stats
        # logbook.record(gen=g, evals=len(pool), **stats.compile(pool))
        # print(logbook.stream)
        # config['Tiempo_Total'] = time.time() - inicio_tiempo

    # print(logbook.chapters)
    print("best solution: ", best_solution)
    return config


if __name__ == "__main__":
    # config = None
    # with open(r"..\config.json", "r") as conf_file:
    #     config = json.load(conf_file)
    config = {'list_size': 2,
              'pool_size': 20,
              'walker_rate': 0.2,
              'dimension': 2,
              'a': -5.0,
              'b': 5.0,
              'n_leaders': 4,
              'n_gens': 10
              }
    results = main(config)
    # print(results)
