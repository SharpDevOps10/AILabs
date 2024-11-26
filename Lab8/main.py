import random
import numpy as np
from deap import base, creator, tools

NUM_HOSTS = 11  # Кількість вузлів
NUM_HUBS = 3  # Кількість концентраторів
POP_SIZE = 300  # Розмір популяції
GENERATIONS = 100  # Кількість поколінь
ELITE_SIZE = 20  # Розмір елітної групи
TOURNAMENT_SIZE = 2  # Розмір турніру
MAX_NODES_PER_HUB = 5  # Максимальна кількість вузлів на концентратор
MAX_CHANNEL_CAPACITY = 300  # Максимальна пропускна здатність каналу

traffic_matrix = np.random.randint(1, 100, size=(NUM_HOSTS, NUM_HOSTS))
np.fill_diagonal(traffic_matrix, 0)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", lambda: random.randint(1, NUM_HUBS))
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: generate_valid_individual())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def check_ring_topology(individual):
    hubs = set(individual)
    if len(hubs) != NUM_HUBS:
        return False

    for i in range(NUM_HUBS):
        current_hub = i + 1
        next_hub = (i + 1) % NUM_HUBS + 1
        if current_hub not in individual or next_hub not in individual:
            return False
    return True


def generate_valid_individual():
    while True:
        individual = [random.randint(1, NUM_HUBS) for _ in range(NUM_HOSTS)]
        if check_ring_topology(individual):
            return individual


def evaluate(individual):
    total_traffic = 0
    hub_loads = [0] * NUM_HUBS
    channel_traffic = [0] * NUM_HUBS
    penalties = 0

    if not check_ring_topology(individual):
        penalties += 1000

    for i in range(NUM_HOSTS):
        hub_loads[individual[i] - 1] += 1

        for j in range(i + 1, NUM_HOSTS):
            if individual[i] != individual[j]:
                traffic = traffic_matrix[i][j]
                total_traffic += traffic
                channel_traffic[individual[i] - 1] += traffic
                channel_traffic[individual[j] - 1] += traffic

    penalties += sum(max(0, load - MAX_NODES_PER_HUB) for load in hub_loads)

    if total_traffic > MAX_CHANNEL_CAPACITY:
        penalties += total_traffic - MAX_CHANNEL_CAPACITY

    return total_traffic + penalties,


toolbox.register("evaluate", evaluate)

toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Однорідний кросинговер
toolbox.register("mutate", tools.mutUniformInt, low=1, up=NUM_HUBS, indpb=0.2)  # Мутація


def main():
    random.seed(42)
    population = toolbox.population(n=POP_SIZE)  # Ініціалізація популяції

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)
    best_generation = 0
    min_traffic = float("inf")

    for gen in range(GENERATIONS):
        population.sort(key=lambda ind: ind.fitness.values[0])

        next_generation = population[:ELITE_SIZE]

        while len(next_generation) < POP_SIZE:
            parents = toolbox.select(population, len(population) // 2)
            offspring1, offspring2 = toolbox.mate(parents[0], parents[1])
            del offspring1.fitness.values, offspring2.fitness.values
            next_generation.extend([offspring1, offspring2])

        for ind in next_generation[ELITE_SIZE:]:
            if random.random() < 0.5:
                toolbox.mutate(ind)
                del ind.fitness.values

        population[:] = next_generation

        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        hof.update(population)

        best = tools.selBest(population, k=1)[0]
        print(f"Generation {gen + 1}: Best Fitness = {best.fitness.values[0]}")

        if best.fitness.values[0] < min_traffic:
            min_traffic = best.fitness.values[0]
            best_generation = gen + 1

    best_individual = hof[0]
    print("\nBest Solution from Hall of Fame: ", best_individual)
    print("Minimum Traffic: ", evaluate(best_individual)[0])
    print(f"Minimum Traffic found in Generation: {best_generation}")


if __name__ == "__main__":
    main()
