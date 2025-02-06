import math
import random

import matplotlib.pyplot as plt

NUM_NODES=20
NUM_CHROMOSOME=10
SINK_NODE= [49, 49]
MAX_CH= 5

# Euclidean Distance
def euclidean_distance(node_array, ch_array):
    return math.sqrt((node_array[0] - ch_array[0])**2 + (node_array[1] - ch_array[1])**2)

# Nodes Coordinates
nodes=[[round(random.uniform(0,100),2), round(random.uniform(0,100),2)] for _ in range(NUM_NODES)]
print(f"Nodes:\n{nodes}")

# Remaining node energy
energy_remaining = [50 for _ in range(NUM_NODES)]

# Binary Tournament Selection
def binary_tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        i1, i2 = random.sample(range(len(population)), 2)
        if fitnesses[i1] > fitnesses[i2]:
            selected.append(population[i1])
        else:
            selected.append(population[i2])
    return selected

# Crossover Function
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    offspring1 = adjust_cluster_heads(offspring1,MAX_CH)
    offspring2 = adjust_cluster_heads(offspring2,MAX_CH)
    return offspring1, offspring2

def adjust_cluster_heads(chromosome, max_cluster_heads):
    num_cluster_heads = sum(chromosome)
    
    if num_cluster_heads > max_cluster_heads:
        # Too many cluster heads, need to adjust
        excess = num_cluster_heads - max_cluster_heads
        
        # Find indices of excess cluster heads
        indices = [i for i, x in enumerate(chromosome) if x == 1]
        
        # Randomly convert excess cluster heads to non-cluster heads (0)
        for _ in range(excess):
            idx_to_convert = random.choice(indices)
            chromosome[idx_to_convert] = 0
            indices.remove(idx_to_convert)  # Remove index to avoid converting again
        
    return chromosome

# Mutation Function
def mutate(chromosome, mutation_rate=0.01):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 if chromosome[i] == 0 else 0
    return chromosome

# Genetic algorithm function
def genetic_algo(nodes, energy_remaining, max_cluster_heads, generations=120, elite_size=int(NUM_NODES/6), mutation_rate=0.01 ):
    cluster_heads = []
    for _ in range(NUM_CHROMOSOME):
        chromosome = [0] * NUM_NODES
        pop = random.sample(range(NUM_NODES), max_cluster_heads)
        for ch in pop:
            chromosome[ch] = 1
        cluster_heads.append(chromosome)
    best_fitness_arr=[]
    
    # cluster_heads=[[random.choice([0, 1]) for _ in range(NUM_NODES)] for _ in range(NUM_CHROMOSOME)]
    for generation in range(generations):
        # print(f"\nCluster Heads Chromosome form:\n{cluster_heads}")
        print(f"Cluster Heads: {cluster_heads}")
        residual_energies=[]
        distances=[]
        minimum_distances = []

        for ch in cluster_heads:
            sum_energy = sum(energy_remaining[i] for i, x in enumerate(ch) if x)
            residual_energies.append(sum_energy)
            
            distance_ch_bs = [round(euclidean_distance(nodes[i], SINK_NODE),2) for i, x in enumerate(ch) if x]
            distances.append(distance_ch_bs)
            sum_distances = [round(sum(distances[i]),2) for i in range(len(distances))]
                        
            distance_ch_ch = []
            for i, x in enumerate(ch):
                if x:
                    ch_one = nodes[i]
                    for j in range(i + 1, NUM_NODES):
                        if ch[j]:
                            ch_two = nodes[j]
                            distance_ch_ch.append(round(euclidean_distance(ch_one, ch_two),2))
            minimum_distances.append(min(distance_ch_ch))


        print(f"Residual Energy: {residual_energies} \nDistance between ch and bs: {sum_distances} \nMinimum distances between cluster head:{minimum_distances}")
        
        # Fitness values
        fitnesses = [round(((residual_energies[i]) - (sum_distances[i]) + (1/minimum_distances[i]))/10, 2) for i in range(len(cluster_heads))]
        # print(f"\nFitnesses: {fitnesses}")
        print(f"Fitnesses: {fitnesses}")
        # Elitism
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [cluster_heads[i] for i in elite_indices]
        # print(f"Elites: {elites}")
        
        # Selection
        selected_population = binary_tournament_selection(cluster_heads, fitnesses)
        # print(f"Selected Population: {selected_population}")
        
        # Crossover
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % len(selected_population)]
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            next_generation.extend([offspring1, offspring2])
        
        # Mutation
        next_generation = [mutate(individual, mutation_rate) for individual in next_generation]
        # print(f"Next Genertion: {next_generation}")
            
        # Combine elites with next generation
        cluster_heads = elites + next_generation[:len(cluster_heads) - elite_size]
        
        # print(f"Population: {population}")
        # Optionally, you can print or log the best fitness of the current generation
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}\n")
        print(f"Best fitness arr:{best_fitness_arr}")
        best_fitness_arr.append(best_fitness)
        
        def plot_nodes():
            arr= [i for i in range (generations)]
            # Array xch and ych containing coordinates only x and y for cluster heads
            
            plt.figure(figsize=(10, 10))
            
            # Average Remaining Energy
            plt.figure(figsize=(10,10))
            # plt.xticks(range(generations))
            plt.yticks(range(100))
            plt.grid(True)
            plt.title("LEACH Average Remaining Energy")
            plt.xlabel("Round")
            plt.ylabel("Average energy")
            plt.plot(arr,best_fitness_arr,'r')
            # plt.ylim(min(avg_energy_of_rem_nodes) - 0.01, max(avg_energy_of_rem_nodes) + 0.01)
            # plt.xlim(0, MAX_ROUNDS)
            plt.savefig("best_fitness.png")
            plt.close()
        if len(best_fitness_arr)==generations:
            plot_nodes()
    # Return the final population
    return cluster_heads
        
        
# Call genetic algo function
final_population= genetic_algo(nodes, energy_remaining, MAX_CH)
print(f"\nFinal Population: {final_population}")
print("Cluster Heads:")
for i,x in enumerate(final_population[0]):
    if x:
        print(nodes[i],  end=" ")
    
print("\nNodes:")
for i,x in enumerate(final_population[0]):
    if not x:
        print(nodes[i], end=" ")