import math
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt


def start(fileName):
    instances = open(fileName)

    qtd_cities = int(instances.readline())
    cities_coords = {}
    graph = np.zeros([qtd_cities, qtd_cities], dtype=float)

    for x in range(qtd_cities):
        linha = instances.readline().split()
        a = linha[0]
        b = linha[1]
        cities_coords[x] = a, b

    for x in range(qtd_cities):
        px1 = int(cities_coords[x][0])
        py1 = int(cities_coords[x][1])

        for y in range(qtd_cities):
            px2 = int(cities_coords[y][0])
            py2 = int(cities_coords[y][1])

            if x == y:
                graph[x][y] = 0
            else:
                aux = pow((px2 - px1), 2) + pow((py2 - py1), 2)
                graph[x][y] = math.sqrt(aux)

    instances.close()
    return graph

def calculatepopSize(fileName):
    instances = open(fileName)

    qtd_cities = int(instances.readline())
    instances.close()

    return qtd_cities


def generate_first_population(graph, popSize):
    cities = []
    pop = []

    for x in range(len(graph)):
        cities.append(x)

    for x in range(len(graph)):
        cities.remove(x)
        sample = rd.sample(cities, k=len(graph)-1)
        sample.insert(0, x)
        pop.append(sample)
        cities.append(x)

    return pop

def generate_population(popSize, graph, pheromoneMap):
    pop = []
    route = []
    for ant in range(len(graph)):
        first_position = True
        for iterator in range(len(graph)):
            if (first_position == True):
                route.clear()
                ant_next_position = pick_next_city(ant, route, graph, pheromoneMap)
                route.clear()
                route.append(ant_next_position)
                first_position = False
            else:
                ant_next_position = pick_next_city(ant_next_position, route, graph, pheromoneMap)
                for aux in range(0, popSize-iterator):
                    del route[-1]
                route.append(ant_next_position)
        copy = route.copy()
        pop.append(copy)

    return pop

def fitness(pop, graph):
    performance = []

    for x in range(len(pop)):
        performance.append(1 / route_time(pop[x], graph))

    return performance

def pick_best_ant(pop, popFitness):
    best_result = 0
    best_ant = 0

    for ant in range(0, len(pop)):
        if popFitness[ant] > best_result:
            best_ant = ant
            best_result = popFitness[ant]

    return best_ant

def route_time(route, graph):
    time = 0.0

    for x in range(len(route) - 1):
        if x < len(route):
            time += graph[route[x]][route[x + 1]]

    time += graph[route[0]][route[len(route) - 1]]

    return time

def travelTime(node_a, node_b, graph):

    if graph[node_a][node_b] == 0:
        return 0

    else:
        traveltime = 1/graph[node_a][node_b]

        return traveltime

def probabilisticFunction(node_a, node_b, alpha, beta, pheromoneMap, graph):

    first_term = pow(pheromoneMap[node_a][node_b], alpha) + rd.randint(1, 5)
    second_term = pow(travelTime(node_a, node_b, graph), beta) + rd.randint(1, 5)

    probability = first_term * second_term

    return probability



def selection(pop, popFitness):
    # PARTE 1 - ORDENAR INDIVIDUOS PELO COEFICIENTE DE APTIDAO (1/DISTANCIA) DO MAIOR PARA O MENOR
    rankPop = rank_routes(pop, popFitness)

    # PARTE 2 - SELECIONAR O INDICE DOS INDIVIDUOS POR ELITISMO + ROLETA
    selectedIndex = selectionIndex(rankPop)

    # PARTE 3 - BUSCAR OS CROMOSSOMOS DOS INDICES SELECIONADOS
    selectedCromossomos = selectionCromo(pop, selectedIndex)  # busca os cromossomos dos indices selecionados

    return selectedCromossomos

#pick best ant
def rank_routes(pop, popFitness):
    bettersFitness = {}

    for x in range(len(pop)):
        bettersFitness[x] = popFitness[x]

    return sorted(bettersFitness.items(), key=lambda item: item[1], reverse=True)

#picking next city
def pick_next_city(present_node, visited_nodes, graph, pheromoneMap):
    best_result = 0
    best_next_city = 0

    for next_city in range(0, len(graph)):
        if next_city in visited_nodes:
            a = 0
        else:
            visited_nodes.append(next_city)
            probabilistic_result = probabilisticFunction(present_node, next_city, 1, 50000, pheromoneMap, graph)

            if probabilistic_result > best_result:
                best_result = probabilistic_result
                best_next_city = next_city

    return best_next_city

def update_pheromoneMap(pheromoneMap, bestAnt, bestAntValue, pheromoneEvaporationTax, Q, bestPath, bestPathValue):
    for cities in range(0, len(bestAnt)-1):
        x = bestAnt[cities]
        y = bestAnt[cities+1]
        if pheromoneMap[x][y] == 0:
            pheromoneMap[x][y] = (1 - pheromoneEvaporationTax)
        else:
            if x and y in bestPath:
                if x and y in bestAnt:
                    pheromoneMap[x][y] = (1 - pheromoneEvaporationTax) * float(pheromoneMap[x][y]) + Q/bestAntValue + 0.001 * Q/bestPathValue
                else:
                    pheromoneMap[x][y] = (1 - pheromoneEvaporationTax) * float(pheromoneMap[x][y]) + 0 + 0.001 * Q/bestAntValue
            else:
                if x and y in bestAnt:
                    pheromoneMap[x][y] = (1 - pheromoneEvaporationTax) * float(pheromoneMap[x][y]) + Q/bestAntValue + 0
                else:
                    pheromoneMap[x][y] = (1 - pheromoneEvaporationTax) * float(pheromoneMap[x][y]) + 0 + 0

    return pheromoneMap

def geneticAlgorithm(graphCities):
    popSize = calculatepopSize("instances.txt")
    progress = []
    bestPath = []
    bestPathValue = 0
    pheromoneMap = np.zeros([popSize, popSize], dtype=float)

    pop = generate_first_population(graphCities, popSize)
    popFitness = fitness(pop, graphCities)
    bestAnt = pick_best_ant(pop, popFitness)
    pheromoneMap = update_pheromoneMap(pheromoneMap, pop[bestAnt], popFitness[bestAnt], 0.00001, 100, bestPath, bestPathValue)
    aux = rank_routes(pop, popFitness)
    progress.append(1 / aux[0][1])
    bestPath = pop[bestAnt]
    bestPathValue = popFitness[bestAnt]
    numGenerations = 10000  # numero de gerações

    print("Melhor distancia inicial: " + str(1 / aux[0][1]))
    print("Melhor rota inicial: " + str(pop[aux[0][0]]))

    for i in range(0, numGenerations-1):
        new_generation = generate_population(popSize, graphCities, pheromoneMap)
        popFitness = fitness(new_generation, graphCities)
        bestAnt = pick_best_ant(new_generation, popFitness)
        pheromoneMap = update_pheromoneMap(pheromoneMap, pop[bestAnt], popFitness[bestAnt], 0.00001, 100, bestPath, bestPathValue)
        aux = rank_routes(new_generation, popFitness)
        progress.append(1 / aux[0][1])
        if popFitness[bestAnt] > bestPathValue:
            bestPathValue = popFitness[bestAnt]
            bestPath = pop[bestAnt]

    aux = rank_routes(new_generation, popFitness)
    print("Melhor distancia final: " + str(1 / aux[0][1]))
    bestRoute = new_generation[aux[0][0]]
    print("Melhor rota final: " + str(bestRoute))

    plt.plot(progress)
    x = range(len(progress))
    plt.title("GA applied to TSP", loc='center')
    plt.text(x[len(x) // 2], progress[0], 'minimum distance: {}'.format(progress[-1]), ha='center', va='center')
    plt.ylabel('Distance')
    plt.xlabel('Generations')
    plt.show()

# PARAMETROS
#   tamanho da população: 10*num_cidades
#   taxa do elitismo: 10% da população
#   taxa de mutação: 1%
#   numero de geracoes: 100 gerações