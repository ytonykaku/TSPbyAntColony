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

def popSize(fileName):
    instances = open(fileName)

    qtd_cities = int(instances.readline())
    instances.close()

    return qtd_cities


def generate_first_population(graph, popSize):
    cities = []
    pop = []

    for x in range(len(graph)):
        cities.append(x)

    for x in range(popSize):
        pop.append(rd.sample(cities, k=len(graph)))

    return pop

def generate_population(popSize, graph, pheromoneMap):
    ant_route = []
    for ant in popSize:
        first_position = True
        while (len(ant_rout[ant]) > popSize):
            if (first_position == True):
                ant_next_position = pick_next_city(ant, ant_route, graph, pheromoneMap)
                ant_route[ant].append(ant_next_position)
                first_position = False
            else:
                ant_next_position = pick_next_city(ant_next_position, ant_route, graph, pheromoneMap)
                ant_route[ant].append(ant_next_position)

        for city in popSize:
            ant_route[ant].append(pick_next_city(city, ant_route[ant], graphCities, pheromoneMap))

    return ant_route

def fitness(pop, graph):
    performance = []

    for x in range(len(pop)):
        performance.append(1 / route_time(pop[x], graph))

    return performance

def travelTime(node_a, node_b, graph):

    traveltime = 1/graph[node_a][node_b]

    return traveltime

def probabilisticFunction(node_a, node_b, alpha, beta, pheromoneMap):

    probability  = ((float(pheromoneMap[node_a][node_b]^alpha)) * (travelTime(node_a, node_b)^beta))

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
    for next_city in range(0, len(graph)):
        if next_city in visited_nodes:
        else:
            visited_nodes.append(next_city)
            probabilistic_result = probabilisticFunction(present_node, next_city, 1, 5, pheromoneMap)

            if probabilistic_result > best_result:
                best_result = probabilistic_result
                best_next_city = next_city

    return best_next_city

def update_pheromoneMap(pheromoneMap, bestAnt, pheromoneEvaporationTax, Q):
    for cities in range(0, len(bestAnt)-1):
        if pheromoneMap[bestAnt[cities]][bestAnt[cities+1]] == 0:
            pheromoneMap[bestAnt[cities]][bestAnt[cities+1]] = (1 - pheromoneEvaporationTax)
        else:
            [bestAnt[cities]][bestAnt[cities+1]] = (1-pheromoneEvaporationTax) * float([bestAnt[cities]][bestAnt[cities+1]]) + (pheromoneEvaporationTax * float)

    return pheromoneMap

def geneticAlgorithm(graphCities):
    popSize = popSize("instances.txt")
    progress = []
    ant_route = []
    pheromoneMap = np.zeros([popSize, popSize], dtype=float)

    for x in range (0, popSize):
        for y in range (0, popSize):
            pheromoneMap[x][y] = 0

    popFitness = fitness(pop, graphCities)
    popPheromones = pheromones(pop, graphCities)
    aux = rank_routes(pop, popFitness, popPheromones)
    progress.append(1 / aux[0][1])
    numGenerations = 800  # numero de gerações

    print("Melhor distancia inicial: " + str(1 / aux[0][1]))
    print("Melhor rota inicial: " + str(pop[aux[0][0]]))

    for i in range(0, numGenerations):
        if i == 0:
            pop = generate_first_population(graphCities, popSize)
            popFitness = fitness(pop, graphCities)
            bestAnt = rank_routes(pop, popFitness)
            pheromoneMap = update_pheromoneMap(pheromoneMap, bestAnt, 0.1, 100)
            progress.append(1 / rank_routes(pop, popFitness)[0][1])

        else:
            new_generation = generate_population(popSize, graphCities, pheromoneMap)
            popFitness = fitness(new_generation, graphCities)
            bestAnt = rank_routes(new_generation, popFitness)
            pheromoneMap = update_pheromoneMap(pheromoneMap, bestAnt, 0.00001, 100)
            progress.append(1 / rank_routes(pop, popFitness)[0][1])


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