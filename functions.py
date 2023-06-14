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


def generate_population(graph):
    cities = []
    sizepop = 4 * len(graph)  # tamanho da população
    #sizepop = 800
    pop = []

    for x in range(len(graph)):
        cities.append(x)

    for x in range(sizepop):
        pop.append(rd.sample(cities, k=len(graph)))

    return pop


def fitness(pop, graph):
    performance = []

    for x in range(len(pop)):
        performance.append(1 / route_time(pop[x], graph))

    return performance


def route_time(route, graph):
    time = 0.0

    for x in range(len(route) - 1):
        if x < len(route):
            time += graph[route[x]][route[x + 1]]

    time += graph[route[0]][route[len(route) - 1]]

    return time


def selection(pop, popFitness):
    # PARTE 1 - ORDENAR INDIVIDUOS PELO COEFICIENTE DE APTIDAO (1/DISTANCIA) DO MAIOR PARA O MENOR
    rankPop = rank_routes(pop, popFitness)

    # PARTE 2 - SELECIONAR O INDICE DOS INDIVIDUOS POR ELITISMO + ROLETA
    selectedIndex = selectionIndex(rankPop)

    # PARTE 3 - BUSCAR OS CROMOSSOMOS DOS INDICES SELECIONADOS
    selectedCromossomos = selectionCromo(pop, selectedIndex)  # busca os cromossomos dos indices selecionados

    return selectedCromossomos


def rank_routes(pop, popFitness):
    bettersFitness = {}

    for x in range(len(pop)):
        bettersFitness[x] = popFitness[x]

    return sorted(bettersFitness.items(), key=lambda item: item[1], reverse=True)


def selectionIndex(rankpop):
    selectionResults = []
    eliteSize = 100  # tamanho da elite

    df = pd.DataFrame(np.array(rankpop), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    selectionResults = elite(rankpop, eliteSize)

    for i in range(0, len(rankpop) - eliteSize):
        pick = 100 * rd.random()

        for i in range(0, len(rankpop)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(rankpop[i][0])
                break

    return selectionResults


def elite(rankpop, eliteSize):
    selectedElite = []

    if type(rankpop[0]) == tuple:
        for i in range(0, eliteSize):
            selectedElite.append(rankpop[i][0])
    else:
        for i in range(0, eliteSize):
            selectedElite.append(rankpop[i])

    return selectedElite


def selectionCromo(pop, selected):
    rep = []

    for i in range(0, len(selected)):
        index = selected[i]
        rep.append(pop[index])

    return rep


def crossover(father, mother):
    child = []
    herancaFather = []
    herancaMother = []

    IndiceGeneFather = IndiceGeneMother = 0

    while (IndiceGeneFather == IndiceGeneMother):
        IndiceGeneFather = int(rd.random() * len(father))
        IndiceGeneMother = int(rd.random() * len(father))

    startGene = min(IndiceGeneFather, IndiceGeneMother)
    endGene = max(IndiceGeneFather, IndiceGeneMother)

    for x in range(startGene, endGene):
        herancaFather.append(father[x])

    for y in mother:
        if y not in herancaFather:
            herancaMother.append(y)

    child = herancaFather + herancaMother

    return child


def reproduction(pop, sizeElite):
    nextGeneration = []
    numDescendants = len(pop) - sizeElite
    mixedIndividuals = rd.sample(pop, len(pop))

    nextGeneration = elite(pop, sizeElite)

    for i in range(0, numDescendants):
        nextGeneration.append(crossover(mixedIndividuals[i], mixedIndividuals[len(pop) - i - 1]))

    return nextGeneration


def mutate(individual, mutationRate):
    for gene1 in range(len(individual)):
        if (rd.random() < mutationRate):
            gene2 = int(rd.random() * len(individual))

            aux1 = individual[gene1]
            aux2 = individual[gene2]

            individual[gene1] = aux2
            individual[gene2] = aux1

    return individual


def mutation(pop, mutationRate):
    mutatePop = []

    for i in range(len(pop)):
        mutatePop.append(mutate(pop[i], mutationRate))

    return mutatePop


def geneticAlgorithm(graphCities):
    pop = generate_population(graphCities)
    progress = []
    popFitness = fitness(pop, graphCities)
    aux = rank_routes(pop, popFitness)
    progress.append(1 / aux[0][1])
    numGenerations = 800  # numero de gerações

    print("Melhor distancia inicial: " + str(1 / aux[0][1]))
    print("Melhor rota inicial: " + str(pop[aux[0][0]]))

    for i in range(0, numGenerations):
        popFitness = fitness(pop, graphCities)
        selectedIndividuos = selection(pop, popFitness)
        popCrossover = reproduction(selectedIndividuos, 100)  # tamanho da elite
        nextGeneration = mutation(popCrossover, 0.01)  # taxa de mutação
        pop = nextGeneration
        progress.append(1 / rank_routes(pop, popFitness)[0][1])

    aux = rank_routes(pop, popFitness)
    print("Melhor distancia final: " + str(1 / aux[0][1]))
    bestRoute = pop[aux[0][0]]
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