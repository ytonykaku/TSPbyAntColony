import sys
import functions as f
import numpy as np


def main():
    print("\nGenetic algorithm applied to the traveling salesman problem.\n")

    try:
        graphCities = f.start("instances.txt")
    except:
        print(f'File "{sys.argv[1]}" not found.')
        return

    f.geneticAlgorithm(graphCities)

main()