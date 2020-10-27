import codecs
import json
import os
import warnings

from genetic.objects.Darwin import Darwin
from genetic.objects.Population import Population
from genetic_algo_params import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    raw_data = []
    data_directory = ".." + os.path.sep + "data"
    for filename in os.listdir(data_directory):
        with codecs.open(data_directory + os.path.sep + filename, "r", "utf-8") as fin:
            raw_data += json.load(fin)

    # Corpus building.
    corpus = []
    labels = []
    n_bug = 0
    boost_summary = 3
    for n_file in raw_data:
        txt = ""
        for i in range(boost_summary):
            txt += n_file["summary"] + " "
        corpus.append(txt + " " + n_file["description"])
        labels.append(n_file["label"])
        if n_file["label"] == "BUG":
            n_bug += 1
    print(f"{n_bug} BUG / {len(labels)} TOTAL", flush=True)

    population = Population(POPULATION_SIZE, MIN_NEURONS, MAX_NEURONS, MIN_LAYERS, MAX_LAYERS, MIN_FEATURES, MAX_FEATURES)
    darwin = Darwin(TARGET, NB_GENERATION, population, corpus, labels)
    darwin.run_natural_selection()
    darwin.print_fitness_history_best_individual()
    print("Fitness history array = ", darwin.fitness_history_best_individual, flush=True)
    darwin.plot_fitness_history_best_individual()
