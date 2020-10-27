from random import random, randint, choice

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from genetic.config import config_mlp, config_genetic_algo, \
    PRINT_INDIVIDUALS_LAYERS, PRINT_FITNESS_INDIVIDUALS, PRINT_ACCURACY_INDIVIDUALS, PRINT_INDIVIDUALS_FEATURES, \
    PRINT_F1_INDIVIDUALS, PRINT_RECALL_INDIVIDUALS, PRINT_PRECISION_INDIVIDUALS
from genetic.tools.CrossoverTool import CrossoverTool
from genetic.tools.MutationTool import MutationTool
from genetic.tools.MutationType import MutationType
from genetic_algo_params import FIT_ON


class Darwin:

    def __init__(self, target, nb_generation, population, corpus, labels):
        self.__crossover_tool = CrossoverTool()
        self.__mutation_tool = MutationTool()
        self.__target = target
        self.__nb_generation = nb_generation
        self.__population = population
        self.__fitness_history_best_individual = []
        self.__best_individual = None
        self.__labels = labels
        self.__corpus = corpus
    # ================================ Getters ================================
    @property
    def fitness_history_best_individual(self):
        return self.__fitness_history_best_individual

    @property
    def target(self):
        return self.__target

    @property
    def nb_generation(self):
        return self.__nb_generation

    @property
    def population(self):
        return self.__population

    @property
    def best_individual(self):
        return self.__best_individual

    # ================================ Functions for printing and plotting ================================
    def __print_generation(self, sorted_individuals):
        print("\nBest fitness :", self.__best_individual.fitness,
              "\naccuracy :", self.__best_individual.accuracy,
              "\nf1 :", self.__best_individual.f1,
              "\nprecision :", self.__best_individual.precision,
              "\nrecall :", self.__best_individual.recall,
              flush=True)

        if PRINT_INDIVIDUALS_FEATURES:
            print("Features :    ", [individual.features for individual in sorted_individuals], flush=True)
        if PRINT_INDIVIDUALS_LAYERS:
            print("Layers :    ", [individual.layers for individual in sorted_individuals], flush=True)
        if PRINT_FITNESS_INDIVIDUALS:
            print("Fitness :   ", [individual.fitness for individual in sorted_individuals], flush=True)
        if PRINT_ACCURACY_INDIVIDUALS:
            print("Accuracies :", [individual.accuracy for individual in sorted_individuals], flush=True)
        if PRINT_F1_INDIVIDUALS:
            print("F1s :", [individual.f1 for individual in sorted_individuals], flush=True)
        if PRINT_RECALL_INDIVIDUALS:
            print("Recalls :", [individual.recall for individual in sorted_individuals], flush=True)
        if PRINT_PRECISION_INDIVIDUALS:
            print("precisions :", [individual.precision for individual in sorted_individuals], flush=True)
        print("", flush=True)

    def print_fitness_history_best_individual(self):
        for i in range(len(self.__fitness_history_best_individual)):
            print("Generation ", i, ", fitness = ", self.__fitness_history_best_individual[i], flush=True)

    def plot_fitness_history_best_individual(self):
        plt.plot([i for i in range(len(self.__fitness_history_best_individual))],
                 self.__fitness_history_best_individual)
        plt.show()

    # ================================ Functions for mutations ================================
    def run_natural_selection(self):
        # Evolve to find the better solution
        for i in range(self.__nb_generation - 1):
            print("Generation ", i, flush=True)
            self.__evolve()

        # Computes the fitness for the last generation
        for individual in self.__population.individuals:
            x_train, x_test, y_train, y_test = self.__feature_computing(individual)
            self.__fitness(individual, x_train, y_train, x_test=x_test, y_test=y_test)

        print("Generation ", self.__nb_generation - 1, flush=True)
        sorted_individuals = sorted(self.__population.individuals, key=lambda x: x.fitness)
        self.__select_best_individual(sorted_individuals)
        self.__print_generation(sorted_individuals)

        # Print the best individual found
        print("Best individual found : \n" +
              self.__best_individual.__str__(), flush=True)

    def __select_best_individual(self, graded):
        best_individual_in_graded = graded[0]
        self.__fitness_history_best_individual.append(best_individual_in_graded.fitness)
        if self.__best_individual is None or best_individual_in_graded.fitness < self.__best_individual.fitness:
            self.__best_individual = best_individual_in_graded

    def __evolve(self,
                 retain=config_genetic_algo.RETAIN,
                 random_select_level=config_genetic_algo.RANDOM_SELECT_LEVEL,
                 prob_to_mutate=config_genetic_algo.PROB_TO_MUTATE):

        # Computes the fitness for all individuals in pop
        for individual in self.__population.individuals:
            x_train, x_test, y_train, y_test = self.__feature_computing(individual)
            self.__fitness(individual, x_train, y_train, x_test=x_test, y_test=y_test)

        # Sorts the population according to the fitness
        sorted_individuals = sorted(self.__population.individuals, key=lambda x: x.fitness)
        # Select the best individual in the sorted array sorted_individuals
        self.__select_best_individual(sorted_individuals)

        self.__print_generation(sorted_individuals)

        retain_length = int(len(sorted_individuals) * retain)
        parents = sorted_individuals[:retain_length]

        for individual in sorted_individuals[retain_length:]:
            # If the selection random level defined by user is greater than a random value
            if random_select_level > random():
                # Add to parents to increase the genetic diversity
                parents.append(individual)

        # crossover parents to create children
        # On calcul les crossovers
        parents_length = len(parents)
        desired_length = len(self.__population.individuals) - parents_length
        children = []
        while len(children) < desired_length:
            # Get index for male and female individuals
            male_index = randint(0, parents_length - 1)
            female_index = randint(0, parents_length - 1)
            if male_index != female_index:
                male = self.__population.get_individual(male_index)
                female = self.__population.get_individual(female_index)

                # Create crossover with the crossovertool classe
                child = self.__crossover_tool.cross(male, female)
                children.append(child)
        # Ajout de la liste des enfants aux parents pour reconstituer une population
        parents.extend(children)

        #Mutation
        for individual in parents:
            # If the mutation probability defined by user is greater than a random value between
            if prob_to_mutate > random():
                #mutation_type = MutationType.SUBSTITUTION
                mutation_type = choice(list(MutationType))
                self.__mutation_tool.mutate(mutation_type, individual,
                                            self.__population.min_layers,
                                            self.__population.max_layers)

        self.__population.individuals = parents

    def __fitness(self, individual, x_train, y_train, x_test=None, y_test=None):
        # If the fitness is not already computed
        if individual.fitness == -1:
            mlp = MLPClassifier(hidden_layer_sizes=individual.layers, activation=config_mlp.ACTIVATION,
                                solver=config_mlp.SOLVER, alpha=config_mlp.ALPHA, batch_size=config_mlp.BATCH_SIZE,
                                learning_rate=config_mlp.LEARNING_RATE,
                                learning_rate_init=config_mlp.LEARNING_RATE_INIT,
                                power_t=config_mlp.POWER_T, max_iter=config_mlp.MAX_ITER, shuffle=config_mlp.SHUFFLE,
                                random_state=config_mlp.RANDOM_STATE, tol=config_mlp.TOL, verbose=config_mlp.VERBOSE,
                                warm_start=config_mlp.WARM_START, momentum=config_mlp.MOMENTUM,
                                nesterovs_momentum=config_mlp.NESTEROVS_MOMENTUM,
                                early_stopping=config_mlp.EARLY_STOPPING,
                                validation_fraction=config_mlp.VALIDATION_FRACTION, beta_1=config_mlp.BETA_1,
                                beta_2=config_mlp.BETA_2, epsilon=config_mlp.EPSILON,
                                n_iter_no_change=config_mlp.N_ITER_NO_CHANGE
                                )
            mlp.fit(x_train, y_train)
            predictions = mlp.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, pos_label='BUG')
            precision = precision_score(y_test, predictions, pos_label='BUG')
            recall = recall_score(y_test, predictions, pos_label='BUG')

            if FIT_ON == 'f1' :
                fitness_value = abs(self.__target - f1)
            elif FIT_ON == 'recall':
                fitness_value = abs(self.__target - recall)
            elif FIT_ON == 'precision':
                fitness_value = abs(self.__target - recall)
            else:
                fitness_value = abs(self.__target - accuracy)

            individual.accuracy = accuracy
            individual.f1 = f1
            individual.precision = precision
            individual.recall = recall
            individual.fitness = fitness_value

        return individual.fitness

    def __feature_computing(self, individual):
        # TF-IDF.
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 3),stop_words={'english'}, sublinear_tf=True)
        X = vectorizer.fit_transform(self.__corpus)
        feature_names = vectorizer.get_feature_names()
        # Feature selection.
        feature_selection = True
        if feature_selection:
            # k_best = 20000
            k_best = individual.features
            #print("Extracting %d best features by a chi-squared test" % k_best)
            ch2 = SelectKBest(chi2, k=k_best)
            X = ch2.fit_transform(X, self.__labels)
            if feature_names:  # keep selected feature names.
                feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
            #print("n_samples: %d, n_features: %d." % X.shape, flush=True)

        return train_test_split(X, self.__labels, train_size=0.75, test_size=0.25, random_state=12)

