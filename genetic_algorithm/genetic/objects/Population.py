from random import randint

from genetic.objects.Individual import Individual


class Population:
    def __init__(self, count=0, min_nb_neurons=0, max_nb_neurons=0, min_layers=0, max_layers=0, min_features=0, max_features=0):
        self.__individuals = [Individual(min_nb_neurons, max_nb_neurons, min_features, max_features, randint(min_layers, max_layers)) for _ in range(count)]
        self.__min_layers = min_layers
        self.__max_layers = max_layers
        self.__min_features = min_features
        self.__max_features = max_features
        self.__min_nb_neurons = min_nb_neurons
        self.__max_nb_neurons = max_nb_neurons

    # ================================ Getters ================================
    @property
    def individuals(self):
        return self.__individuals

    @property
    def min_layers(self):
        return self.__min_layers

    @property
    def max_layers(self):
        return self.__max_layers

    @property
    def min_features(self):
        return self.__min_features

    @property
    def max_features(self):
        return self.__max_features

    # ================================ Setters ================================
    @individuals.setter
    def individuals(self, individuals):
        self.__individuals = individuals

    # ================================ Methods ================================
    def get_individual(self, individual_index):
        if individual_index > len(self.__individuals) or individual_index < 0:
            raise IndexError("The index to get an individual is out out of population range")

        return self.__individuals[individual_index]

    def __str__(self):
        ret = ""
        for i in range(len(self.__individuals)):
            ret + "Individual number : " + str(i) + "\n" \
                    + self.__individuals[i].__str__() + "\n\n"
        return ret