from random import randint
from genetic.tools.MutationType import MutationType


class MutationTool:

    def mutate(self, mutation_type, individual, min_layers, max_layers):
        if mutation_type not in MutationType:
            raise TypeError("MutationType is invalid")

        if individual is None:
            raise Exception("Individual cannot be null for mutation")

        self.__mutation_nb_features(individual)
        return self.__mutation_layers(mutation_type, individual, min_layers, max_layers)

    def __mutation_nb_features(self, individual):
        individual.set_random_feature_value()

    def __mutation_layers(self, mutation_type, individual, min_layers, max_layers):

        if mutation_type == MutationType.ADDITION and len(individual.layers) < max_layers:
            individual.add_layer_at_layer_index(randint(0, len(individual.layers) - 1))
            individual.update_nb_layers()
            return individual

        if mutation_type == MutationType.DELETION and len(individual.layers) > min_layers:
            individual.del_layer_at_layer_index(randint(0, len(individual.layers) - 1))
            individual.update_nb_layers()
            return individual

        individual.set_random_value_at_layer_index(randint(0, len(individual.layers) - 1))

        return individual
