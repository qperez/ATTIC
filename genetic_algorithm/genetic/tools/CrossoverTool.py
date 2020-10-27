from genetic.objects.Individual import Individual


class CrossoverTool:

    def cross(self, male_individual, female_individual):
        if male_individual is None:
            raise Exception("Male individual cannot be null for a crossover")

        if female_individual is None:
            raise Exception("Female individual cannot be null for a crossover")

        if len(male_individual.layers) < 2:
            raise Exception("Crossover cannot be applied on individual with a number of layers < 2")

        child_individual = Individual()
        child_individual.layers = self.__crossover_layers(male_individual, female_individual)
        child_individual.update_min_max_neurons()
        child_individual.update_nb_layers()
        child_individual.features = self.__crossover_feature(male_individual, female_individual)
        child_individual.max_features = male_individual.max_features
        child_individual.min_features = female_individual.min_features

        return child_individual

    def __crossover_layers(self, male_individual, female_individual):
        half = int(len(male_individual.layers) / 2)
        child_layers = list(male_individual.layers)[:half] + list(female_individual.layers)[half:]
        child_layers = tuple(child_layers)
        return child_layers

    def __crossover_feature(self, male_individual, female_individual):
        return int((male_individual.features + female_individual.features) / 2)
