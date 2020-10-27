from random import randint


class Individual:

    def __init__(self, min_neurons=0, max_neurons=0, min_features=0, max_features=0, nb_layers=0):
        self.__min_neurons = min_neurons
        self.__max_neurons = max_neurons
        self.__min_features = min_features
        self.__max_features = max_features
        self.__nb_layers = nb_layers
        self.__layers = tuple(i for i in [randint(self.__min_neurons
                                                   , self.__max_neurons) for _ in range(self.__nb_layers)])
        self.__features = randint(min_features, max_features)

        self.__fitness = -1
        self.__accuracy = 0.0
        self.__f1 = 0.0
        self.__recall = 0.0
        self.__precision = 0.0
        self.__mutations = []

    @property
    def fitness(self):
        return self.__fitness

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def f1(self):
        return self.__f1

    @property
    def recall(self):
        return self.__recall

    @property
    def precision(self):
        return self.__precision

    @property
    def min_neurons(self):
        return self.__min_neurons

    @property
    def max_neurons(self):
        return self.__max_neurons

    @property
    def min_features(self):
        return self.__min_features

    @property
    def max_features(self):
        return self.__max_features

    @property
    def features(self):
        return self.__features

    @property
    def nb_layers(self):
        return self.__nb_layers

    @property
    def layers(self):
        return self.__layers

    @fitness.setter
    def fitness(self, fitness):
        self.__fitness = fitness

    @accuracy.setter
    def accuracy(self, accuracy):
        self.__accuracy = accuracy

    @f1.setter
    def f1(self, f1):
        self.__f1 = f1

    @recall.setter
    def recall(self, recall):
        self.__recall = recall

    @precision.setter
    def precision(self, precision):
        self.__precision = precision

    @min_neurons.setter
    def min_neurons(self, min_neurons):
        self.__min_neurons = min_neurons

    @max_neurons.setter
    def max_neurons(self, max_neurons):
        self.__max_neurons = max_neurons

    @min_features.setter
    def min_features(self, min_features):
        self.__min_features = min_features

    @max_features.setter
    def max_features(self, max_features):
        self.__max_features = max_features

    @features.setter
    def features(self, features):
        self.__features = features

    @nb_layers.setter
    def nb_layers(self, nb_layers):
        self.__nb_layers = nb_layers

    @layers.setter
    def layers(self, layers):
        self.__layers = layers

    def update_min_max_neurons(self):
        self.__min_neurons = min(list(self.__layers))
        self.__max_neurons = max(list(self.__layers))

    def update_nb_layers(self):
        self.__nb_layers = len(self.__layers)

    def add_layer_at_layer_index(self, layer_index):
        self.__layers = list(self.__layers)
        self.__layers.insert(layer_index, randint(self.__min_neurons, self.__max_neurons))
        self.__layers = tuple(self.__layers)

    def set_random_value_at_layer_index(self, layer_index):
        self.__layers = list(self.__layers)
        self.__layers[layer_index] = randint(self.__min_neurons, self.__max_neurons)
        self.__layers = tuple(self.__layers)

    def del_layer_at_layer_index(self, layer_index):
        self.__layers = list(self.__layers)
        del self.__layers[layer_index]
        self.__layers = tuple(self.__layers)

    def set_random_feature_value(self):
        self.__features = randint(self.__features - int(self.__features * 0.1) if self.__features - int(
            self.__features * 0.1) >= self.__min_features else self.min_features,
                                  self.__features + int(self.__features * 0.1) if self.__features + int(
                                      self.__features * 0.1) <= self.__max_features else self.max_features)

    def __str__(self):
        return "Layers : " + str(self.__layers) + \
               "\nFeatures : " + str(self.__features) + \
               "\nFitness : " + str(self.__fitness) + \
               "\nAccuracy : " + str(self.__accuracy) + \
               "\nF1 : " + str(self.__f1) + \
               "\nPrecision : " + str(self.__precision) + \
               "\nRecall : " + str(self.__recall)
