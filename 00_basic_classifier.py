from collections import Counter
import math
from random import shuffle
import numpy as np
from Pokemon import Pokedex, PokemonType, PokemonAI
from sklearn import svm, neighbors, tree, naive_bayes


# Function to make easier to print the contents of a group of samples
def type_counter_str(data: list, types: list):
    occurences = Counter([types[x] for x in data])
    return str.join(" | ", [f"{x}: {occurences[x]}" for x in types])

# define base pokedex for reference
dex = Pokedex("pokemon.csv")

# define types used
types = [PokemonType.Fire, PokemonType.Water, PokemonType.Grass]

# get data
data = []
for pkm_type in types:
    data += dex.get_type(pkm_type)

# get feature vectors
input_data = [pkm.histogram() for pkm in data]

# suffle data
shuffle(input_data)

# get expected labels for feature vectors
expected_labels = [types.index(pkm.pokemon_type) for pkm in data]

# get index to separate training and evaluation groups
cut_index = math.floor(0.7 * len(data))

# separate training and evaluation samples
train = input_data[:cut_index]
train_labels = expected_labels[:cut_index]
eval = input_data[cut_index:]
eval_labels = expected_labels[cut_index:]

# print groups composition
print(f"training: {len(train)} samples -> {type_counter_str(train_labels, types)}")
print(f"    eval: {len(eval)} samples -> {type_counter_str(eval_labels, types)}")

# DEFINE A CLASSIFIER

# SVM
# classifier = svm.SVC(kernel='poly')

# KNN
classifier = neighbors.KNeighborsClassifier()

# Decision Tree
# classifier = tree.DecisionTreeClassifier()

# naive bayes
# classifier = naive_bayes.GaussianNB()

# fit classifier model to training samples 
classifier.fit(train, train_labels)
# get the predictions from the fitted model for the evaluation samples
predictions = classifier.predict(eval)

# create empty square matrix to build confusion matrix
confusion = np.zeros((len(types), len(types)))

# count classifications for each evaluation sample on its respective place for the matrix
for i in range(len(predictions)):
    confusion[int(eval_labels[i])][int(predictions[i])] += 1

# get model accuracy from confusion matrix
acc = sum([confusion[i][i] for i in range(len(types))]) / len(predictions)

# get model precision for each class
for i in range(len(types)):
    precision = confusion[i][i] / sum(confusion[i])
    print(f"Precision for {types[i]}: {100*precision:.2f}%")

print(f"Accuracy: {100*acc:.2f}%")
print(confusion)
