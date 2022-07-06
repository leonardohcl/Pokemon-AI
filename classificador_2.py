from collections import Counter
import math
import numpy as np
from Pokemon import Pokedex, PokemonType, PokemonAI
from sklearn import svm, neighbors, tree, naive_bayes, feature_selection


# Function to make easier to print the contents of a group of samples
def type_counter_str(data: list, types: list):
    occurences = Counter([types[x] for x in data])
    return str.join(" | ", [f"{x}: {occurences[x]}" for x in types])

# define base pokedex for reference
dex = Pokedex("pokemon.csv")

# define types used
types = [PokemonType.Fire, PokemonType.Water, PokemonType.Grass]

# get balanced data considering data augmentation
data = PokemonAI.get_balanced_data(dex, types,
                                    ["h_flip", "v_flip", "rot_45", "rot_90",
                                    "rot_135", "rot_180", "rot_225", "rot_270",
                                    "zoom_0", "zoom_1", "zoom_2", "zoom_3", "zoom_4",
                                    "thin", "short"])

# get complete feature vectors and labels for the samples
input_raw = [pkm.histogram() for pkm in data]
expected_labels = [types.index(pkm.pokemon_type) for pkm in data]

# create filter selector to reduce feature vectors dimensionality  
feature_selector = feature_selection.SelectKBest(feature_selection.chi2, k=100)
# apply filter to feature vectos
input_reduced = feature_selector.fit_transform(input_raw, expected_labels)

# split samples into training and evaluation groups
train, eval, _ = PokemonAI.split_data(input_reduced, 0.7, 0.3)
train_labels, eval_labels, _ = PokemonAI.split_data(expected_labels, 0.7, 0.3)

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
