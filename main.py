from collections import Counter
from random import shuffle
import numpy as np
from Pokemon import Pokedex, PokemonType, PokemonAI
from sklearn import svm, neighbors, tree, ensemble, naive_bayes, feature_selection, linear_model


def type_counter_str(data: list[int], types: list[PokemonType]):
    occurences = Counter([types[x] for x in data])
    return str.join(" | ", [f"{x}: {occurences[x]}" for x in types])


dex = Pokedex("pokemon.csv")

types = [t for t in [PokemonType(i) for i in range(18)] if t not in [PokemonType.Flying]]
max_generation = 0

data = PokemonAI.get_balanced_data(dex, types,
                                    ["h_flip", "v_flip", "rot_45", "rot_90",
                                    "rot_135", "rot_180", "rot_225", "rot_270",
                                    "zoom_0", "zoom_1", "zoom_2", "zoom_3", "zoom_4",
                                    "thin", "short"],
                                   with_legendary=True,
                                   max_generation=max_generation)

input_raw = [pkm.histogram() for pkm in data]
expected_labels = [types.index(pkm.pokemon_type) for pkm in data]

input_reduced = feature_selection.SelectKBest(
    feature_selection.chi2, k=210).fit_transform(input_raw, expected_labels)
# input_reduced = input_raw

train, eval, test = PokemonAI.split_data(
    input_reduced, train_prct=0.7, eval_prct=0.3)
train_labels, eval_labels, test_labels = PokemonAI.split_data(
    expected_labels, train_prct=0.7, eval_prct=0.3)

print(
    f"training: {len(train)} samples -> {type_counter_str(train_labels, types)}")
print(
    f"    eval: {len(eval)} samples -> {type_counter_str(eval_labels, types)}")

# SVM
# classifier = svm.SVC(kernel='poly')

# KNN
# classifier = neighbors.KNeighborsClassifier()
# K*
# classifier = neighbors.NearestCentroid()

# Decision Tree
# classifier = tree.DecisionTreeClassifier()
# Random Forest
# classifier = ensemble.RandomForestClassifier()

# naive bayes
# classifier = naive_bayes.GaussianNB()

# Ensemble!
classifier = ensemble.VotingClassifier(
    estimators=[('svm', svm.SVC(kernel='poly')),
                ('knn', neighbors.KNeighborsClassifier()),
                ('dt', tree.DecisionTreeClassifier()),
                ('nb', naive_bayes.GaussianNB())])

# classifier = linear_model.Perceptron()


classifier.fit(train, train_labels)
out = classifier.predict(eval)

confusion = np.zeros((len(types), len(types)))
for i in range(len(out)):
    confusion[int(eval_labels[i])][int(out[i])] += 1

acc = sum([confusion[i][i] for i in range(len(types))]) / len(out)

for i in range(len(types)):
    precision = confusion[i][i] / sum(confusion[i])
    print(f"Precision for {types[i]}: {100*precision:.2f}%")

print(f"Accuracy: {100*acc:.2f}%")
print(confusion)
