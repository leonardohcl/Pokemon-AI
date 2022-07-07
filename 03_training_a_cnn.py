from collections import Counter
import torch
from torchvision import models
import torch.nn as nn
import random
import numpy as np
from Pokemon import Pokedex, PokemonAI, PokemonType
from augmentation_helper import AUGMENTATIONS

def occurence_str(data, types):
    occurences = Counter([x.pokemon_type for x in data])
    return str.join(" | ", [f"{x}: {occurences[x]}" for x in types])

dex = Pokedex("pokemon.csv")

# TRAINING PARAMETERS
learning_rate = 0.1
learning_rate_drop = 0.9
learning_rate_drop_step_size = 2
epochs = 1
types = [PokemonType.Fire, PokemonType.Grass, PokemonType.Water]
max_generation = 0
folds = 1
augmentation_count = 0
batch_size = 50
use_transfer_learning = True
train_prct = 0.8
eval_prct = 0.1
save_trained_model = False

# GET DATA
selected_augmentations = random.sample(AUGMENTATIONS, augmentation_count) if augmentation_count > 0 else []
data = PokemonAI.get_balanced_data(dex, types, variations=selected_augmentations, max_generation=max_generation)

train, eval, test = PokemonAI.split_data(data, train_prct, eval_prct)

print(
    f"training set: {len(train)} samples -> {occurence_str(train, types)}")
print(
    f"    eval set: {len(eval)} samples -> {occurence_str(eval, types)}")
print(
    f"    test set: {len(eval)} samples -> {occurence_str(test, types)}\n")


# SELECT CNN MODEL
model = models.alexnet(pretrained=use_transfer_learning)
# model = models.resnet50(pretrained=use_transfer_learning)

# changed output layer to match the number of classes used
model.classifier[6] = nn.Linear(4096, len(types))
# model.fc = nn.Linear(2048, len(classes))


# TRAIN MODEL
if folds > 1:
    trained_model = PokemonAI.k_fold_training(
        model=model,
        input_data=train+eval,
        types=types,
        epochs=epochs,
        k=folds,
        learning_rate=learning_rate,
        use_gpu=True,
        error_fn=nn.CrossEntropyLoss(),
        plot_loss=False,
        batch_size=batch_size,
        learning_rate_drop=learning_rate_drop,
        learning_rate_drop_step_size=learning_rate_drop_step_size,
        using_transfer_learning=use_transfer_learning
    )
else:
    trained_model = PokemonAI.train(
        model=model,
        train=train,
        eval=eval,
        types=types,
        epochs=epochs,
        learning_rate=learning_rate,
        use_gpu=True,
        error_fn=nn.CrossEntropyLoss(),
        plot_loss=False,
        batch_size=batch_size,
        learning_rate_drop=learning_rate_drop,
        learning_rate_drop_step_size=learning_rate_drop_step_size,
        using_transfer_learning=use_transfer_learning
    )[0]


# Create confusion matrix
confusion = np.zeros((len(types), len(types)))
prediction = PokemonAI.predict_pokemons(
    trained_model, test, types, use_gpu=True, batch_size=batch_size, using_transfer_learning=use_transfer_learning)
for idx in range(len(test)):
    expected = types.index(test[idx].pokemon_type)
    predicted = types.index(prediction[idx])
    confusion[expected][predicted] += 1

# Get accuracy and precisions for the model
acc = sum([confusion[i][i] for i in range(len(types))]) / len(test)

for i in range(len(types)):
    precision = confusion[i][i] / sum(confusion[i])
    print(f"Precision for {types[i]}: {100*precision:.2f}%")

print(f"\nAccuracy: {100*acc:.2f}%")

print("\nConfusion matrix:")
print(confusion)

# Save trained model to use later
if save_trained_model:
    torch.save(trained_model.state_dict(), "trained_model")
