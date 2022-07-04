from collections import Counter
import torch
from torchvision import models
import torch.nn as nn
import random
import numpy as np

from Pokemon import Pokedex, PokemonAI, PokemonType


def occurence_str(data):
    occurences = Counter([x.pokemon_type for x in data])
    return str.join(" | ", [f"{x}: {occurences[x]}" for x in classes])


dex = Pokedex("pokemon.csv")

learning_rate = 0.1
learning_rate_drop = 0.9
learning_rate_drop_step_size = 2
epochs = 3
classes = [PokemonType.Fire, PokemonType.Grass]
max_generation = 0
folds = 0
augmentation_count = 0
batch_size = 50
use_transfer_learning = True
fine_tune_output_only = False
train_prct = 0.8
eval_prct = 0.1

augmentations = ["h_flip", "v_flip",
                 "rot_90", "rot_180", "rot_270",
                 "rot_45", "rot_135", "rot_225",
                 "zoom_0", "zoom_1", "zoom_2", "zoom_3", "zoom_4",
                 "darken_10", "darken_25", "darken_50",
                 "lighten_10", "lighten_25", "lighten_50",
                 "fat", "thin",
                 ]

selected_augmentations = random.sample(
    augmentations, augmentation_count) if augmentation_count > 0 else []

data = PokemonAI.get_balanced_data(
    dex, classes, variations=selected_augmentations, max_generation=max_generation)

train, eval, test = PokemonAI.split_data(data, train_prct, eval_prct)

print(
    f"training set: {len(train)} samples -> {occurence_str(train)}")
print(
    f"    eval set: {len(eval)} samples -> {occurence_str(eval)}")
print(
    f"    test set: {len(eval)} samples -> {occurence_str(test)}\n")

model = models.alexnet(pretrained=use_transfer_learning)
# model = models.vgg11_bn(pretrained=use_transfer_learning)
# model = models.densenet121(pretrained=use_transfer_learning)
# model = models.resnet18(pretrained=use_transfer_learning)
# model = models.resnet50(pretrained=use_transfer_learning)
# model = models.resnet152(pretrained=use_transfer_learning)
# model = models.efficientnet_b2(pretrained=use_transfer_learning)

if use_transfer_learning and fine_tune_output_only:
    for param in model.parameters():
        param.requires_grad = False

model.classifier[6] = nn.Linear(4096, len(classes))
# model.classifier[6] = nn.Linear(4096, len(classes))
# model.classifier = nn.Linear(1024, len(classes))
# model.fc = nn.Linear(512, len(classes))
# model.fc = nn.Linear(2048, len(classes))
# model.fc = nn.Linear(2048, len(classes))
# model.classifier[1] = nn.Linear(1408, len(classes))

if folds > 1:
    trained_model = PokemonAI.k_fold_training(
        model=model,
        input_data=train+eval,
        classes=classes,
        epochs=epochs,
        k=folds,
        learning_rate=learning_rate,
        use_gpu=True,
        error_fn=nn.CrossEntropyLoss(),
        plot_loss=False,
        batch_size=batch_size,
        learning_rate_drop=learning_rate_drop,
        learning_rate_drop_step_size=learning_rate_drop_step_size
    )
else:
    trained_model = PokemonAI.train(
        model=model,
        train=train,
        eval=eval,
        classes=classes,
        epochs=epochs,
        learning_rate=learning_rate,
        use_gpu=True,
        error_fn=nn.CrossEntropyLoss(),
        plot_loss=False,
        batch_size=batch_size,
        learning_rate_drop=learning_rate_drop,
        learning_rate_drop_step_size=learning_rate_drop_step_size
    )[0]


confusion = np.zeros((len(classes), len(classes)))
prediction = PokemonAI.predict(
    trained_model, test, classes, use_gpu=True, batch_size=batch_size)
for idx in range(len(test)):
    expected = classes.index(test[idx].type)
    predicted = classes.index(prediction[idx])
    confusion[expected][predicted] += 1

acc = sum([confusion[i][i] for i in range(len(classes))]) / len(test)

for i in range(len(classes)):
    precision = confusion[i][i] / sum(confusion[i])
    print(f"Precision for {classes[i]}: {100*precision:.2f}%")

print(f"\nAccuracy: {100*acc:.2f}%")

print("\nConfusion matrix:")
print(confusion)

torch.save(trained_model.state_dict(), "trained_model")
