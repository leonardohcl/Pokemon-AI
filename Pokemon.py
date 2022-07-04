import copy
from enum import Enum
import statistics
from typing import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from random import shuffle, sample


class PokemonType(Enum):
    Grass = 0
    Poison = 1
    Fire = 2
    Flying = 3
    Water = 4
    Bug = 5
    Normal = 6
    Dark = 7
    Electric = 8
    Ground = 9
    Ice = 10
    Fairy = 11
    Fighting = 12
    Psychic = 13
    Rock = 14
    Steel = 15
    Ghost = 16
    Dragon = 17

    def __int__(self):
        return self.value

    def __str__(self) -> str:
        return self.name


class Pokemon:
    def __init__(self, name: str, pokedex_number: int, type_1: str, type_2: str, is_legendary: bool, generation: int, img_variation: str = None) -> None:
        self.name = name
        self.pokedex_number = pokedex_number
        self.type_1 = type_1
        self.type_2 = type_2
        self.is_legendary = is_legendary
        self.generation = generation
        self.img_variation = img_variation

    def __str__(self) -> str:
        if(self.img_variation != None):
            return f"({self.pokedex_number}){self.name} ({self.img_variation}) - {self.pokemon_type}"
        return f"{self.name} - {self.pokemon_type}"

    @property
    def pokemon_type(self) -> PokemonType:
        return PokemonType[self.type_1]

    def base_image(self) -> Image.Image:
        return Image.open(f"pokemon/{self.pokedex_number}.png")

    def image(self) -> Image.Image:
        if(self.img_variation != None):
            return Image.open(f"data_augmentation/{self.img_variation}/{self.pokedex_number}.png")
        return self.base_image()

    def histogram(self) -> tuple[list, list, list, list]:
        hist = self.image().histogram()
        red = hist[:256]
        green = hist[256:512]
        blue = hist[512:768]
        alpha = hist[768:]

        red[0] -= alpha[0]
        if red[0] < 0:
            red[0] = 0

        green[0] -= alpha[0]
        if green[0] < 0:
            green[0] = 0

        blue[0] -= alpha[0]
        if blue[0] < 0:
            blue[0] = 0

        return red + green + blue

    def tensor(self):
        return transforms.Compose([transforms.ToTensor()])(self.image().convert("RGB"))


class Pokedex:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        self.pokemons: list[Pokemon] = []
        for _, row in data.iterrows():
            self.pokemons.append(
                Pokemon(row["name"], row["pokedex_number"], row["type_1"], row["type_2"], True if row["is_legendary"] == 1 else False, row["generation"]))

    def __len__(self):
        return len(self.pokemons)

    def get_pokemon(self, number) -> Pokemon:
        return self.pokemons[number-1]

    def get_type(self, pokemon_type: PokemonType, with_legendary: bool = True, max_generation: int = 0) -> list[Pokemon]:
        filtered = list(filter(lambda pkm: pkm.pokemon_type ==
                        pokemon_type, self.pokemons))
        if not with_legendary:
            filtered = list(
                filter(lambda pkm: pkm.is_legendary == False, filtered))
        if max_generation > 0:
            filtered = list(
                filter(lambda pkm: pkm.generation <= max_generation, filtered))
        return filtered


class PokemonAI:

    @staticmethod
    def get_balanced_data(dex:Pokedex, classes, variations=[], with_legendary: bool = True, max_generation: int = 0):
        entries = {}
        smallest_set_size = float("inf")
        for pkm_class in classes:
            class_list = dex.get_type(
                pkm_class, with_legendary=with_legendary, max_generation=max_generation)

            sample_count = len(class_list)
            if sample_count < smallest_set_size:
                smallest_set_size = sample_count

            shuffle(class_list)
            entries[pkm_class] = class_list

        data = []
        sample_count = smallest_set_size * (len(variations)+1)
        for pkm_class in classes:
            available_slots = sample_count - len(entries[pkm_class])
            if available_slots > 0:
                augmentation = [Pokemon(pkm.name, pkm.pokedex_number, pkm.type_1, pkm.type_2, pkm.is_legendary, pkm.generation, var) for pkm in entries[pkm_class] for var in variations]
                shuffle(augmentation)
                entries[pkm_class] += augmentation[:available_slots]
                shuffle(entries[pkm_class])
            elif available_slots < 0:
                entries[pkm_class] = entries[pkm_class][:sample_count]

            data += entries[pkm_class]
        shuffle(data)
        return data

    @staticmethod
    def split_data(data: list, train_prct: float, eval_prct: float):
        sample_count = len(data)
        train_size = math.floor(sample_count * train_prct)
        eval_size = math.floor(sample_count * eval_prct)

        return data[:train_size], data[train_size:train_size + eval_size], data[train_size + eval_size:]

    @staticmethod
    def get_vector_output(result: list) -> int:
        top = max(result)
        return result.index(top)

    @staticmethod
    def create_folds(data, fold_count: int):
        # shuffle the list
        shuffled_data = sample(data, len(data))
        # get fold size
        fold_size = len(shuffled_data) // fold_count
        # separate in equally sized folds
        folds = [shuffled_data[i*fold_size:(i+1)*fold_size]
                 for i in range(fold_count)]
        # count left out indexes
        left_out = len(shuffled_data) % fold_count
        # distribute left out indexes into folds
        for idx in range(left_out):
            folds[idx].append(shuffled_data[-1*(idx+1)])

        return folds

    @staticmethod
    def get_weights(model):
        return copy.deepcopy(model.state_dict())

    @staticmethod
    def get_expected_tensor(idx: int, size: int):
        aux = np.zeros(size)
        aux[idx] = 1.0
        return torch.tensor(aux, dtype=torch.float)

    @staticmethod
    def get_fold_groups(current_fold, folds):
        train = []
        for x in folds:
            if x != current_fold:
                train += x
        return current_fold, train

    def create_batches(data, batch_size):
        batch_count = math.floor(len(data)/batch_size) + 1
        batches = []
        for batch_idx in range(batch_count):
            start_idx = batch_idx*batch_size
            end_idx = start_idx + batch_size
            batches.append(data[start_idx:end_idx])
        return [batch for batch in batches if len(batch) > 0]

    @staticmethod
    def evaluate(model, data, labels, error_fn, device):
        # get correct labels
        _, correct_labels = torch.max(labels.to(device), 1)

        output = model(data.to(device))
        loss_output = error_fn(output, labels.to(device))

        _, predictions = torch.max(output, 1)

        correct = torch.sum(predictions == correct_labels)
        loss = loss_output.item() / len(data)
        accuracy = float(correct) / len(data)

        return loss, accuracy, loss_output

    @staticmethod
    def process_input(model, data: torch.Tensor, labels, error_fn, optimization_fn=None, device=torch.device("cpu"), batch_size: int = 10, learn=False):
        # clean the gradient adjust

        if learn:
            model.train()
        else:
            model.eval()

        data_input = PokemonAI.create_batches(data, batch_size)
        data_labels = PokemonAI.create_batches(labels, batch_size)
        batch_count = len(data_input)
        loss = 0
        accuracy = 0

        for batch_idx in range(batch_count):
            batch_input = torch.stack([pkm.tensor()
                                      for pkm in data_input[batch_idx]])
            batch_labels = torch.stack(data_labels[batch_idx])
            if learn:
                optimization_fn.zero_grad()

            batch_loss, batch_accuracy, loss_info = PokemonAI.evaluate(
                model, batch_input, batch_labels, error_fn=error_fn, device=device)

            accuracy += batch_accuracy
            loss += batch_loss

            if learn:
                loss_info.backward()
                optimization_fn.step()

            if device == torch.device("cuda:0"):
                torch.cuda.empty_cache()

        return model, accuracy/batch_count, loss/batch_count

    @staticmethod
    def k_fold_training(model, input_data, classes, epochs: int,
                        k: int = 3,
                        optimizer=optim.SGD,
                        weight_decay: float = 0,
                        learning_rate: float = 0.1,
                        error_fn=nn.MSELoss(),
                        plot_acc=True,
                        plot_loss=True,
                        batch_size=10,
                        use_gpu=False,
                        learning_rate_drop=0,
                        learning_rate_drop_step_size=0):

        print("Setting variables")

        untrained = PokemonAI.get_weights(model)

        best_acc = 0.0
        best_weights = PokemonAI.get_weights(model)

        # Start epoch history track
        train_acc_hist = []
        train_loss_hist = []
        eval_acc_hist = []
        eval_loss_hist = []

        print("Obtaining Folds")

        folds = PokemonAI.create_folds(input_data, k)

        print(f"Got {len(folds)} folds with {len(folds[0])} samples")

        for fold in folds:
            # get folds for training and evaluating
            eval, train = PokemonAI.get_fold_groups(fold, folds)

            fold_best_model, fold_best_acc, fold_best_loss, fold_train_acc, fold_train_loss, fold_eval_acc, fold_eval_loss = PokemonAI.train(
                model=model,
                train=train,
                eval=eval,
                classes=classes,
                epochs=epochs,
                optimizer=optimizer,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                error_fn=error_fn,
                batch_size=batch_size,
                use_gpu=use_gpu,
                plot_loss=False,
                plot_acc=False,
                learning_rate_drop=learning_rate_drop,
                learning_rate_drop_step_size=learning_rate_drop_step_size)

            if fold_best_acc > best_acc:
                best_acc = fold_best_acc
                best_weights = PokemonAI.get_weights(fold_best_model)

            train_acc_hist.append(fold_train_acc)
            train_loss_hist.append(fold_train_loss)
            eval_acc_hist.append(fold_eval_acc)
            eval_loss_hist.append(fold_eval_loss)

            # reset model weights
            model.load_state_dict(untrained)

        avg_train_acc = np.zeros(epochs)
        avg_train_loss = np.zeros(epochs)
        avg_eval_acc = np.zeros(epochs)
        avg_eval_loss = np.zeros(epochs)

        for epoch in range(epochs):
            for fold in range(k):
                avg_train_acc[epoch] += train_acc_hist[fold][epoch]
                avg_train_loss[epoch] += train_loss_hist[fold][epoch]
                avg_eval_acc[epoch] += eval_acc_hist[fold][epoch]
                avg_eval_loss[epoch] += eval_loss_hist[fold][epoch]
            avg_train_acc[epoch] /= k
            avg_train_loss[epoch] /= k
            avg_eval_acc[epoch] /= k
            avg_eval_loss[epoch] /= k

        # Plot graphs
        if plot_acc:
            fig, ax = plt.subplots()
            ax.plot(avg_train_acc, 'r', label="Average training acurracy")
            ax.plot(avg_eval_acc, 'b', label="Average evaluation accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
            ax.legend()
            plt.show()

        if plot_loss:
            fig, ax = plt.subplots()
            ax.plot(avg_train_loss, 'r', label="Average training loss")
            ax.plot(avg_eval_loss, 'b', label="Average evaluation loss")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
            ax.legend()
            plt.show()

        # load best model weights
        model.load_state_dict(best_weights)

        return model

    @staticmethod
    def train(model: nn.Module, train, eval, classes, epochs: int,
              optimizer=optim.SGD,
              weight_decay: float = 0,
              learning_rate: float = 0.1,
              error_fn=nn.MSELoss(),
              plot_acc=True,
              plot_loss=True,
              use_gpu=False,
              batch_size=10,
              learning_rate_drop=0,
              learning_rate_drop_step_size=0) -> nn.Module:

        can_use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if can_use_gpu else "cpu")
        training_model = model.to(device)
        best_acc = 0.0
        best_loss = 0.0
        best_weights = PokemonAI.get_weights(training_model)

        # Start epoch history track
        train_acc_hist = np.zeros(epochs)
        train_loss_hist = np.zeros(epochs)
        eval_acc_hist = np.zeros(epochs)
        eval_loss_hist = np.zeros(epochs)

        # get folds for training and evaluating
        eval_labels = [classes.index(x.pokemon_type) for x in eval]
        train_labels = [classes.index(x.pokemon_type) for x in train]

        # transform labels to tensors
        eval_labels = [PokemonAI.get_expected_tensor(
            x, len(classes)) for x in eval_labels]
        train_labels = [PokemonAI.get_expected_tensor(
            x, len(classes)) for x in train_labels]

        optimization_fn = optimizer(
            training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        learning_rate_drop_fn = None
        if(learning_rate_drop_step_size > 0 and learning_rate_drop != 0):
            learning_rate_drop_fn = optim.lr_scheduler.StepLR(
                optimization_fn, step_size=learning_rate_drop_step_size, gamma=learning_rate_drop)

        progress = tqdm(range(epochs))
        for epoch in progress:
            # Training phase
            progress.set_description("Learning")

            training_model, accuracy, loss = PokemonAI.process_input(training_model, train, train_labels,
                                                                     error_fn, optimization_fn=optimization_fn, device=device, batch_size=batch_size, learn=True)

            # Save data to training history
            train_acc_hist[epoch] = accuracy
            train_loss_hist[epoch] = loss

            # Evaluation phase
            progress.set_description("Evaluating")
            _, accuracy, loss = PokemonAI.process_input(training_model, eval, eval_labels,
                                                        error_fn, optimization_fn=optimization_fn, device=device, batch_size=batch_size, learn=False)

            if accuracy > best_acc:
                best_acc = accuracy
                best_loss = loss
                best_weights = PokemonAI.get_weights(training_model)

            # Save data to eval history
            eval_acc_hist[epoch] = accuracy
            eval_loss_hist[epoch] = loss

            if(learning_rate_drop_fn != None):
                learning_rate_drop_fn.step()

        # Plot graphs
        if plot_acc:
            fig, ax = plt.subplots()
            ax.plot(train_acc_hist, 'r', label="Average training acurracy")
            ax.plot(eval_acc_hist, 'b', label="Average evaluation accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
            ax.legend()
            plt.show()

        if plot_loss:
            fig, ax = plt.subplots()
            ax.plot(train_loss_hist, 'r', label="Average training loss")
            ax.plot(eval_loss_hist, 'b', label="Average evaluation loss")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
            ax.legend()
            plt.show()

        # load best model weights
        training_model.load_state_dict(best_weights)

        return training_model, best_acc, best_loss, train_acc_hist, train_loss_hist, eval_acc_hist, eval_loss_hist

    @staticmethod
    def predict(model, data, classes, use_gpu=False, batch_size=0):
        can_use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if can_use_gpu else "cpu")
        model.to(device)
        model.eval()

        if batch_size > 0:
            batches = PokemonAI.create_batches(data, batch_size)
            predictions = []
            for idx, batch in enumerate(batches):
                batch_input = torch.stack([x.tensor() for x in batch])
                batch_output = model(batch_input.to(device))
                predictions += torch.max(batch_output, 1)[1]

                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()

            return [classes[int(x)] for x in predictions]
        else:
            data_input = torch.stack([x.tensor() for x in data])
            output = model(data_input.to(device))
            output.to(torch.device("cpu"))
            predictions = torch.max(output, 1)
            return [classes[int(x)] for x in predictions[1]]
