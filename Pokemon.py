import copy
from enum import Enum
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
    """Enumerable for Pokémon type"""
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

    def color(self) -> str:
        """Gets hexadecimal code related to Pokémon type"""
        type_color = {
            PokemonType.Grass:  "#78c850",
            PokemonType.Fire:  "#F08030",
            PokemonType.Water:  "#6890F0",
            PokemonType.Bug:  "#A8B820",
            PokemonType.Normal:  "#A8A878",
            PokemonType.Poison:  "#A040A0",
            PokemonType.Electric:  "#F8D030",
            PokemonType.Ground:  "#E0C068",
            PokemonType.Fairy:  "#EE99AC",
            PokemonType.Fighting:  "#C03028",
            PokemonType.Psychic:  "#F85888",
            PokemonType.Rock:  "#B8A035",
            PokemonType.Ghost:  "#705898",
            PokemonType.Ice:  "#98D8D8",
            PokemonType.Dragon:  "#7038f8",
            PokemonType.Flying:  "#66c3e8",
            PokemonType.Dark:  "#472e07",
            PokemonType.Steel:  "#828282",
        }
        return type_color[self]


class Pokemon:
    """
    A class to represent a Pokémon

    ...

    Attributes
    ----------
    name : str
        The name of the Pokémon
    pokedex_number: int
        Number of the Pokémon on the original Pokédex
    pokemon_type : PokemonType
        Enumerable for the main type of this Pokémon 
    type_1 : str
        Name of the main type of the Pokémon
    type_2 : str
        Name of the secondary type of the Pokémon
    is_legendary : bool
        Wether the Pokémon is a legendary or not  
    generation : int
        Number of the generation the Pokémon belongs
    img_variation : str
        If this pokemon entry represent a variantion of the original image, this holds the name for the variation  

    Methods
    ---
    base_image:
        Get base image for the Pokémon    
    image:
        Get image that represents this Pokémon, if it's a variant the return is the variant image
    histogram:
        Get Pokémon's image histogram
    tensor:
        Get tensor for Pokémon's image
    """

    def __init__(self, name: str, pokedex_number: int, type_1: str, type_2: str, is_legendary: bool, generation: int, img_variation: str = None) -> None:
        """
        Paramenters
        ---
        name : str
            The name of the Pokémon
        pokemon_type : PokemonType
            Enumerable for the main type of this Pokémon 
        type_1 : str
            Name of the main type of the Pokémon
        type_2 : str
            Name of the secondary type of the Pokémon
        is_legendary : bool
            Wether the Pokémon is a legendary or not  
        generation : int
            Number of the generation the Pokémon belongs
        img_variation : str, optional
            If this pokemon entry represent a variantion of the original image, this holds the name for the variation
        """
        self.name = name
        self.pokedex_number = pokedex_number
        self.type_1 = type_1
        self.type_2 = type_2
        self.is_legendary = is_legendary
        self.generation = generation
        self.img_variation = img_variation

    def __str__(self) -> str:
        if(self.img_variation != None):
            return f"#{self.pokedex_number} {self.name} ({self.img_variation}) - {self.pokemon_type}"
        return f"#{self.pokedex_number} {self.name} - {self.pokemon_type}"

    @property
    def pokemon_type(self) -> PokemonType:
        """Enumerable for the main type of this Pokémon """
        return PokemonType[self.type_1]

    def base_image(self) -> Image.Image:
        """
        Get base image for the Pokémon

        Returns
        ---
        PIL.Image.Image
            Base image for the Pokémon
        """
        return Image.open(f"pokemon/{self.pokedex_number}.png")

    def image(self) -> Image.Image:
        """
        Get image that represents this Pokémon, if it's a variant the return is the variant image

        Returns
        ---
        PIL.Image.Image
            Image for the Pokémon
        """
        if(self.img_variation != None):
            return Image.open(f"data_augmentation/{self.img_variation}/{self.pokedex_number}.png")
        return self.base_image()

    def histogram(self) -> list:
        """
        Get Pokémon's image histogram

        Returns
        ---
        list
            List with 768 countings, the histogram for the RGB channels of the Pokémon's image,
            the first 256 numbers are for the Red, the following 256 for the Green and the last
            256 for the Blue
        """
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

    def tensor(self, transform=[]) -> torch.Tensor:
        """
        Get tensor for Pokémon's image

        Parameters
        ---
        transform : list, optional
            List of transforms to apply to tensor (default is [])

        Returns
        ---
        torch.Tensor
            Pokémon's image as a Tensor
        """
        transform_fn = [transforms.ToTensor()]
        for t in transform:
            transform_fn.append(t)
        return transforms.Compose(transform_fn)(self.image().convert("RGB"))


class Pokedex:
    """
    Pokémon enciclopedia holding all data available for them

    ...

    Attributes
    ---
    pokemons : list
        List of all Pokémon available

    Methods
    ---
    get_pokemon:
        Find pokemon given a pokédex number
    get_type:
        Get all pokémon from a given type
    """

    def __init__(self, path) -> None:
        """
        Parameters
        ---
        path : str
            Path to csv file containing all the pokemon data 
        """
        data = pd.read_csv(path)
        self.pokemons: list[Pokemon] = []
        for _, row in data.iterrows():
            self.pokemons.append(
                Pokemon(row["name"], row["pokedex_number"], row["type_1"], row["type_2"], True if row["is_legendary"] == 1 else False, row["generation"]))

    def __len__(self):
        return len(self.pokemons)

    def get_pokemon(self, number: int) -> Pokemon:
        """
            Find pokemon given a pokédex number

            Parameters
            ---
            number : int
                Pokedex number to search for

            Returns
            ---
            Pokemon
                The pokemon with the given number
        """
        return next(x for x in self.pokemons if x.pokedex_number == number)

    def get_type(self, pokemon_type: PokemonType, with_legendary: bool = True, max_generation: int = 0) -> list:
        """
        Get all pokémon from a given type

        Parameters
        ---
        pokemon_type : PokemonType
            Type to filter the pokémon list
        with_legendary : bool, optional
            Whether or not to include legendary Pokémon on the listing (default is True)
        max_generation : int, optional
            Limit for the generations included on the list, 0 means any (default is 0)

        Returns
        ---
        list[Pokemon]
            List with all Pokemons matching the filter parameters
        ---
        """
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
    """
    Auxiliary class to support machine learning algorithms using `Pokemon` data

    ...

    Methods
    ---
    get_balanced_data
        Get balanced list of samples from a pokémon list given the types desired
    split_data
        Returns data splitted in training, evaluation and test groups
    create_folds
        Separate data into folds
    copy_weights
        Get a copy of the weights from a neural network model
    get_expected_tensor
        Transform label information into Tensor to be used in neural network training
    get_fold_groups
        Aggregate fold list into separate groups containing training data and evaluation data
        according to which is the current fold. 
    create_batches
        Separete data into chunks of a given size
    evaluate_model
        Evaluate neural network classification model based on the supplied data and error
        function
    process_input
        Process input data in batches using a neural network model
    train
        Trains a neural network
    k_fold_train
        Trains a neural network using k-fold cross validation
    predict_pokemons
        Predicts list of Pokémon using a trained neural network
    get_probs
        Gets prediction probabilities of a neural network for a list of inputs
    predict_image
        Gets the prediction for a given image using a neural network
    """

    _imagenet_normalization_fn = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def get_balanced_data(dex: Pokedex, types: list, variations: list = [], with_legendary: bool = True, max_generation: int = 0):
        """
        Get balanced list of samples from a pokémon list given the types desired

        Parameters
        ---
        dex : Pokedex
            A dex containing pokémon data
        types : list[PokemonType]
            List of types to include in the balance data
        variations: list[str], optional
            List with the names of variations to include in the balancing (default is [])
        with_legendary : bool, optional
            Whether or not to include legendary Pokémon on the listing (default is True)
        max_generation : int, optional
            Limit for the generations included on the list, 0 means any (default is 0)

        Returns
        ---
        list[Pokemon]
            Shuffled list with balanced samples of Pokémon from the given types including the possible variations
        """

        # get all base pokemon from classes
        entries = {}
        smallest_set_size = float("inf")
        for pkm_class in types:
            type_list = dex.get_type(
                pkm_class, with_legendary=with_legendary, max_generation=max_generation)

            sample_count = len(type_list)
            if sample_count < smallest_set_size:
                smallest_set_size = sample_count

            shuffle(type_list)
            entries[pkm_class] = type_list

        # check how many there could be from each type considering the smallest group
        data = []
        sample_count = smallest_set_size * (len(variations)+1)

        # add random augmentations to fill the available slots for each class to match the ideal size
        for pkm_class in types:
            available_slots = sample_count - len(entries[pkm_class])
            if available_slots > 0:
                augmentation = [Pokemon(pkm.name, pkm.pokedex_number, pkm.type_1, pkm.type_2, pkm.is_legendary,
                                        pkm.generation, var) for pkm in entries[pkm_class] for var in variations]
                shuffle(augmentation)
                entries[pkm_class] += augmentation[:available_slots]
                shuffle(entries[pkm_class])
            elif available_slots < 0:
                entries[pkm_class] = entries[pkm_class][:sample_count]

            data += entries[pkm_class]

        # shuffle list and return it
        shuffle(data)
        return data

    @staticmethod
    def split_data(data: list, train_prct: float, eval_prct: float):
        """
        Returns data splitted in training, evaluation and test groups

        Parameters
        ---
        data : list
            List of data to be split
        train_prct : float
            Percentage of data to be included into training set (0 to 1)
        eval_prct : float
            Percentage of data to be included into evaluation set (0 to 1)

        Returns
        ---
        tuple[list, list, list]
            The training, evaluation and test groups respectively 
        """
        sample_count = len(data)
        train_size = math.floor(sample_count * train_prct)
        eval_size = math.floor(sample_count * eval_prct)

        return data[:train_size], data[train_size:train_size + eval_size], data[train_size + eval_size:]

    @staticmethod
    def create_folds(data: list, fold_count: int):
        """
            Separate data into folds

            Parameters
            ---
            data : list
                Data to be split into folds
            fold_count : int
                Number of folds to split the data

            Returns
            ---
            list
                List with each fold (other lists) of data
        """

        # shuffle the list
        shuffled_data = sample(data, len(data))
        # get fold size
        fold_size = len(shuffled_data) // fold_count
        # separate in equally sized folds
        folds = [shuffled_data[i*fold_size:(i+1)*fold_size]
                 for i in range(fold_count)]
        # count left out data
        left_out = len(shuffled_data) % fold_count
        # distribute left out data into folds
        for idx in range(left_out):
            folds[idx].append(shuffled_data[-1*(idx+1)])

        return folds

    @staticmethod
    def copy_weights(model: nn.Module):
        """
            Get a copy of the weights from a neural network model

            Parameters
            ---
            model : torch.nn.Module
                Neural network from which the weights should be extracted

            Returns
            ---
            OrderedDict[str,Tensor]
                Weights for a neural network                
        """
        return copy.deepcopy(model.state_dict())

    @staticmethod
    def get_expected_tensor(idx: int, size: int):
        """
        Transform label information into Tensor to be used in neural network training

        Parameters
        ---
        idx : int
            Index for the expected value of class
        size : int
            Number of classes used in the problem to model the tensor after

        Returns
        ---
        Tensor
            Expected tensor matching the information of the label
        """
        aux = np.zeros(size)
        aux[idx] = 1.0
        return torch.tensor(aux, dtype=torch.float)

    @staticmethod
    def get_fold_groups(current_fold_index: int, all_folds: list):
        """
            Aggregate fold list into separate groups containing training data and evaluation data
            according to which is the current fold. 

            Parameters
            ---
            current_fold_index : int
                Index for the current fold (evaluation data)
            all_folds : list[list]
                List with all the separated folds

            Returns
            ---
            tuple[list,list]
                Respectively the evaluation and training groups of data
        """
        train = []
        for idx in range(len(all_folds)):
            if idx != current_fold_index:
                train += all_folds[idx]
        return all_folds[current_fold_index], train

    @staticmethod
    def create_batches(data: list, batch_size: int):
        """
            Separete data into chunks of a given size

            Parameters
            ---
            data : list
                List of data to be split into batches
            batch_size : int
                Expected size of the output batches

            Returns
            ---
            list[list]
                List with the splitted data into separated lists
        """
        batch_count = math.floor(len(data)/batch_size) + 1
        batches = []
        for batch_idx in range(batch_count):
            start_idx = batch_idx*batch_size
            end_idx = start_idx + batch_size
            batches.append(data[start_idx:end_idx])
        return [batch for batch in batches if len(batch) > 0]

    @staticmethod
    def evaluate_model(model: nn.Module, data: list, labels: list, error_fn, device: str):
        """
            Evaluate neural network classification model based on the supplied data and error function

            Parameters
            ---
            model : torch.nn.Module
                Neural network model to evaluate
            data : list
                List with evaluation data
            labels : list
                List of labels expected for the evaluation data (with matching order)
            error_fn  : torch.nn error function
                Error function to be applied on the evaluation
            device : str
                Device used to perform the computing (either cpu or cuda)

            Returns
            ---
             tuple[float, float, Tensor]
                Respectively the value for the evaluation loss, acuracy and 
                information for the error function to be used on the backpropagation
        """
        # get correct labels
        _, correct_labels = torch.max(labels.to(device), 1)

        # predict data
        output = model(data.to(device))
        # get error
        loss_output = error_fn(output, labels.to(device))

        # define predictions as a single value for the highest probability output
        _, predictions = torch.max(output, 1)

        # count the correct predictions
        correct = torch.sum(predictions == correct_labels)
        # calculate loss
        loss = loss_output.item() / len(data)
        # calculate model accuracy
        accuracy = float(correct) / len(data)

        return loss, accuracy, loss_output

    @staticmethod
    def process_input(model: nn.Module, data: torch.Tensor, labels: list, error_fn, optimization_fn: optim.Optimizer = None, device=torch.device("cpu"), batch_size: int = 25, learn=False, using_transfer_learning=False):
        """
            Process input data in batches using a neural network model

            Parameters
            ---
            model : torch.nn.Module
                Neural network model to evaluate
            data : list
                List with input data
            labels : list
                List of labels expected for the data (with matching order)
            error_fn  : torch.nn error function
                Error function to be applied on the processing
            optimization_fn : torch.optim.Optimizer, optional
                Optimization function used on training to realize the backpropagation.
                Only required if training while processing
            device : torch.device
                Device used to perform the computing (default is cpu).
            batch_size : int, optional
                Size of the batches processed (default is 25)
            learn : bool, optional
                Whether or not to train the model while processing (default is False)
            using_transfer_learning : bool
                Whether or not the model is use/used transfer learning for the training process (default is False)

            Returns
            ---
             tuple[float, float, Tensor]
                Respectively the value for the evaluation loss, acuracy and 
                information for the error function to be used on the backpropagation
        """

        # define processing mode on the neural network
        if learn:
            model.train()
        else:
            model.eval()

        # split input into batches to avoid memory overload
        data_input = PokemonAI.create_batches(data, batch_size)
        data_labels = PokemonAI.create_batches(labels, batch_size)
        batch_count = len(data_input)
        loss = 0
        accuracy = 0

        # get correct normalizations if using transfer learning
        transform_fn = [
            PokemonAI._imagenet_normalization_fn] if using_transfer_learning else []

        # process batches
        for batch_idx in range(batch_count):

            # create stacked input from data
            batch_input = torch.stack([pkm.tensor(transform_fn)
                                      for pkm in data_input[batch_idx]])
            batch_labels = torch.stack(data_labels[batch_idx])

            if learn:
                # if learning, clear the gradient to get a brand new adjustment
                # with clean information
                optimization_fn.zero_grad()

            # evaluate model with input data
            batch_loss, batch_accuracy, loss_info = PokemonAI.evaluate_model(
                model, batch_input, batch_labels, error_fn=error_fn, device=device)

            # hold loss and accuracy values to summarize later
            accuracy += batch_accuracy
            loss += batch_loss

            if learn:
                # if learning, apply the adjustment with backpropagation
                loss_info.backward()
                optimization_fn.step()

            if device == torch.device("cuda:0"):
                # if using gpu, clear the cache of used data to save up on
                # memory use
                torch.cuda.empty_cache()

        return model, accuracy/batch_count, loss/batch_count

    @staticmethod
    def train(model: nn.Module, train: list, eval: list, types: list, epochs: int,
              optimizer=optim.SGD,
              weight_decay: float = 0,
              learning_rate: float = 0.1,
              error_fn=nn.MSELoss(),
              plot_acc=True,
              plot_loss=True,
              use_gpu=False,
              batch_size=25,
              learning_rate_drop=0,
              learning_rate_drop_step_size=0,
              using_transfer_learning=False):
        """
        Trains a neural network

        Parameters
        ---
        model : torch.nn.Module
            Neural network model to train
        train : list[Pokemon]
            List of Pokemon used for the training process
        eval : list[Pokemon]
            List of Pokemon used for the evaluation process
        types : list[PokemonType]
            List of PokemonType contained on the set
        epochs : int
            Number of iterations for the training
        optimizer : torch.optim.Optimizer, optional
            Function used to calculate the backpropagation on training phase (default
            is SGD, also known as stochastic gradient descent)
        weight_decay : float, option
            Decay multiplier to be applied to weight adjustments on the optimization. 
            If 0, no decay will be considered (default is 0)
        learning_rate : float, optional
            Learning rate used to calculate the adjustments on backpropagation (default
            is 0.1)
        error_fn : torch.nn error function, optional
            Function used to obtain the error parameters on the processing, used to
            acquire the data necessary for the adjustments (default is MSE, also known
            as mean square error).
        plot_acc : bool, optional
            Whether or not to plot the accuracy chart after the training (default is True)
        plot_loss : bool, optional
            Whether or not to plot the loss chart after the training (default is True)
        use_gpu : bool, optional
            Whether or not to use the GPU, if available, on training (default is False)
        learning_rate_drop: float, optional
            Multiplier to reduce learning rate afte a given number of epochs. If 0 there
            will not be a drop on the learning rate (default is 0).
        learning_rate_drop_step_size : int, optional 
            Number of steps between drops on the learning rate. If 0 there will not be a
                drop on the learning rate (default is 0).
        using_transfer_learning : bool, optional
                Whether or not the model is using transfer learning for the training process (default is False)

        Returns
        ---
        tuple[torch.Module, float, float, list, list, list, list]
            Respectively the trained neural network, it's accuracy, it's loss, it's 
            history of accuracy on training, it's history of loss on training, it's 
            history of accuracy on evaluation, it's history of loss on evaluation

        """

        # setup environment usage
        can_use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if can_use_gpu else "cpu")
        training_model = model.to(device)

        # start best weights as empty
        best_acc = 0.0
        best_loss = 0.0
        best_weights = PokemonAI.copy_weights(training_model)

        # Start epoch history track
        train_acc_hist = np.zeros(epochs)
        train_loss_hist = np.zeros(epochs)
        eval_acc_hist = np.zeros(epochs)
        eval_loss_hist = np.zeros(epochs)

        # get folds for training and evaluating
        eval_labels = [types.index(x.pokemon_type) for x in eval]
        train_labels = [types.index(x.pokemon_type) for x in train]

        # transform labels to tensors
        eval_labels = [PokemonAI.get_expected_tensor(
            x, len(types)) for x in eval_labels]
        train_labels = [PokemonAI.get_expected_tensor(
            x, len(types)) for x in train_labels]

        # define optimization function
        optimization_fn = optimizer(
            training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # define decay on learning rate if any is required
        learning_rate_drop_fn = None
        if(learning_rate_drop_step_size > 0 and learning_rate_drop != 0):
            learning_rate_drop_fn = optim.lr_scheduler.StepLR(
                optimization_fn, step_size=learning_rate_drop_step_size, gamma=learning_rate_drop)

        progress = tqdm(range(epochs))
        for epoch in progress:
            # Training phase
            progress.set_description("Learning")

            training_model, accuracy, loss = PokemonAI.process_input(training_model, train, train_labels,
                                                                     error_fn, optimization_fn=optimization_fn, device=device, batch_size=batch_size, learn=True, using_transfer_learning=using_transfer_learning)

            # Save data to training history
            train_acc_hist[epoch] = accuracy
            train_loss_hist[epoch] = loss

            # Evaluation phase
            progress.set_description("Evaluating")
            _, accuracy, loss = PokemonAI.process_input(training_model, eval, eval_labels,
                                                        error_fn, optimization_fn=optimization_fn, device=device, batch_size=batch_size, learn=False, using_transfer_learning=using_transfer_learning)

            if accuracy > best_acc:
                best_acc = accuracy
                best_loss = loss
                best_weights = PokemonAI.copy_weights(training_model)

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
    def k_fold_training(model: nn.Module, input_data: list, types: list, epochs: int,
                        k: int = 3,
                        optimizer=optim.SGD,
                        weight_decay: float = 0,
                        learning_rate: float = 0.1,
                        error_fn=nn.MSELoss(),
                        plot_acc=True,
                        plot_loss=True,
                        batch_size=25,
                        use_gpu=False,
                        learning_rate_drop=0,
                        learning_rate_drop_step_size=0,
                        using_transfer_learning=False):
        """
        Trains a neural network using k-fold cross validation

        Parameters
        ---
        model : torch.nn.Module
            Neural network model to train
        train : list[Pokemon]
            List of Pokemon used for the training process
        eval : list[Pokemon]
            List of Pokemon used for the evaluation process
        types : list[PokemonType]
            List of PokemonType contained on the set
        epochs : int
            Number of iterations for the training
        k : int, optional
            Number of folds used on cross validation (default is 3)
        optimizer : torch.optim.Optimizer, optional
            Function used to calculate the backpropagation on training phase (default
            is SGD, also known as stochastic gradient descent)
        weight_decay : float, option
            Decay multiplier to be applied to weight adjustments on the optimization. 
            If 0, no decay will be considered (default is 0)
        learning_rate : float, optional
            Learning rate used to calculate the adjustments on backpropagation (default
            is 0.1)
        error_fn : torch.nn error function, optional
            Function used to obtain the error parameters on the processing, used to
            acquire the data necessary for the adjustments (default is MSE, also known
            as mean square error).
        plot_acc : bool, optional
            Whether or not to plot the accuracy chart after the training (default is True)
        plot_loss : bool, optional
            Whether or not to plot the loss chart after the training (default is True)
        use_gpu : bool, optional
            Whether or not to use the GPU, if available, on training (default is False)
        learning_rate_drop: float, optional
            Multiplier to reduce learning rate afte a given number of epochs. If 0 there
            will not be a drop on the learning rate (default is 0).
        learning_rate_drop_step_size : int, optional 
            Number of steps between drops on the learning rate. If 0 there will not be a
                drop on the learning rate (default is 0).
        using_transfer_learning : bool, optional
                Whether or not the model is using transfer learning for the training process (default is False)

        Returns
        ---
        torch.Module
            The best trained neural network from all the folds
        """

        # save inital state to recover between folds
        untrained = PokemonAI.copy_weights(model)

        # setup initial best
        best_acc = 0.0
        best_weights = PokemonAI.copy_weights(model)

        # Start epoch history track
        train_acc_hist = []
        train_loss_hist = []
        eval_acc_hist = []
        eval_loss_hist = []

        # create folds from data
        folds = PokemonAI.create_folds(input_data, k)

        print(f"Got {len(folds)} folds with {len(folds[0])} samples")

        # train each fold
        for current_fold_index in range(len(folds)):
            # get training and evaluating data from fold list
            eval, train = PokemonAI.get_fold_groups(current_fold_index, folds)

            # train fold
            fold_best_model, fold_best_acc, fold_best_loss, fold_train_acc, fold_train_loss, fold_eval_acc, fold_eval_loss = PokemonAI.train(
                model=model,
                train=train,
                eval=eval,
                types=types,
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
                learning_rate_drop_step_size=learning_rate_drop_step_size,
                using_transfer_learning=using_transfer_learning)

            # if is better than the previous, store it
            if fold_best_acc > best_acc:
                best_acc = fold_best_acc
                best_weights = PokemonAI.copy_weights(fold_best_model)

            # keep track of the performance
            train_acc_hist.append(fold_train_acc)
            train_loss_hist.append(fold_train_loss)
            eval_acc_hist.append(fold_eval_acc)
            eval_loss_hist.append(fold_eval_loss)

            # reset model weights
            model.load_state_dict(untrained)

        # summarize the performance
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
    def predict_pokemons(model: nn.Module, data: list, types: list, use_gpu=False, batch_size=0, using_transfer_learning=False):
        """
            Predicts list of Pokémon using a trained neural network

            Parameters
            ---
            model : torch.nn.Module
                A neural network
            data : list[Pokemon]
                List of Pokemon to predict
            types : list[PokemonType]
                List of possible pokémon types expected on the prediction 
                (same order as used in training)
            use_gpu : bool, optional
                Whether or not to use the GPU to process the images (default is False)
            batch_size : int, optional
                Size of batches to divide the input into. If 0, the input will not be
                divided (default is 0)
            using_transfer_learning : bool, optional
                Whether or not the model used transfer learning for the training process (default is False)

            Returns
            ---
            list[int]
                List of the indexes of the highest probabilities on the predictions
        """
        can_use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if can_use_gpu else "cpu")
        model.to(device)
        model.eval()

        transform_fn = [
            PokemonAI._imagenet_normalization_fn] if using_transfer_learning else []

        if batch_size > 0:
            batches = PokemonAI.create_batches(data, batch_size)
            predictions = []
            for _, batch in enumerate(batches):
                batch_input = torch.stack(
                    [x.tensor(transform_fn) for x in batch])
                batch_output = model(batch_input.to(device))
                predictions += torch.max(batch_output, 1)[1]

                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()

            return [types[int(x)] for x in predictions]
        else:
            data_input = torch.stack([x.tensor(transform_fn) for x in data])
            output = model(data_input.to(device))
            output.to(torch.device("cpu"))
            predictions = torch.max(output, 1)
            return [types[int(x)] for x in predictions[1]]

    @staticmethod
    def get_probs(model: nn.Module, data: list, device: torch.device) -> list:
        """
        Gets prediction probabilities of a neural network for a list of inputs

        Parameters
        ---
        model : torch.nn.Module
            Neural network model to evaluate
        data : list
            List with input data
        device : torch.device
            Device used to perform the computing.

        Returns
        ---
        list
            List of probabilities for the predictions to the inputs
        """

        # stack input
        data_input = torch.stack(data)

        # get predictions
        output = model(data_input.to(device))

        # send variables to cpu to avoid unaccesibility when dealing with values
        output.to(torch.device("cpu"))

        # return probabilities (predictions normalized to be between 0 and 1)
        return nn.functional.softmax(output, dim=1)

    @staticmethod
    def predict_img(path: str, model: nn.Module, types: list, use_gpu=False, model_use_transfer_learning=False):
        """
        Gets the prediction for a given image using a neural network

        Parameters
        ---
        path : str
            Path to the input image
        model : torch.nn.Module
            A neural network
        types : list[PokemonType]
            List of possible pokémon types expected on the prediction 
            (same order as used in training)
        use_gpu : bool, optional
            Whether or not to use the GPU to process the images (default is False)
        using_transfer_learning : bool, optional
            Whether or not the model used transfer learning for the training process (default is False)

        """
        # setup environment
        can_use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device("cuda:0" if can_use_gpu else "cpu")

        # get input
        img = Image.open(path).convert("RGB")
        transform_fn = [transforms.ToTensor()]
        if model_use_transfer_learning:
            transform_fn.append(PokemonAI._imagenet_normalization_fn)
        data = transforms.Compose(transform_fn)(img)

        # get probabilities
        probs = PokemonAI.get_probs(model, [data], device)

        # match type with probability
        types_probs = [(types[i], float(probs[0][i]) * 100)
                       for i in range(len(types))]

        # get hights probability to define prediction and confidence
        highest_prob_idx = int(torch.max(probs, 1)[1])
        prediction = types[highest_prob_idx]
        confidence = types_probs[highest_prob_idx][1]

        return prediction, confidence, types_probs, data
