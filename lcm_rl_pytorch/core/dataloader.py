import torch
from torch.utils.data import Dataset
from random import choices
import inflect
from importlib import resources

IE = inflect.engine()
datasets = resources.files("lcm_rl_pytorch.datasets")


class SimpleAnimals:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size
        with open(datasets.joinpath("simple_animals.txt"), "r") as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def sample(self):
        return choices(self.data, k=self.batch_size)

class ComplexAnimals:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size
        with open(datasets.joinpath("complex_animals.txt"), "r") as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def sample(self):
        return choices(self.data, k=self.batch_size)

class AnimalsWithActions:
    def __init__(self, batch_size):
        self.animals = []
        self.actions = []

        with open(datasets.join_path("simple_animals.txt"), "r") as f:
            for line in f:
                self.animals.append(line.strip())
        with open(datasets.join_path("actions.txt"), "r") as f:
            for line in f:
                self.actions.append(line.strip())

        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.animals) * len(self.actions)
    
    def sample(self):
        return [ f"{IE.a(animal)} {action}" for animal, action in zip(choices(self.animals, k=self.batch_size), choices(self.actions, k=self.batch_size))]
    
