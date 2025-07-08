"""Data generation related stuff."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import trange


class SyntheticDataGenerator:
    def __init__(
        self,
        n_baskets_default : int = 400,
        proba_complementary_items: float = 0.7,
        proba_neutral_items: float = 0.3,
        noise_proba: float = 0.15,
        items_nest: dict = {
                0: ({0, 1, 2}, [-1, 1, 0, 0]),
                1: ({3, 4, 5}, [1, -1, 0, 0]),
                2: ({6}, [0, 0, -1, 0]),
                3: ({7}, [0, 0, 0, -1]),
            },
        assortment: set = {0, 1, 2, 3, 4, 5, 6, 7}, #argument d'entrée
    ) -> None:


        self.n_baskets_default = n_baskets_default

        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba
        
  
        self.items_nest = items_nest

        self.assortment = assortment
        self.available_sets = list( # Not sure what it is supposed to do
            key
            for key, value in self.items_nest.items()
            if value[0].intersection(self.assortment)
        )

    def generate_basket(self) -> list:
        """Generates a basket of items based on the defined item sets and their relations."""

        def select_first_item() -> tuple:
            """Selects the first item and its nest randomly from the available sets."""

            chosen_nest = random.choice(self.available_sets) # Why not use items_nest ?
            chosen_item = random.choice(list(self.items_nest[chosen_nest][0]))
            return chosen_item, chosen_nest

        def complete_basket(first_item: int, first_nest: str) -> list:
            """Completes the basket by adding items based on the relations of the first item."""

            basket = [first_item]
            first_key_index = first_nest
            for key in self.available_sets:
                nest, relations = self.items_nest[key]
                if (
                    relations[first_key_index] == 1 # At this point you may use "complementary" (e.g.) instead of an int to make it more understandable
                    and random.random() < self.proba_complementary_items
                ):
                    basket.append(random.choice(list(nest)))
                elif (
                    relations[first_key_index] == 0
                    and random.random() < self.proba_neutral_items
                ):
                    basket.append(random.choice(list(nest)))
            return basket

        def add_noise(basket: list) -> list:
            """Adds noise items to the basket based on the defined noise probability."""

            noise_proba = self.noise_proba
            for item in self.assortment:
                if item not in basket and random.random() < noise_proba:
                    basket.append(item)
            return basket

        first_chosen_item, first_chosen_nest = select_first_item()
        basket = complete_basket(first_chosen_item, first_chosen_nest)
        basket = add_noise(basket)

        return basket

    def generate_synthetic_dataset(self, n_baskets = None) -> list:
        """Generates a dataset of baskets."""


        if n_baskets is None:
            n_baskets = self.n_baskets_default

        baskets = []
        for _ in range(n_baskets):
            baskets.append(self.generate_basket())
        return baskets

DG = SyntheticDataGenerator()
baskets = DG.generate_synthetic_dataset(10)
print(baskets)
