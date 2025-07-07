"""Data generation related stuff."""

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import trange


class SyntheticDataGenerator:
    def __init__(
        self,
        n_items: int = 8, # Redundant with items_nest ?
        # min_items_basket: int = 2, # Isn't part of the data generation algorithm
        n_baskets: int = 400, # The algo should generate baskets, the train / test split is done afterwards
        proba_complementary_items: float = 0.7,
        proba_neutral_items: float = 0.3,
        noise_proba: float = 0.15,
        items_nest: dict = None, # Redundant with n_items ? # Also I changed the name for a more informative one
    ) -> None:

        self.n_items = n_items

        self.n_baskets = n_baskets

        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba
        
        # Either this dictionnary is the default one and therefore it should be in the function signature
        # But here it is a bit "hidden"
        self.items_nest = (
            items_nest
            if items_nest
            else {
                0: ({0, 1, 2}, [-1, 1, 0, 0]),
                1: ({3, 4, 5}, [1, -1, 0, 0]),
                2: ({6}, [0, 0, -1, 0]),
                3: ({7}, [0, 0, 0, -1]),
            }
        )

        self.assortment = {0, 1, 2, 3, 4, 5, 6, 7} # Should be a function of items_nest
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

        def complete_basket(first_item: int, first_nest: str) -> set:
            """Completes the basket by adding items based on the relations of the first item."""

            basket = {first_item} # Why a set and not a list ?
            first_key_index = first_nest
            for key in self.available_sets:
                nest, relations = self.items_nest[key]
                if (
                    relations[first_key_index] == 1 # At this point you may use "complementary" (e.g.) instead of an int to make it more understandable
                    and random.random() < self.proba_complementary_items
                ):
                    basket.add(random.choice(list(nest)))
                elif (
                    relations[first_key_index] == 0
                    and random.random() < self.proba_neutral_items
                ):
                    basket.add(random.choice(list(nest)))
            return basket

        def add_noise(basket: set) -> list:
            """Adds noise items to the basket based on the defined noise probability."""

            noise_proba = self.noise_proba
            possible_noise = []
            for item in self.assortment:
                if item not in basket:
                    possible_noise.append(item)
            if len(possible_noise) > 0: # Not sure why there are two loops
                for item in possible_noise:
                    if random.random() < noise_proba:
                        basket.add(item)
                return list(basket)
            return [] # Should return basket ?

        first_chosen_item, first_chosen_nest = select_first_item()
        basket = complete_basket(first_chosen_item, first_chosen_nest)
        basket = add_noise(basket)

        return basket

    def generate_synthetic_dataset(self, n_baskets = None) -> list:
        """Generates a dataset of baskets."""

        # batch_size = self.n_baskets_training # ? why should that be the case ? Why is there a batch_size attribute then ?
        # baskets = []
        # count = 0
        # while count < batch_size:
        #     basket = self.generate_basket()
        #     if len(basket) >= self.min_items_basket:
        #         baskets.append(basket)
        #         count += 1
        
        # Should more be something like:
        if n_baskets is None:
            n_baskets = self.n_baskets

        baskets = []
        for _ in range(n_baskets):
            baskets.append(self.generate_basket())
        return baskets

    # Should not be in a data generator

    # def get_batches(self, dataset: list) -> list:
    #     """Generates batches of baskets for training or testing."""

    #     indices = list(range(len(dataset)))
    #     random.shuffle(indices)
    #     for i in range(0, len(indices), self.batch_size):
    #         batch_indices = indices[i : i + self.batch_size]
    #         yield [dataset[j] for j in batch_indices]

