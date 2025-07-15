"""Data generation related stuff."""
import os

import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)

# sets -> list or ndarray (Not so important right now, will change it later and compare runtimes)



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
        default_assortment: set = {0, 1, 2, 3, 4, 5, 6, 7},
    ) -> None:


        self.n_baskets_default = n_baskets_default

        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba
        
  
        self.items_nest = items_nest

        self.assortment = default_assortment
        

    def get_available_sets(self) -> list:
        """Returns the available sets based on the current assortment."""

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

            basket = {first_item}
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

            if random.random() < self.noise_proba:
                basket.add(random.choice(list(self.assortment.difference(basket))))


            return basket

        first_chosen_item, first_chosen_nest = select_first_item()
        basket = complete_basket(first_chosen_item, first_chosen_nest)
        basket = add_noise(basket)

        return list(basket)

    def generate_synthetic_dataset(self, n_baskets = None, assortment = None,  padded = False):
        """Generates a dataset of baskets."""

        if assortment is not None:
            self.assortment = assortment

        self.get_available_sets()

        if n_baskets is None:
            n_baskets = self.n_baskets_default

        baskets = []
        for _ in range(n_baskets):
            baskets.append(self.generate_basket())

        if padded:
            max_len = max(len(row) for row in baskets)
            return np.array([row + [0]*(max_len - len(row)) for row in baskets])
            
        return baskets
    
    
