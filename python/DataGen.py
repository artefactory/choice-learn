"""Data generation module for synthetic basket data."""

import numpy as np
import random

np.random.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    """ Class to generate synthetic basket data based on predefined item sets and their relations. """

    def __init__(
        self,
        n_baskets_default: int = 400,
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
        """Initializes the data generator with parameters for basket generation.

        Parameters
        ----------
            n_baskets_default : int
                Default number of baskets to generate if not specified.
            proba_complementary_items : float
                Probability of adding complementary items to the basket.
            proba_neutral_items : float
                Probability of adding neutral items to the basket.
            noise_proba : float
                Probability of adding noise items to the basket.
            items_nest : dict
                Dictionary defining item sets and their relations.
            default_assortment : set
                Default assortment of items available for basket generation.
        """

        self.n_baskets_default = n_baskets_default

        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba

        self.items_nest = items_nest

        self.assortment = default_assortment

    def get_available_sets(self) -> list:
        """ Returns the available sets based on the current assortment.
        
        Returns
        -------
            list
                List of keys from items_nest where the first item set intersects with the current assortment.
        """

        self.available_sets = list(  # Not sure what it is supposed to do
            key
            for key, value in self.items_nest.items()
            if value[0].intersection(self.assortment)
        )

    def generate_basket(self) -> list:
        """ Generates a basket of items based on the defined item sets and their relations.
        
        Returns
        -------
            list
                List of items in the generated basket.
        """

        def select_first_item() -> tuple:
            """ Selects the first item and its nest randomly from the available sets.
            
            Returns
            -------
                tuple
                    A tuple containing the first item and its corresponding nest.
            """

            chosen_nest = random.choice(self.available_sets) 
            chosen_item = random.choice(list(self.items_nest[chosen_nest][0]))
            return chosen_item, chosen_nest

        def complete_basket(first_item: int, first_nest: str) -> set:
            """ Completes the basket by adding items based on the relations of the first item.
            
            Parameters
            ----------
                first_item : int
                    The first item to be added to the basket.
                first_nest : str
                    The nest corresponding to the first item.  
            
            Returns
            -------
                set
                    A set containing the first item and potentially additional items based on their relations.
            """

            basket = {first_item}
            first_key_index = first_nest
            for key in self.available_sets:
                nest, relations = self.items_nest[key]
                if (
                    relations[first_key_index]
                    == 1 
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
            """ Adds noise items to the basket based on the defined noise probability.
            
            Parameters 
            ----------
                basket : set
                    The current basket of items.
            
            Returns
            -------
                list
                    A list containing the items in the basket, potentially with noise items added.
            """

            if random.random() < self.noise_proba:
                basket.add(random.choice(list(self.assortment.difference(basket))))

            return basket

        first_chosen_item, first_chosen_nest = select_first_item()
        basket = complete_basket(first_chosen_item, first_chosen_nest)
        basket = add_noise(basket)

        return list(basket)

    def generate_synthetic_dataset(self, n_baskets=None, assortment=None) -> list:
        """ Generates a dataset of baskets.
        
        Parameters  
        ----------
            n_baskets : int, optional
                Number of baskets to generate. If None, uses the default value.
            assortment : set, optional
                Custom assortment of items to use for basket generation. If None, uses the default assortment.
                
        Returns
        -------
            list or np.ndarray
                List of baskets or a padded numpy array of baskets if padded is True.
        """

        if assortment is not None:
            self.assortment = assortment

        self.get_available_sets()

        if n_baskets is None:
            n_baskets = self.n_baskets_default

        baskets = []
        for _ in range(n_baskets):
            baskets.append(self.generate_basket())


        return baskets
