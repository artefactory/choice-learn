"""Data generation module for synthetic basket data."""

import random
from typing import Union

import numpy as np

from .dataset import Trip, TripDataset

#np.random.seed(42)
#random.seed(42)


class SyntheticDataGenerator:
    """Class to generate synthetic basket data based on predefined item sets and their relations."""

    def __init__(
        self,
        items_nest: dict = {
            0: ({0, 1, 2}, [-1, 1, 0, 0]),
            1: ({3, 4, 5}, [1, -1, 0, 0]),
            2: ({6}, [0, 0, -1, 0]),
            3: ({7}, [0, 0, 0, -1]),
        },
    ) -> None:
        """Initialize the data generator with parameters for basket generation.

        Parameters
        ----------
            items_nest : dict
                Dictionary defining item sets and their relations.
        """

        self.items_nest = items_nest
        self.instantiate(
            proba_complementary_items=0.7,
            proba_neutral_items=0.3,
            noise_proba=0.15,
            assortment_matrix=np.ones((1, 8)),
        )

        

    def instantiate(self, 
        proba_complementary_items: float,
        proba_neutral_items: float,
        noise_proba: float,
        assortment_matrix:  np.ndarray = np.ones((1, 8))) -> None:
        """Instantiate the data generator with parameters for basket generation.

        Parameters
        ----------
            proba_complementary_items : float
                Probability of adding complementary items to the basket.       
            proba_neutral_items : float
                Probability of adding neutral items to the basket.
            noise_proba : float
                Probability of adding noise items to the basket.
            assortment_matrix : np.ndarray, optional
                Matrix of assortments to use for generating baskets.
                If None, uses the default assortment matrix.   
        """

        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba
        self.assortment_matrix = assortment_matrix
        self.default_assortment = self.assortment_matrix[0,:]


    def get_assortment_items(self, assortment : Union[int, np.ndarray] = None) -> np.ndarray:
        """Return the assortment based on the provided index or array.

        Parameters
        ----------
            assortment : int or np.ndarray, optional
                Index of the assortment or an array representing the assortment.

        Returns
        -------
            np.ndarray
                The assortment corresponding to the provided index or array.
        """
        if isinstance(assortment, int) and self.assortment_matrix.shape[0] > assortment:
            assortment = np.array(
                [
                    i
                    for i in range(self.assortment_matrix.shape[1])
                    if self.assortment_matrix[assortment, i] == 1
                ]
            )
        elif isinstance(assortment, np.ndarray):

            assortment = np.array(
                [
                    i
                    for i in range(self.assortment_matrix.shape[1])
                    if assortment[i] == 1
                ]
            )
        else :
            assortment = np.array(
                [
                    i
                    for i in range(self.assortment_matrix.shape[1])
                    if self.assortment_matrix[0, i] == 1
                ]
            )
        return assortment

    def get_available_sets(self, assortment: Union[int, np.ndarray] = None) -> list:
        """Return the available sets based on the current assortment.

        Returns
        -------
            list
                List of keys from items_nest
                Where the first item set intersects with the current assortment.
        """
        assortment_items = set(self.get_assortment_items(assortment))


        return list(
            key
            for key, value in self.items_nest.items()
            if value[0].intersection(assortment_items)
        )

    def generate_basket(self,
                        assortment: Union[int, np.ndarray] = None,
                        len_basket : int = None) -> list:
        """Generate a basket of items based on the defined item sets and their relations.

        Parameters
        ----------
            assortment : int or np.ndarray, optional
                Index of the assortment or an array representing the assortment.
            len_basket : int, optional
                Length of the basket to be generated.
                If None, the basket length is determined by the available sets.

        Returns
        -------
            list
                List of items in the generated basket.
        """
        available_sets = self.get_available_sets(assortment)
        available_items = self.get_assortment_items(assortment)


        def select_first_item() -> tuple:
            """Select the first item and its nest randomly from the available sets.

            Returns
            -------
                tuple
                    A tuple containing the first item and its corresponding nest.
            """
            chosen_nest = random.choice(available_sets)
            #chosen_item = random.choice(list(self.items_nest[chosen_nest][0]))
            chosen_item = random.choice(
                [i for i in self.items_nest[chosen_nest][0] if i in available_items])

            return chosen_item, chosen_nest

        def complete_basket(first_item: int, first_nest: str) -> list:
            """Completes the basket by adding items based on the relations of the first item.

            Parameters
            ----------
                first_item : int
                    The first item to be added to the basket.
                first_nest : str
                    The nest corresponding to the first item.

            Returns
            -------
                list
                    list next basket items.
            """
            basket = [first_item]
            first_key_index = first_nest
            for key in available_sets:
                nest, relations = self.items_nest[key]
                if (
                    relations[first_key_index] == 1
                    and random.random() < self.proba_complementary_items
                ):
                    #basket.add(random.choice(list(nest)))
                    basket.append(random.choice(
                        [i for i in nest if (i not in basket) and (i in available_items)]
                        ))
                elif (
                    relations[first_key_index] == 0
                    and random.random() < self.proba_neutral_items
                ):
                    #basket.add(random.choice(list(nest)))
                    basket.append(random.choice(
                        [i for i in nest if (i not in basket) and (i in available_items)]
                        ))
            return basket

        def add_noise(basket: list) -> list:
            """Add noise items to the basket based on the defined noise probability.

            Parameters
            ----------
                basket : list
                    The current basket of items.

            Returns
            -------
                list
                    A list containing the items in the basket, potentially with noise items added.
            """
            if random.random() < self.noise_proba:
                try:
                    basket.append(int(random.choice(
                        [i for i in available_items if i not in basket]
                        )))
                except IndexError:
                    print(
                        "Warning: No more items available to add as noise. " \
                        "Returning the current basket."
                    )
            return basket
        if available_sets != []:
            first_chosen_item, first_chosen_nest = select_first_item()
            basket = complete_basket(first_chosen_item, first_chosen_nest)
            basket = add_noise(basket)
        else:
            basket = []

        if len_basket is not None:
            assert isinstance(len_basket, int), "len_basket should be an integer"
            if len(basket) < len_basket:
                basket = self.generate_basket(assortment, len_basket)
            else:
                basket = random.sample(basket, len_basket)


        return np.array(list(basket))

    def generate_trip(self, assortment: Union[int, np.ndarray] = None) -> Trip:
        """Generate a trip object from the generated basket.

        Returns
        -------
            Trip
                A Trip object containing the generated basket.
        """
        if assortment is None:
            assortment = self.default_assortment

        basket = self.generate_basket(assortment=assortment)
        return Trip(
            purchases=basket,
            prices=np.array([1.0] * 8),  # Assuming uniform price of 1.0 for simplicity
            assortment=np.array(list(self.assortment_matrix[0])),
        )

    def generate_trip_dataset(
        self, n_baskets=400, assortments_matrix: np.ndarray = None
    ) -> TripDataset:
        """Generate a TripDataset from the generated baskets.

        Parameters
        ----------
            n_baskets : int, optional
                Number of baskets to generate. If None, uses the default value.
            assortment_matrix : list of sets, optional
                Matrix of assortments to use for generating baskets.
                If None, uses the default assortment matrix.
                shape (n_assortments, n_items)

        Returns
        -------
            TripDataset
                A TripDataset object containing the generated baskets.
        """
        if assortments_matrix is None:
            assortments = self.assortment_matrix
        else:
            assortments = assortments_matrix

        if n_baskets is None:
            n_baskets = self.n_baskets_default

        trips = []
        for _ in range(n_baskets):
            assortment_id = random.randint(0, assortments.shape[0] - 1)
            trip = self.generate_trip(assortments[assortment_id])

            trips.append(trip)

        return TripDataset(trips, assortments)


