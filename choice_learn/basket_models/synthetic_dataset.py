"""Data generation module for synthetic basket data."""

from typing import Union

import numpy as np

from .dataset import Trip, TripDataset


class SyntheticDataGenerator:
    """Class to generate synthetic basket data based on predefined item sets and their relations."""

    def __init__(
        self,
        items_nest: dict,
        nests_interactions: list,
        proba_complementary_items: float = 0.7,
        proba_neutral_items: float = 0.15,
        noise_proba: float = 0.05,
    ) -> None:
        """Initialize the data generator with parameters for basket generation.

        Parameters
        ----------
            items_nest : dict
                Dictionary defining item sets and their relations.
                Key should be next index and values list of items indexes, e.g.
                items_nests = {0:[0, 1, 2],
                               1: [3, 4, 5],
                               2: [6],
                               3: [7]}
            nests_interactions: list
                List of interactions between nests for each nest. Symmetry should
                be ensure by users, e.g.
                nests_interactions = [["", "compl", "neutral", "neutral"],
                                      ["compl", "", "neutral", "neutral"],
                                      ["neutral", "neutral", "", "neutral"],
                                      ["neutral", "neutral", "neutral", ""]]
            proba_complementary_items : float
                Probability of adding complementary items to the basket.
            proba_neutral_items : float
                Probability of adding neutral items to the basket.
            noise_proba : float
                Probability of adding noise items to the basket.
        """
        self.proba_complementary_items = proba_complementary_items
        self.proba_neutral_items = proba_neutral_items
        self.noise_proba = noise_proba

        self.items_nest = items_nest
        self.nests_interactions = nests_interactions

    def get_available_sets(self, assortment: Union[int, np.ndarray] = None) -> np.ndarray:
        """Return the available sets based on the current assortment.

        Parameters
        ----------
            assortment : int or np.ndarray, optional
                Index of the assortment or an array representing the assortment.

        Returns
        -------
            np.ndarray
                List of keys from items_nest
                Where the first item set intersects with the current assortment.
        """
        assortment_items = set(self.get_assortment_items(assortment))

        return np.array(
            list(
                key
                for key, value in self.items_nest.items()
                if len(set(value).intersection(assortment_items)) > 0
            )
        )

    def select_first_item(self, available_sets, available_items) -> tuple:
        """Select the first item and its nest randomly from the available sets.

        Returns
        -------
            tuple
                A tuple containing the first item and its corresponding nest.
        """
        chosen_nest = np.random.choice(available_sets)
        chosen_item = np.random.choice(
            np.array([i for i in self.items_nest[chosen_nest][0] if i in available_items])
        )
        return chosen_item, chosen_nest

    def complete_basket(self, first_item: int, first_nest: str, available_items) -> list:
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
        relations = self.nests_interactions[first_nest]
        for nest_id, items in self.items_nest.items():
            if (
                relations[nest_id] == "compl"
                and np.random.random() < self.proba_complementary_items
            ):
                basket.append(np.random.choice([i for i in items if i in available_items]))
            elif relations[nest_id] == "neutral" and np.random.random() < self.proba_neutral_items:
                basket.append(np.random.choice([i for i in items if i in available_items]))
        return basket

    def add_noise(self, basket: list, available_items) -> list:
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
        if np.random.rand() < self.noise_proba:
            try:
                basket.append(
                    int(np.random.choice([i for i in available_items if i not in basket]))
                )
            except IndexError:
                print(
                    "Warning: No more items available to add as noise. "
                    "Returning the current basket."
                )
        return basket

    def generate_basket(self, assortment: np.ndarray = None, len_basket: int = None) -> list:
        """Generate a basket of items based on the defined item sets and their relations.

        Parameters
        ----------
            assortment : np.ndarray, optional
                Index of the assortment or an array representing the assortment.
            len_basket : int, optional
                Length of the basket to be generated.
                If None, the basket length is determined by the available sets.

        Returns
        -------
            array
                array of items in the generated basket.
        """
        available_sets = self.get_available_sets(assortment)
        available_items = np.where(assortment > 0)

        if len(available_sets) != 0:
            first_chosen_item, first_chosen_nest = self.select_first_item(
                available_sets=available_sets, available_items=available_items
            )
            basket = self.complete_basket(
                first_chosen_item, first_chosen_nest, available_items=available_items
            )
            basket = self.add_noise(basket, available_items=available_items)
        else:
            basket = []

        if len_basket is not None:
            if not isinstance(len_basket, int):
                raise TypeError("len_basket should be an integer")
            if len(basket) < len_basket:
                basket = self.generate_basket(assortment, len_basket)
            else:
                basket = np.random.choice(basket, len_basket, replace=False)

        return np.array(basket)

    def generate_trip(self, assortment: Union[int, np.ndarray] = None) -> Trip:
        """Generate a trip object from the generated basket.

        Parameters
        ----------
            assortment : int or np.ndarray, optional
                Index of the assortment or an array representing the assortment.

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
            # Assuming uniform price of 1.0 for simplicity
            prices=np.ones((1, self.n_items)),
            assortment=assortment,
        )

    def generate_trip_dataset(
        self, n_baskets: int = 400, assortments_matrix: np.ndarray = None
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
        trips = []
        assortment_id = np.random.randint(0, assortments.shape[0])
        for _ in range(n_baskets):
            trip = self.generate_trip(assortments[assortment_id])
            trips.append(trip)

        trip_dataset_assortments = assortments * n_baskets

        return TripDataset(trips, trip_dataset_assortments)
