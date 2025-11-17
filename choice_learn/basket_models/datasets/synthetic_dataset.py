"""Data generation module for synthetic basket data."""

import logging
from typing import Union

import numpy as np

from ..data import Trip, TripDataset


class SyntheticDataGenerator:
    """Class to generate synthetic basket data based on predefined item sets and their relations."""

    def __init__(
        self,
        items_nest: dict,
        nests_interactions: list,
        proba_complementary_items: float = 0.7,
        proba_neutral_items: float = 0.15,
        noise_proba: float = 0.05,
        plant_seed: int = None,
        user_profile: dict = None,
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
            user_profile = {0:{ "nest" : 0, "item" : 0}, 1: {"nest" : 0, "item" : 1}, 2: {"nest" : 0, "item" : 2}}
                Dictionary defining user profiles with preferred nest and item. Structure is:
                {user_id: {"nest": nest, "item": preferred_item}, ...}
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
        self.user_profile = user_profile

        if plant_seed is not None:
            np.random.seed(plant_seed)

    def get_available_sets(self, assortment_items: np.ndarray = None) -> np.ndarray:
        """Return the available nests based on the current assortment.

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
        return np.array(
            list(
                key
                for key, value in self.items_nest.items()
                if set(value).intersection(set(assortment_items))
            )
        )

    def select_first_item(
        self, available_sets: np.ndarray, available_items: np.ndarray, user_id: int
    ) -> tuple:
        """Select the first item and its nest randomly from the available sets.

        Parameters
        ----------
            available_sets : np.ndarray
                List of available sets from which to select the first item.
            available_items : np.ndarray
                List of available items in the current assortment.

        Returns
        -------
            tuple
                A tuple containing the first item and its corresponding nest.
        """
        chosen_nest = np.random.choice(available_sets)

        chosen_item = np.random.choice(
            [i for i in self.items_nest[chosen_nest] if i in available_items]
        )
        if self.user_profile is not None and self.user_profile[user_id]["nest"] == chosen_nest:
            if self.user_profile[user_id]["item"] in available_items and np.random.rand() < 0.7:
                chosen_item = self.user_profile[user_id]["item"]

        return chosen_item, chosen_nest

    def complete_basket(
        self, first_item: int, first_nest: int, available_items: np.ndarray, user_id: int
    ) -> list:
        """Completes the basket by adding items based on the relations of the first item.

        Parameters
        ----------
            first_item : int
                The first item to be added to the basket.
            first_nest : int
                The nest corresponding to the first item.
            available_items: np.ndarray
                Avaialbe item IDs

        Returns
        -------
            list
                list next basket items.
        """
        basket = [first_item]
        interactions = self.nests_interactions[first_nest]
        for nest_id, items in self.items_nest.items():
            if (
                interactions[nest_id] == "compl"
                and np.random.random() < self.proba_complementary_items
            ):
                try:
                    if (
                        self.user_profile is not None
                        and self.user_profile[user_id]["nest"] == nest_id
                    ):
                        if (
                            self.user_profile[user_id]["item"] in available_items
                            and np.random.rand() < 0.7
                        ):
                            basket.append(self.user_profile[user_id]["item"])
                        else:
                            basket.append(
                                np.random.choice([i for i in items if i in available_items])
                            )
                    else:
                        basket.append(np.random.choice([i for i in items if i in available_items]))
                except ValueError:
                    logging.warning(
                        f"Warning: No more complementary items available in nest {nest_id}"
                    )
                    pass
            elif (
                interactions[nest_id] == "neutral" and np.random.random() < self.proba_neutral_items
            ):
                try:
                    basket.append(np.random.choice([i for i in items if i in available_items]))
                except ValueError:
                    logging.warning(f"Warning: No more neutral items available in nest {nest_id}")
                    pass
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
        if np.random.rand() <= self.noise_proba:
            try:
                basket.append(np.random.choice([i for i in available_items if i not in basket]))
            except IndexError:
                logging.warning(
                    "Warning: No more items available to add as noise.Returning the current basket."
                )
            except ValueError:
                logging.warning(
                    "Warning: No more items available to add as noise.Returning the current basket."
                )
        return basket

    def generate_basket(
        self, assortment: Union[int, np.ndarray] = None, len_basket: int = None, user_id: int = None
    ) -> list:
        """Generate a basket of items based on the defined item sets and their relations.

        Parameters
        ----------
            assortment : np.ndarray, optional
                Index of the assortment or an array representing the assortment.
                1 represent sold items while 0 represnet missing items.
            len_basket : int, optional
                Length of the basket to be generated.
                If None, the basket length is determined by the available sets.

        Returns
        -------
            array
                array of items in the generated basket.
        """
        available_items = np.where(assortment > 0)[0]
        available_sets = self.get_available_sets(available_items)

        if len(available_sets) != 0:
            first_chosen_item, first_chosen_nest = self.select_first_item(
                available_sets=available_sets, available_items=available_items, user_id=user_id
            )
            basket = self.complete_basket(
                first_chosen_item,
                first_chosen_nest,
                available_items=available_items,
                user_id=user_id,
            )
            basket = self.add_noise(basket, available_items=available_items)
        else:
            basket = []

        if len_basket is not None:
            if not isinstance(len_basket, int) or len_basket < 1:
                raise TypeError("len_basket should be an integer larger than 0.")
            if len(basket) < len_basket:
                basket = self.generate_basket(assortment, len_basket, user_id=user_id)
            else:
                basket = np.random.choice(basket, len_basket, replace=False)

        return np.array(basket)

    def generate_trip(
        self, assortment: Union[int, np.ndarray] = None, len_basket: int = None
    ) -> Trip:
        """Generate a trip object from the generated basket.

        Parameters
        ----------
            assortment : int or np.ndarray
                Index of the assortment or an array representing the assortment.
            len_basket : int, optional
                Length of the basket to be generated.
                If None, the basket length is determined by the
                available sets.

        Returns
        -------
            Trip
                A Trip object containing the generated basket.
        """

        user_id = (
            np.random.randint(0, len(self.user_profile)) if self.user_profile is not None else None
        )
        basket = self.generate_basket(assortment, len_basket=len_basket, user_id=user_id).astype(
            int
        )
        return Trip(
            purchases=basket,
            # Assuming uniform price of 1.0 for simplicity
            prices=np.ones((1, len(assortment))),
            assortment=assortment,
            user_id=user_id,
        )

    def generate_trip_dataset(
        self, n_baskets: int = 400, assortments_matrix: np.ndarray = None, len_basket: int = None
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
            len_basket : int, optional
                Length of the basket to be generated.
                If None, the basket length is determined by the
                available sets.

        Returns
        -------
            TripDataset
                A TripDataset object containing the generated baskets.
        """
        trips = []
        assortments = []
        assortment_id = np.random.randint(0, len(assortments_matrix))
        for _ in range(n_baskets):
            trip = self.generate_trip(assortments_matrix[assortment_id], len_basket=len_basket)
            assortments.append(assortments_matrix[assortment_id])
            trips.append(trip)

        return TripDataset(trips, np.array(assortments_matrix))
