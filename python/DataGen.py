"""Data generation module for synthetic basket data."""

import numpy as np
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../choice-learn"))
)
from choice_learn.basket_models.dataset import Trip, TripDataset
from typing import Union

np.random.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    """Class to generate synthetic basket data based on predefined item sets and their relations."""

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
        assortment_matrix: np.ndarray = np.ones((1, 8), dtype=int),
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

        self.assortment_matrix = assortment_matrix

    def get_available_sets(self, assortment: Union[int, np.ndarray] = None) -> list:
        """Returns the available sets based on the current assortment.

        Returns
        -------
            list
                List of keys from items_nest where the first item set intersects with the current assortment.
        """

        return list(
            key
            for key, value in self.items_nest.items()
            if value[0].intersection(assortment)
        )

    def generate_basket(self, assortment: Union[int, np.ndarray] = None) -> list:
        """Generates a basket of items based on the defined item sets and their relations.

        Returns
        -------
            list
                List of items in the generated basket.
        """
        if assortment is None:
            assortment = np.array(
                [
                    i
                    for i in range(self.assortment_matrix.shape[1])
                    if self.assortment_matrix[0, i] == 1
                ]
            )
        if isinstance(assortment, int) and self.assortment_matrix.shape[0] > assortment:
            assortment = np.array(
                [
                    i
                    for i in range(self.assortment_matrix.shape[1])
                    if self.assortment_matrix[assortment, i] == 1
                ]
            )
        elif isinstance(assortment, np.ndarray):
            assert assortment.shape == (8,), "Assortment should be a single row array."
            assortment = np.array(assortment)

        assortment = set(assortment)
        available_sets = self.get_available_sets(assortment)

        def select_first_item() -> tuple:
            """Selects the first item and its nest randomly from the available sets.

            Returns
            -------
                tuple
                    A tuple containing the first item and its corresponding nest.
            """

            chosen_nest = random.choice(available_sets)
            chosen_item = random.choice(list(self.items_nest[chosen_nest][0]))
            return chosen_item, chosen_nest

        def complete_basket(first_item: int, first_nest: str) -> set:
            """Completes the basket by adding items based on the relations of the first item.

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
            for key in available_sets:
                nest, relations = self.items_nest[key]
                if (
                    relations[first_key_index] == 1
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
            """Adds noise items to the basket based on the defined noise probability.

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
                basket.add(random.choice(list(assortment.difference(basket))))
            return basket

        first_chosen_item, first_chosen_nest = select_first_item()
        basket = complete_basket(first_chosen_item, first_chosen_nest)
        basket = add_noise(basket)

        return np.array(list(basket))

    def generate_trip(self, assortment: Union[int, np.ndarray] = None) -> Trip:
        """Generates a trip object from the generated basket.

        Returns
        -------
            Trip
                A Trip object containing the generated basket.
        """

        basket = self.generate_basket()
        return Trip(
            purchases=basket,
            prices=np.array([1.0] * 8),  # Assuming uniform price of 1.0 for simplicity
            assortment=np.array(list(self.assortment_matrix[0])),
        )

    def generate_trip_dataset(
        self, n_baskets=None, assortments_matrix: np.ndarray = None
    ) -> TripDataset:
        """Generates a TripDataset from the generated baskets.

        Parameters
        ----------
            n_baskets : int, optional
                Number of baskets to generate. If None, uses the default value.
            assortments : list of sets, optional
                List of assortments to use for basket generation. If None, uses the default assortment.

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


"""
sdg = SyntheticDataGenerator()
trip_dataset = sdg.generate_trip_dataset(n_baskets=10000)
print(f"Generated {len(trip_dataset.trips)} trips with assortments  {trip_dataset.available_items}")
distribution = np.zeros((8,8))
for trip in trip_dataset.trips:
    for i in trip.purchases:
        for j in trip.purchases:
            if i != j:
                distribution[i, j] += 1

distribution = distribution / np.sum(distribution, axis=1, keepdims=True)
plt.imshow(distribution, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Item Co-occurrence Distribution")
plt.xlabel("Item Index")
plt.ylabel("Item Index")
plt.savefig("item_cooccurrence_distribution.png")
plt.show()
"""
