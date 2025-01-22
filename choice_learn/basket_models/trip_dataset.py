"""Classes to handle datasets with baskets of products."""

import random
from typing import Union

import numpy as np

from .utils.permutation import permutations


class Trip:
    """Class for a trip.

    A trip is a sequence of purchases made by a given customer at a specific time and
    at a specific location (with given prices).
    It can be seen as the content of a time-stamped purchase receipt with customer identification.

    Trip = (purchases, customer, week, prices)

    """

    def __init__(
        self,
        id: int,
        purchases: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
        assortment: int,
    ) -> None:
        """Initialize the trip.

        Parameters
        ----------
        id: int
            Trip ID
        purchases: np.ndarray
            List of the ID of the purchased items, 0 to n_items - 1 (0-indexed)
            Shape: (len_basket,), the last item is the checkout item 0
        customer: int
            Customer ID, 0 to n_customers - 1 (0-indexed)
        week: int
            Week number, 0 to 51 (0-indexed)
        prices: np.ndarray
            Prices of items
            Shape: (len_basket,)
        assortment: int
            Assortment ID (corresponding to the assortment, ie the available items,
            of a specific store at a given time)
        """
        if week not in range(52):
            raise ValueError("Week number must be between 0 and 51, inclusive.")

        self.id = id

        # Constitutive elements of a trip
        self.purchases = purchases
        self.customer = customer
        self.week = week
        self.prices = prices
        self.assortment = assortment

        self.trip_length = len(purchases)

    def get_items_up_to_i(self, i: int) -> np.ndarray:
        """Get items up to index i.

        Parameters
        ----------
        i: int
            Index of the item to get

        Returns
        -------
        np.ndarray
            List of items up to index i (excluded)
            Shape: (i,)
        """
        return self.purchases[:i]


class TripDataset:
    """Class for a dataset of trips."""

    def __init__(self, trips: list[Trip], assortments: dict[int, np.ndarray]) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        trips: list[Trip]
            List of trips
            Shape: (n_trips,)
        assortments: dict[int, np.ndarray]
            Dictionary of assortments
            Keys: assortment ID
            Values: np.ndarray of available items
        """
        self.trips = trips
        self.max_length = max([trip.trip_length for trip in self.trips])
        self.n_samples = len(self.transactions())
        self.assortments = assortments

    def __len__(self) -> int:
        """Return the number of trips in the dataset.

        Returns
        -------
        int
            Number of trips in the dataset
        """
        return len(self.trips)

    def __str__(self) -> str:
        """Return short representation of the dataset.

        Returns
        -------
        str
            Representation of the dataset
        """
        return f"TripDataset with {len(self)} trips"

    def __iter__(self) -> iter:
        """Iterate over the trips in the dataset.

        Returns
        -------
        iter
            Iterator over the trips
        """
        return iter(self.trips)

    def add(self, other: object, inplace: bool = False) -> object:
        """Add a dataset to another.

        Parameters
        ----------
        other: TripDataset
            Dataset to add
        inplace: bool
            Whether to add the dataset in-place or not, by default False

        Returns
        -------
        TripDataset
            Concatenated dataset
        """
        if inplace:  # Add another dataset to the current one (in-place)
            # Add new trips
            self.trips += other.trips
            # Update the attributes of the TripDataset
            self.max_length = max([trip.trip_length for trip in self.trips])
            self.n_samples = len(self.transactions())
            # Update the dictionary of assortments (add new keys)
            # If a key already exists, the value is updated
            self.assortments = {**self.assortments, **other.assortments}

            return self

        # Else: create a new dataset by adding 2 datasets together
        return TripDataset(
            trips=self.trips + other.trips,
            # Update the dictionary of assortments (add new keys)
            # If a key already exists, the value is updated
            assortments={**self.assortments, **other.assortments},
        )

    def get_trip(self, index: int) -> Trip:
        """Return the trip at the given index.

        Parameters
        ----------
        index: int
            Index of the trip to get

        Returns
        -------
        Trip
            Trip at the given index
        """
        return self.trips[index]

    def transactions(self) -> np.ndarray:
        """Return the transactions of the TripDataset.

        One transaction is a triplet (customer, trip, item).

        Returns
        -------
        dict
            Transactions of the TripDataset
            keys: trans_id
            values: (customer, trip, item)
        """
        transactions = {}

        trans_id = 0
        for i, trip in enumerate(self.trips):
            for item in trip.purchases:
                transactions[trans_id] = (trip.customer, i, item)
                trans_id += 1

        return transactions

    def get_all_items(self) -> np.ndarray:
        """Return the list of all items available in the dataset.

        Returns
        -------
        np.ndarray
            List of items available in the dataset
        """
        items_list = list(self.assortments.values())
        items_list_flattened = [item for sublist in items_list for item in sublist]
        return np.unique(items_list_flattened)

    def get_all_baskets(self) -> np.ndarray:
        """Return the list of all baskets in the dataset.

        Returns
        -------
        np.ndarray
            List of baskets in the dataset
        """
        return [self.trips[i].purchases for i in range(len(self))]

    def get_all_customers(self) -> np.ndarray:
        """Return the list of all customers in the dataset.

        Returns
        -------
        np.ndarray
            List of customers in the dataset
        """
        # If preprocessing working well, equal to [0, 1, ..., n_customers - 1]
        return np.array(list({self.trips[i].customer for i in range(len(self))}))

    def get_all_weeks(self) -> np.ndarray:
        """Return the list of all weeks in the dataset.

        Returns
        -------
        np.ndarray
            List of weeks in the dataset
        """
        # If preprocessing working well, equal to [0, 1, ..., 51 or 52]
        return np.array(list({self.trips[i].week for i in range(len(self))}))

    def get_all_prices(self) -> np.ndarray:
        """Return the list of all price arrays in the dataset.

        Returns
        -------
        np.ndarray
            List of price arrays in the dataset
        """
        return np.array([self.trips[i].prices for i in range(len(self))])

    def get_all_assortments(self) -> np.ndarray:
        """Return the list of all assortments in the dataset.

        Returns
        -------
        np.ndarray
            List of assortments in the dataset
        """
        return np.array(list({self.trips[i].assortment for i in range(len(self))}))

    def n_items(self) -> int:
        """Return the number of items available in the dataset.

        Returns
        -------
        int
            Number of items available in the dataset
        """
        return len(self.get_all_items())

    def n_customers(self) -> int:
        """Return the number of customers in the dataset.

        Returns
        -------
        int
            Number of customers in the dataset
        """
        return len(self.get_all_customers())

    def n_assortments(self) -> int:
        """Return the number of assortments in the dataset.

        Returns
        -------
        int
            Number of assortments in the dataset
        """
        return len(self.get_all_assortments())

    def get_augmented_data_from_trip_index(
        self,
        trip_index: int,
    ) -> dict[str, np.ndarray]:
        """Get augmented data from a trip index.

        Augmented data includes all the transactions obtained sequentially from the trip:
            - permuted items,
            - permuted, truncated and padded baskets,
            - customers,
            - weeks,
            - prices,
            - item availabilities.

        Parameters
        ----------
        trip_index: int
            Index of the trip from which to get the data

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of data (ie transactions) from the trip
            Keys are: "items", "baskets", "customers", "weeks", "prices", "item_availabilities"
        """
        # Get the trip from the index
        trip = self.trips[trip_index]
        length_trip = len(trip.purchases)

        # Draw a random permutation of the items in the basket without the checkout item 0
        # TODO at a later stage: improve by sampling several permutations here
        permutation_list = list(permutations(range(length_trip - 1)))
        permutation = random.sample(permutation_list, 1)[0]

        # Permute the basket while keeping the checkout item 0 at the end
        permuted_purchases = np.array([trip.purchases[j] for j in permutation] + [0])

        # Truncate the baskets: for each batch sample, we consider the truncation possibilities
        # ranging from an empty basket to the basket with all the elements except the checkout item
        # And pad the truncated baskets with -1 to have the same length (because we need
        # numpy arrays for tiling and numpy arrays must have the same length)
        padded_truncated_purchases = np.array(
            [
                np.concatenate((permuted_purchases[:i], -1 * np.ones(self.max_length - i)))
                for i in range(0, length_trip)
            ],
            dtype=int,
        )

        # Building the availability matrix based on the assortment ID
        # The availability matrix is a binary vector of length n_items
        # where 1 means the item is available and 0 means the item is not available
        availability_matrix = np.zeros(self.n_items())
        availability_matrix[self.assortments[trip.assortment]] = 1

        # Each item is linked to a basket, a customer, a week, prices and an assortment
        return {
            "items": permuted_purchases,
            "baskets": padded_truncated_purchases,
            "customers": np.full(length_trip, trip.customer),
            "weeks": np.full(length_trip, trip.week),
            "prices": np.tile(trip.prices, (length_trip, 1)),
            "item_availabilities": np.tile(availability_matrix, (length_trip, 1)),
        }

    def iter_batch(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> object:
        """Iterate over a TripDataset to return batches of items of length batch_size.

        Parameters
        ----------
        batch_size: int
            Batch size (number of items in the batch)
        shuffle: bool
            Whether or not to shuffle the dataset

        Yields
        ------
        list[np.ndarray]
            For each item in the batch: item, basket, customer, week, prices, item availabilities
            Shape: (6, batch_size)
        """
        # Get trip indexes
        num_trips = len(self)
        trip_indexes = np.arange(num_trips)

        # Shuffle trip indexes
        # TODO: shuffling on the trip indexes or on the item indexes?
        if shuffle:
            trip_indexes = np.random.default_rng(seed=42).permutation(trip_indexes)

        # Initialize the buffer
        buffer = {
            "items": np.empty(0, dtype=int),
            "baskets": np.empty((0, self.max_length), dtype=int),
            "customers": np.empty(0, dtype=int),
            "weeks": np.empty(0, dtype=int),
            "prices": np.empty((0, self.n_items()), dtype=int),
            "item_availabilities": np.empty((0, self.n_items()), dtype=int),
        }

        if batch_size == -1:
            # Get the whole dataset in one batch
            for trip_index in trip_indexes:
                additional_trip_data = self.get_augmented_data_from_trip_index(trip_index)
                buffer["items"] = np.concatenate((buffer["items"], additional_trip_data["items"]))
                buffer["baskets"] = np.concatenate(
                    (buffer["baskets"], additional_trip_data["baskets"])
                )
                buffer["customers"] = np.concatenate(
                    (buffer["customers"], additional_trip_data["customers"])
                )
                buffer["weeks"] = np.concatenate((buffer["weeks"], additional_trip_data["weeks"]))
                buffer["prices"] = np.concatenate(
                    (buffer["prices"], additional_trip_data["prices"])
                )
                buffer["item_availabilities"] = np.concatenate(
                    (buffer["item_availabilities"], additional_trip_data["item_availabilities"])
                )

            # Yield the whole dataset
            yield (
                buffer["items"],
                buffer["baskets"],
                buffer["customers"],
                buffer["weeks"],
                buffer["prices"],
                buffer["item_availabilities"],
            )

        else:
            # Yield batches of size batch_size while going through all the trips
            trip_index = 0
            outer_break = False
            while trip_index < num_trips:
                # Fill the buffer with trips' augmented data until it reaches the batch size
                while len(buffer["items"]) < batch_size:
                    if trip_index >= num_trips:
                        # Then the buffer is not full but there are no more trips to consider
                        # Yield the batch partially filled
                        yield (
                            buffer["items"],
                            buffer["baskets"],
                            buffer["customers"],
                            buffer["weeks"],
                            buffer["prices"],
                            buffer["item_availabilities"],
                        )

                        # Exit the TWO while loops when all trips have been considered
                        outer_break = True
                        break  # Exit the inner loop

                    else:
                        # Consider a new trip to fill the buffer
                        additional_trip_data = self.get_augmented_data_from_trip_index(trip_index)
                        trip_index += 1

                        # Fill the buffer with the new trip
                        buffer["items"] = np.concatenate(
                            (buffer["items"], additional_trip_data["items"])
                        )
                        buffer["baskets"] = np.concatenate(
                            (buffer["baskets"], additional_trip_data["baskets"])
                        )
                        buffer["customers"] = np.concatenate(
                            (buffer["customers"], additional_trip_data["customers"])
                        )
                        buffer["weeks"] = np.concatenate(
                            (buffer["weeks"], additional_trip_data["weeks"])
                        )
                        buffer["prices"] = np.concatenate(
                            (buffer["prices"], additional_trip_data["prices"])
                        )
                        buffer["item_availabilities"] = np.concatenate(
                            (
                                buffer["item_availabilities"],
                                additional_trip_data["item_availabilities"],
                            )
                        )

                if outer_break:
                    # Exit the outer loop
                    break

                # Once the buffer is full, get the batch and update the next buffer
                batch = {
                    "items": buffer["items"][:batch_size],
                    "baskets": buffer["baskets"][:batch_size],
                    "customers": buffer["customers"][:batch_size],
                    "weeks": buffer["weeks"][:batch_size],
                    "prices": buffer["prices"][:batch_size],
                    "item_availabilities": buffer["item_availabilities"][:batch_size],
                }
                buffer = {
                    "items": buffer["items"][batch_size:],
                    "baskets": buffer["baskets"][batch_size:],
                    "customers": buffer["customers"][batch_size:],
                    "weeks": buffer["weeks"][batch_size:],
                    "prices": buffer["prices"][batch_size:],
                    "item_availabilities": buffer["item_availabilities"][batch_size:],
                }

                # Yield the batch
                yield (
                    batch["items"],
                    batch["baskets"],
                    batch["customers"],
                    batch["weeks"],
                    batch["prices"],
                    batch["item_availabilities"],
                )

    def __getitem__(self, index: Union[int, list, np.ndarray, range, slice]) -> Trip:
        """Return the trip at the given index or a list of trips if index is a list.

        Parameters
        ----------
        index: int, list[int], np.ndarray, range or list
            Index or list of indices of the trip(s) to get

        Returns
        -------
        Trip or list[Trip]
            Trip at the given index or list of trips at the given indices
        """
        if isinstance(index, int):
            return self.trips[index]
        if isinstance(index, (list, np.ndarray, range)):
            return [self.trips[i] for i in index]
        if isinstance(index, slice):
            return self.trips[index]
        raise TypeError("Type of index must be int, list, np.ndarray, range or slice.")
