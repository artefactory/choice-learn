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

    Trip = (trip id, purchases, customer, week, prices, assortment)
    """

    def __init__(
        self,
        id: int,
        purchases: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
        assortment: Union[int, np.ndarray],
    ) -> None:
        """Initialize the trip.

        Parameters
        ----------
        id: int
            Trip ID
        purchases: np.ndarray
            List of the ID of the purchased items, 0 to n_items - 1 (0-indexed)
            Shape must be (len_basket,), the last item is the checkout item 0
        customer: int
            Customer ID, 0 to n_customers - 1 (0-indexed)
        week: int
            Week number, 0 to 51 (0-indexed)
        prices: np.ndarray
            Prices of items
            Shape must be (len_basket,)
        assortment: int or np.ndarray
            Assortment ID (int) corresponding to the assortment (ie its index in self.assortments)
            OR availability matrix (np.ndarray) of the assortment (binary vector of length n_items
            where 1 means the item is available and 0 means the item is not available)
            An assortment is the list of available items of a specific store at a given time
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

    def get_items_up_to_index(self, i: int) -> np.ndarray:
        """Get items up to index i.

        Parameters
        ----------
        i: int
            Index of the item to get

        Returns
        -------
        np.ndarray
            List of items up to index i (excluded)
            Shape must be (i,)
        """
        return self.purchases[:i]


class TripDataset:
    """Class for a dataset of trips."""

    def __init__(self, trips: list[Trip], assortments: np.ndarray) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        trips: list[Trip]
            List of trips
            Length must be n_trips
        assortments: np.ndarray
            Array of assortments
            assortments[i]: availability matrix of the assortment whose ID is i
            (The availability matrix is a binary vector of length n_items
            where 1 means the item is available and 0 means the item is not available)
            Shape must be (n_assortments, n_items)
        """
        self.trips = trips
        self.max_length = max([trip.trip_length for trip in self.trips])
        self.n_samples = len(self.get_transactions())
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

    def concatenate(self, other: object, inplace: bool = False) -> object:
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
            # Concatenate the list of trips
            self.trips += other.trips
            # Update the attributes of the TripDataset
            self.max_length = max([trip.trip_length for trip in self.trips])
            self.n_samples = len(self.get_transactions())
            # Concatenate the arrays of assortments
            # /!\ When concatenating 2 TripDatasets, the indices of the assortments
            # changes
            self.assortments = np.concatenate((self.assortments, other.assortments), axis=0)
            return self

        # Else: create a new dataset by adding 2 datasets together
        return TripDataset(
            # Concatenate the list of trips
            trips=self.trips + other.trips,
            # Concatenate the arrays of assortments
            # /!\ When concatenating 2 TripDatasets, the indices of the assortments
            # changes
            assortments=np.concatenate((self.assortments, other.assortments), axis=0),
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

    def get_transactions(self) -> np.ndarray:
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
        return np.arange(self.n_items)

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

    @property
    def n_items(self) -> int:
        """Return the number of items available in the dataset.

        Returns
        -------
        int
            Number of items available in the dataset
        """
        return self.assortments.shape[1]

    @property
    def n_customers(self) -> int:
        """Return the number of customers in the dataset.

        Returns
        -------
        int
            Number of customers in the dataset
        """
        return len(self.get_all_customers())

    @property
    def n_assortments(self) -> int:
        """Return the number of assortments in the dataset.

        Returns
        -------
        int
            Number of assortments in the dataset
        """
        return self.assortments.shape[0]

    def get_augmented_data_from_trip_index(
        self,
        trip_index: int,
    ) -> tuple[np.ndarray]:
        """Get augmented data from a trip index.

        Augmented data includes all the transactions obtained sequentially from the trip:
            - permuted items,
            - permuted, truncated and padded baskets,
            - padded future purchases based on the baskets,
            - customers,
            - weeks,
            - prices,
            - available items.

        Parameters
        ----------
        trip_index: int
            Index of the trip from which to get the data

        Returns
        -------
        tuple[np.ndarray]
            For each sample (ie transaction) from the trip:
            item, basket, future purchases, customer, week, prices, available items
            Length must be 7
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

        # padded_future_purchases are the complements of padded_truncated_purchases, ie the
        # items that are not yet in the (permuted) basket but that we know will be purchased
        # during the next steps of the trip
        # Pad the future purchases with -1 to have the same length
        padded_future_purchases = np.array(
            [
                np.concatenate(
                    (
                        permuted_purchases[i + 1 :],
                        -1 * np.ones(self.max_length - len(permuted_purchases) + i + 1),
                    )
                )
                for i in range(0, length_trip)
            ],
            dtype=int,
        )

        if isinstance(trip.assortment, int):
            # Then it is the assortment ID (ie its index in self.assortments)
            assortment = self.assortments[trip.assortment]
        else:  # np.ndarray
            # Then it is directly the availability matrix
            assortment = trip.assortment

        # Each item is linked to a basket, the future purchases,
        # a customer, a week, prices and an assortment
        return (
            permuted_purchases,  # Items
            padded_truncated_purchases,  # Baskets
            padded_future_purchases,  # Future purchases
            np.full(length_trip, trip.customer),  # Customers
            np.full(length_trip, trip.week),  # Weeks
            np.tile(trip.prices, (length_trip, 1)),  # Prices
            np.tile(assortment, (length_trip, 1)),  # Available items
        )

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
        tuple[np.ndarray]
            For each item in the batch: item, basket, future purchases,
            customer, week, prices, available items
            Length must 7
        """
        # Get trip indexes
        num_trips = len(self)
        trip_indexes = np.arange(num_trips)

        # Shuffle trip indexes
        # TODO: shuffling on the trip indexes or on the item indexes?
        if shuffle:
            trip_indexes = np.random.default_rng().permutation(trip_indexes)

        # Initialize the buffer
        buffer = (
            np.empty(0, dtype=int),  # Items
            np.empty((0, self.max_length), dtype=int),  # Baskets
            np.empty((0, self.max_length), dtype=int),  # Future purchases
            np.empty(0, dtype=int),  # Customers
            np.empty(0, dtype=int),  # Weeks
            np.empty((0, self.n_items), dtype=int),  # Prices
            np.empty((0, self.n_items), dtype=int),  # Available items
        )

        if batch_size == -1:
            # Get the whole dataset in one batch
            for trip_index in trip_indexes:
                additional_trip_data = self.get_augmented_data_from_trip_index(trip_index)
                buffer = tuple(
                    np.concatenate((buffer[i], additional_trip_data[i])) for i in range(len(buffer))
                )

            # Yield the whole dataset
            yield buffer

        else:
            # Yield batches of size batch_size while going through all the trips
            index = 0
            outer_break = False
            while index < num_trips:
                # Fill the buffer with trips' augmented data until it reaches the batch size
                while len(buffer[0]) < batch_size:
                    if index >= num_trips:
                        # Then the buffer is not full but there are no more trips to consider
                        # Yield the batch partially filled
                        yield buffer

                        # Exit the TWO while loops when all trips have been considered
                        outer_break = True
                        break

                    else:
                        # Consider a new trip to fill the buffer
                        additional_trip_data = self.get_augmented_data_from_trip_index(
                            trip_indexes[index]
                        )
                        index += 1

                        # Fill the buffer with the new trip
                        buffer = tuple(
                            np.concatenate((buffer[i], additional_trip_data[i]))
                            for i in range(len(buffer))
                        )

                if outer_break:
                    break

                # Once the buffer is full, get the batch and update the next buffer
                batch = tuple(buffer[i][:batch_size] for i in range(len(buffer)))
                buffer = tuple(buffer[i][batch_size:] for i in range(len(buffer)))

                # Yield the batch
                yield batch

    def __getitem__(self, index: Union[int, list, np.ndarray, range, slice]) -> object:
        """Return a TripDataset object populated with the trips at index.

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
            return TripDataset(
                trips=[self.trips[index]],
                assortments=self.assortments,
            )
        if isinstance(index, (list, np.ndarray, range)):
            return TripDataset(
                trips=[self.trips[i] for i in index],
                assortments=self.assortments,
            )
        if isinstance(index, slice):
            return TripDataset(
                trips=self.trips[index],
                assortments=self.assortments,
            )

        raise TypeError("Type of index must be int, list, np.ndarray, range or slice.")
