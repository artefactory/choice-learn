"""Classes to handle datasets with baskets of products."""

import random
from typing import Union

import numpy as np

from ..utils.permutation import permutations


class Trip:
    """Class for a trip.

    A trip is a sequence of purchases made at a specific time and at a
    specific store with given prices and a specific assortment. It can
    be seen as the content of a time-stamped purchase receipt with store
    identification.

    Trip = (purchases, store, week, prices, assortment)
    """

    def __init__(
        self,
        purchases: np.ndarray,
        prices: np.ndarray,
        assortment: Union[int, np.ndarray],
        store: int = 0,
        week: int = 0,
        user_id: int = 0,
    ) -> None:
        """Initialize the trip.

        Parameters
        ----------
        purchases: np.ndarray
            List of the ID of the purchased items, 0 to n_items - 1 (0-indexed)
            Shape must be (len_basket,), the last item is the checkout item 0
        store: int
            Store ID, 0 to n_stores - 1 (0-indexed)
        week: int
            Week number, 0 to 51 (0-indexed)
        prices: np.ndarray
            Prices of all the items in the dataset
            Shape must be (n_items,) with n_items the number of items in
            the TripDataset
        assortment: int or np.ndarray
            Assortment ID (int) corresponding to the assortment (ie its index in
            self.available_items) OR availability matrix (np.ndarray) of the
            assortment (binary vector of length n_items where 1 means the item
            is available and 0 means the item is not available)
            An assortment is the list of available items of a specific store at a given time
        """
        if week not in range(52):
            raise ValueError("Week number must be between 0 and 51, inclusive.")

        # Constitutive elements of a trip
        self.purchases = purchases
        self.store = store
        self.week = week
        self.prices = prices
        self.assortment = assortment
        self.user_id = user_id

        self.trip_length = len(purchases)

    def __str__(self) -> str:
        """Return short representation of the trip.

        Returns
        -------
        str
            Representation of the trip
        """
        desc = f"Trip with {self.trip_length} purchases {self.purchases}"
        desc += f" at store {self.store} in week {self.week} by user {self.user_id}"
        desc += f" with prices {self.prices} and assortment {self.assortment}"
        return desc

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

    def __init__(
        self,
        trips: list[Trip],
        available_items: np.ndarray,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        trips: list[Trip]
            List of trips
            Length must be n_trips
        available_items: np.ndarray
            Array of availability matrices
            available_items[i]: availability matrix of the assortment whose ID is i
            (The availability matrix is a binary vector of length n_items
            where 1 means the item is available and 0 means the item is not available)
            Shape must be (n_assortments, n_items)
        """
        self.trips = trips
        self.max_length = max([trip.trip_length for trip in self.trips])
        self.n_samples = len(self.get_transactions())
        self.available_items = available_items

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
            # Concatenate the arrays of availability matrices
            # /!\ When concatenating 2 TripDatasets, the indices of the availability matrices
            # changes
            self.available_items = np.concatenate(
                (self.available_items, other.available_items), axis=0
            )
            return self

        # Else: create a new dataset by adding 2 datasets together
        return TripDataset(
            # Concatenate the list of trips
            trips=self.trips + other.trips,
            # Concatenate the arrays of availability matrices
            # /!\ When concatenating 2 TripDatasets, the indices of the availability matrices
            # changes
            available_items=np.concatenate(
                (self.available_items, other.available_items), axis=0
            ),
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

        One transaction is a quadruplet (store, trip, item, user_id).

        Returns
        -------
        dict
            Transactions of the TripDataset
            keys: trans_id
            values: (store, trip, item)
        """
        transactions = {}

        trans_id = 0
        for i, trip in enumerate(self.trips):
            for item in trip.purchases:
                transactions[trans_id] = (trip.store, i, item, trip.user_id)
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
        return np.array([self.trips[i].purchases for i in range(len(self))])

    def get_all_stores(self) -> np.ndarray:
        """Return the list of all stores in the dataset.

        Returns
        -------
        np.ndarray
            List of stores in the dataset
        """
        # If preprocessing working well, equal to [0, 1, ..., n_stores - 1]
        return np.array(list({self.trips[i].store for i in range(len(self))}))

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

    def get_all_users(self, shuffled: bool = False) -> np.ndarray:
        """Return the list of all users in the dataset.

        Returns
        -------
        np.ndarray
            List of users in the dataset
        """
        if shuffled:
            user_ids = list({self.trips[i].user_id for i in range(len(self))})
            random.shuffle(user_ids)  # nosec
            return np.array(user_ids)
        return np.array(list({self.trips[i].user_id for i in range(len(self))}))

    @property
    def n_items(self) -> int:
        """Return the number of items available in the dataset.

        Returns
        -------
        int
            Number of items available in the dataset
        """
        return self.available_items.shape[1]

    @property
    def n_stores(self) -> int:
        """Return the number of stores in the dataset.

        Returns
        -------
        int
            Number of stores in the dataset
        """
        return len(self.get_all_stores())

    @property
    def n_users(self) -> int:
        """Return the number of users in the dataset.

        Returns
        -------
        int
            Number of users in the dataset
        """
        return len(self.get_all_users())

    @property
    def n_assortments(self) -> int:
        """Return the number of assortments in the dataset.

        Returns
        -------
        int
            Number of assortments in the dataset
        """
        return self.available_items.shape[0]

    def get_one_vs_all_augmented_data_from_trip_index(
        self,
        trip_index: int,
    ) -> tuple[np.ndarray]:
        """Get augmented data from a trip index.

        Augmented data consists in removing one item from the basket that will be used
        as a target from the remaining items. It is done for all items, leading to returning:
            - items,
            - padded baskets with an item removed,
            - stores,
            - weeks,
            - prices,
            - available items.
            - user_id

        Parameters
        ----------
        trip_index: int
            Index of the trip from which to get the data

        Returns
        -------
        tuple[np.ndarray]
            For each sample (ie transaction) from the trip:
            item, basket, store, week, prices, available items
            Length must be 6
        """
        # Get the trip from the index
        trip = self.trips[trip_index]
        length_trip = len(trip.purchases)
        permuted_purchases = np.array(trip.purchases)

        # Create new baskets with one item removed that will be used as target
        # (len(basket) new baskets created)
        # And pad the truncated baskets with -1 to have the same length (because we need
        # numpy arrays for tiling and numpy arrays must have the same length)
        padded_purchases_lacking_one_item = np.array(
            [
                np.concatenate(
                    (
                        permuted_purchases[:i],
                        # Pad the removed item with -1
                        [-1],
                        permuted_purchases[i + 1 :],
                        # Pad to have the same length
                        -1 * np.ones(self.max_length - length_trip),
                    )
                )
                for i in range(0, length_trip)
            ],
            dtype=int,
        )

        if not (
            isinstance(trip.assortment, np.ndarray) or isinstance(trip.assortment, list)
        ):
            # Then it is the assortment ID (ie its index in self.available_items)
            assortment = self.available_items[trip.assortment]
        else:  # np.ndarray
            # Then it is directly the availability matrix
            assortment = trip.assortment

        if not (isinstance(trip.prices, np.ndarray) or isinstance(trip.prices, list)):
            # Then it is the assortment ID (ie its index in self.available_items)
            prices = self.prices[trip.prices]
        else:  # np.ndarray
            # Then it is directly the availability matrix
            prices = trip.prices

        # Each item is linked to a basket, a store, a week, prices and an assortment
        return (
            permuted_purchases,  # Items
            padded_purchases_lacking_one_item,  # Baskets
            np.empty((0, self.max_length), dtype=int),  # Future purchases
            np.full(length_trip, trip.store),  # Stores
            np.full(length_trip, trip.week),  # Weeks
            np.tile(prices, (length_trip, 1)),  # Prices
            np.tile(assortment, (length_trip, 1)),  # Available items
            np.full(length_trip, trip.user_id),  # User IDs
        )

    def get_subbaskets_augmented_data_from_trip_index(
        self,
        trip_index: int,
    ) -> tuple[np.ndarray]:
        """Get augmented data from a trip index.

        Augmented data includes all the transactions obtained sequentially from the trip.
        In particular, items in the basket are shuffled and sub-baskets are built iteratively
        with the next item that will be used as a target. In particular, it leads to:
            - permuted items,
            - permuted, truncated and padded baskets,
            - padded future purchases based on the baskets,
            - stores,
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
            item, basket, future purchases, store, week, prices, available items
            Length must be 7
        """
        # Get the trip from the index
        trip = self.trips[trip_index]
        length_trip = len(trip.purchases)

        # Draw a random permutation of the items in the basket without the checkout item 0
        # TODO at a later stage: improve by sampling several permutations here
        permutation_list = list(permutations(range(length_trip - 1)))
        permutation = random.sample(permutation_list, 1)[0]  # nosec

        # Permute the basket while keeping the checkout item 0 at the end
        permuted_purchases = np.array([trip.purchases[j] for j in permutation] + [0])

        # Truncate the baskets: for each batch sample, we consider the truncation possibilities
        # ranging from an empty basket to the basket with all the elements except the checkout item
        # And pad the truncated baskets with -1 to have the same length (because we need
        # numpy arrays for tiling and numpy arrays must have the same length)
        padded_truncated_purchases = np.array(
            [
                np.concatenate(
                    (permuted_purchases[:i], -1 * np.ones(self.max_length - i))
                )
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
            # Then it is the assortment ID (ie its index in self.available_items)
            assortment = self.available_items[trip.assortment]
        else:  # np.ndarray
            # Then it is directly the availability matrix
            assortment = trip.assortment

        # Each item is linked to a basket, the future purchases,
        # a store, a week, prices and an assortment
        return (
            permuted_purchases,  # Items
            padded_truncated_purchases,  # Baskets
            padded_future_purchases,  # Future purchases
            np.full(length_trip, trip.store),  # Stores
            np.full(length_trip, trip.week),  # Weeks
            np.tile(trip.prices, (length_trip, 1)),  # Prices
            np.tile(assortment, (length_trip, 1)),  # Available items
            np.full(length_trip, trip.user_id),  # User IDs
        )

    def get_sequential_data_from_trip_index(
        self,
        trip_index: int,
        sequence_length: int = 5,
        n_future_purchases: int = 3,
    ) -> tuple[np.ndarray]:
        """Get augmented data from a trip index for sequential recommendation.

        Parameters
        ----------

        trip_index: int
            Index of the trip from which to get the data
        sequence_length: int
            Lenght of sequence we consider: example sequence_length=5 means
            we consider the 5th item as target and the first 5 items as the basket.
        n_future_purchases: int
            Number of future purchases to consider: example n_future_purchases=3
            means we consider the next 3 items after the target item as future purchases.

        Returns
        -------
        tuple[np.ndarray]
            For each sample (ie transaction) from the trip:
            item, basket, future purchases, store, week, prices, available items, user_id
        """
        # Get the trip from the index
        trip = self.trips[trip_index]
        purchases = np.array(trip.purchases)

        padded_truncated_purchases = np.array(
            [purchases[:sequence_length]],
            dtype=int,
        )

        padded_future_purchases = np.array(
            [
                np.pad(
                    purchases[
                        sequence_length + 1 : sequence_length + 1 + n_future_purchases
                    ],
                    (
                        0,
                        max(
                            0,
                            n_future_purchases
                            - len(
                                purchases[
                                    sequence_length
                                    + 1 : sequence_length
                                    + 1
                                    + n_future_purchases
                                ]
                            ),
                        ),
                    ),
                    constant_values=-1,
                )
            ],
            dtype=int,
        )
        if isinstance(trip.assortment, int):
            # Then it is the assortment ID (ie its index in self.available_items)
            assortment = self.available_items[trip.assortment]
        else:  # np.ndarray
            # Then it is directly the availability matrix
            assortment = trip.assortment

        return (
            np.array([purchases[sequence_length]]),  # Items
            padded_truncated_purchases,  # Baskets
            padded_future_purchases,  # Future purchases
            np.array([trip.store]),  # Stores
            np.array([trip.week]),  # Weeks
            np.array(trip.prices),  # Prices
            np.array([assortment]),  # Available items
            np.array([trip.user_id]),  # User IDs
        )

    def iter_batch(
        self,
        batch_size: int,
        shuffle: bool = False,
        data_method: str = "shopper",
    ) -> object:
        """Iterate over a TripDataset to return batches of items of length
        batch_size.

        Parameters
        ----------
        batch_size: int
            Batch size (number of items in the batch)
        shuffle: bool
            Whether or not to shuffle the dataset
        data_method: str
            Method used to generate sub-baskets from a purchased one. Available methods are:
            - 'shopper': randomly orders the purchases and creates the ordered sub-baskets:
                         (1|0); (2|1); (3|1,2); (4|1,2,3); etc...
            - 'aleacarta': creates all the sub-baskets with N-1 items:
                           (4|1,2,3); (3|1,2,4); (2|1,3,4); (1|2,3,4)

        Yields
        ------
        tuple[np.ndarray]
            For each item in the batch: item, basket, future purchases,
            store, week, prices, available items, user_id
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
            np.empty(0, dtype=int),  # Stores
            np.empty(0, dtype=int),  # Weeks
            np.empty((0, self.n_items), dtype=int),  # Prices
            np.empty((0, self.n_items), dtype=int),  # Available items
            np.empty(0, dtype=int),  # User IDs
        )

        if data_method == "sequential":
            buffer = (
                np.empty(0, dtype=int),  # Items
                np.empty((0, 5), dtype=int),  # Baskets
                np.empty((0, 3), dtype=int),  # Future purchases
                np.empty(0, dtype=int),  # Stores
                np.empty(0, dtype=int),  # Weeks
                np.empty((0, self.n_items), dtype=int),  # Prices
                np.empty((0, self.n_items), dtype=int),  # Available items
                np.empty(0, dtype=int),  # User IDs
            )

        if batch_size == -1:
            # Get the whole dataset in one batch
            for trip_index in trip_indexes:
                if data_method == "shopper":
                    additional_trip_data = (
                        self.get_subbaskets_augmented_data_from_trip_index(trip_index)
                    )
                elif data_method == "aleacarta":
                    additional_trip_data = (
                        self.get_one_vs_all_augmented_data_from_trip_index(trip_index)
                    )
                elif data_method == "sequential":
                    additional_trip_data = self.get_sequential_data_from_trip_index(
                        trip_index
                    )
                else:
                    raise ValueError(f"Unknown data method: {data_method}")

                buffer = tuple(
                    np.concatenate((buffer[i], additional_trip_data[i]))
                    for i in range(len(buffer))
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
                        if data_method == "shopper":
                            additional_trip_data = (
                                self.get_subbaskets_augmented_data_from_trip_index(
                                    trip_indexes[index]
                                )
                            )
                        elif data_method == "aleacarta":
                            additional_trip_data = (
                                self.get_one_vs_all_augmented_data_from_trip_index(
                                    trip_indexes[index]
                                )
                            )
                        elif data_method == "sequential":
                            additional_trip_data = (
                                self.get_sequential_data_from_trip_index(
                                    trip_indexes[index]
                                )
                            )
                        else:
                            raise ValueError(f"Unknown data method: {data_method}")
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
                available_items=self.available_items,
            )
        if isinstance(index, (list, np.ndarray, range)):
            return TripDataset(
                trips=[self.trips[i] for i in index],
                available_items=self.available_items,
            )
        if isinstance(index, slice):
            return TripDataset(
                trips=self.trips[index],
                available_items=self.available_items,
            )

        raise TypeError("Type of index must be int, list, np.ndarray, range or slice.")
