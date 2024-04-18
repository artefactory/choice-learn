"""Implementation of the Nested Logit model."""
import numpy as np

from choice_learn.models import ChoiceModel


class NestedLogit(ChoiceModel):
    """Nested Logit Model."""

    def __init__(self, items_nests, **kwargs):
        """Instantiate the model."""
        super().__init__(**kwargs)

        # Checking the items_nests format:
        if len(items_nests) < 2:
            raise ValueError(f"At least two nests should be given, got {len(items_nests)}")
        for i_nest, nest in enumerate(items_nests):
            if len(nest) < 1:
                raise ValueError(f"Nest {i_nest} is empty.")
            print(f"Got nest {i_nest} on {len(nest)} with {len(nest)} items.")
        flat_items = np.stack(items_nests).flatten()
        if np.max(flat_items) >= len(flat_items):
            raise ValueError(
                f"""{len(flat_items)} have been given,\
                             cannot have an item index greater than this."""
            )
        if len(np.unique(flat_items)) != len(flat_items):
            raise ValueError("Got at least one items in several nests, which is not possible.")
