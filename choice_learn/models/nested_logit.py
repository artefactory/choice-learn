"""Implementation of the Nested Logit model."""

from choice_learn.models import ChoiceModel


class NestedLogit(ChoiceModel):
    """Nested Logit Model."""

    def __init__(self, **kwargs):
        """Instantiate the model."""
        super().__init__(**kwargs)
