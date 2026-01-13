"""Generation of all the permutations of an iterable."""

from typing import Union

import numpy as np


def permutations(iterable: Union[list, np.ndarray, tuple], r: Union[int, None] = None) -> iter:
    """Generate all the r length permutations of an iterable (n factorial possibilities).

    Examples
    --------
        >>> permutations('ABCD', 2)
    'AB', 'AC', 'AD', 'BA', 'BC', 'BD', 'CA', 'CB', 'CD', 'DA', 'DB', 'DC'
        >>> permutations(range(3))
    '012', '021', '102', '120', '201', '210'

    Code taken from https://docs.python.org/3/library/itertools.html.

    Parameters
    ----------
    iterable: iterable (list, np.ndarray or tuple)
        Iterable to generate the permutations from
    r: int, optional
        Length of the permutations, by default None
        If None, then r defaults to the length of the iterable

    Returns
    -------
    generator
       Generator of permutations
    """
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return

    indices = list(range(n))
    cycles = list(range(n, n - r, -1))
    yield tuple(pool[i] for i in indices[:r])

    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return
