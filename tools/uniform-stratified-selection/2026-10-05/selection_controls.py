"""Counts-matched random masks; no models, Torch imports or global RNG use."""
from __future__ import annotations
import math
import numbers
import numpy as np


def _contract(signs, retain):
    if not isinstance(signs, np.ndarray) or signs.dtype != np.bool_ or signs.ndim != 2:
        raise ValueError('signs must be a two-dimensional Boolean NumPy array')
    if min(signs.shape) < 1:
        raise ValueError('empty batches/findings are unsupported')
    if isinstance(retain, (bool, np.bool_)) or not isinstance(retain, numbers.Real):
        raise ValueError('retain must be a real fraction')
    retain = float(retain)
    if not math.isfinite(retain) or not 0. <= retain <= 1.:
        raise ValueError('retain must be finite and in [0,1]')
    return retain


def make_mask_generators(seed):
    """Two selector-indexed, dedicated PCG64 streams; independent of globals."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return [np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        int(seed), spawn_key=(0x50454552, selector)))) for selector in range(2)]


def expected_support(signs, retain):
    retain = _contract(signs, retain)
    support = np.zeros((signs.shape[1], 2), dtype=np.int64)
    for finding in range(signs.shape[1]):
        for positive in (0, 1):
            size = int(np.count_nonzero(signs[:, finding] == bool(positive)))
            if size:
                support[finding, positive] = max(1, math.floor(retain * size))
    return support


def mask_support(mask, signs):
    _contract(signs, 1.)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != signs.shape:
        raise ValueError('mask must match signs shape and Boolean dtype')
    return np.stack([np.count_nonzero(mask & (signs == bool(p)), axis=0)
                     for p in (0, 1)], axis=1).astype(np.int64)


def uniform_stratified_mask(signs, retain, generator):
    """Uniform fixed-size subsets within each finding/sign stratum.

    Cardinality is the original max(1,floor(retain*n)) rule. A uniformly
    permuted stratum followed by a k-prefix gives each k-subset the same
    probability. Full-retention strata consume no random draws.
    """
    retain = _contract(signs, retain)
    if not isinstance(generator, np.random.Generator):
        raise ValueError('an explicit dedicated NumPy Generator is required')
    mask = np.zeros(signs.shape, dtype=np.bool_)
    counts = expected_support(signs, retain)
    for finding in range(signs.shape[1]):
        for positive in (0, 1):
            indices = np.flatnonzero(signs[:, finding] == bool(positive))
            if indices.size:
                count = int(counts[finding, positive])
                chosen = indices if count == indices.size else generator.permutation(indices)[:count]
                mask[chosen, finding] = True
    if not np.array_equal(mask_support(mask, signs), counts):
        raise ValueError('random-mask retained-count contract failed')
    return mask
