from functools import wraps
import time

import torch as pt


def gather_nd(params, indices):
    """Tensorflow gather_nd function ported to Pytorch."""

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = pt.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = pt.take(params, idx)
    return out.view(out_shape)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def is_finite(tensor: pt.Tensor) -> bool:
    return pt.all(pt.isfinite(tensor)).item()
