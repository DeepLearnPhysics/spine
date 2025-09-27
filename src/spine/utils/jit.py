"""JIT compilation decorators using Numba for performance optimization."""

import inspect
from functools import wraps

import numba as nb
import numpy as np

from .conditional import TORCH_AVAILABLE, torch


def numbafy(cast_args=[], list_args=[], keep_torch=False, ref_arg=None):
    """Function which wraps a `numba` function with some checks on the input
    to make the relevant conversions to numpy where necessary.

    Parameters
    ----------
    cast_args : list(str), optional
        List of arguments to be cast to numpy
    list_args : list(str), optional
        List of arguments which need to be cast to a numba typed list
    keep_torch : bool, default False
        Make the output a torch object, if the reference argument is one
    ref_arg : str, optional
        Reference argument used to assign a type and device to the torch output

    Returns
    -------
    callable
        Wrapped function which ensures input type compatibility with numba
    """

    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            # Convert the positional arguments in args into
            # key:value pairs in kwargs
            keys = list(inspect.signature(fn).parameters.keys())
            for i, val in enumerate(args):
                kwargs[keys[i]] = val

            # Extract the default values for the remaining parameters
            for key, val in inspect.signature(fn).parameters.items():
                if key not in kwargs and val.default != inspect.Parameter.empty:
                    kwargs[key] = val.default

            # If a torch output is request, register the input dtype and
            # device location
            if keep_torch:
                assert (
                    ref_arg in kwargs
                ), "Must provide the `ref_arg` to cast back to torch."
                dtype, device = None, None
                if TORCH_AVAILABLE and isinstance(kwargs[ref_arg], torch.Tensor):
                    dtype = kwargs[ref_arg].dtype
                    device = kwargs[ref_arg].device

            # If the cast data is not a numpy array, cast it
            for arg in cast_args:
                assert arg in kwargs, (
                    f"Argument `{arg}` appears in `cast_args` but does "
                    "not appear in the function arguments."
                )
                if not isinstance(kwargs[arg], np.ndarray):
                    if TORCH_AVAILABLE and isinstance(kwargs[arg], torch.Tensor):
                        kwargs[arg] = kwargs[arg].detach().cpu().numpy()
                    else:
                        raise TypeError(
                            f"Can only convert torch.Tensor to numpy when PyTorch is available. "
                            f"Got a request to cast {type(kwargs[arg])} instead."
                        )

            # If there is a reflected list in the input, type it
            for arg in list_args:
                assert arg in kwargs, (
                    f"Argument `{arg}` appears in `list_args` but does "
                    "not appear in the function arguments."
                )
                kwargs[arg] = nb.typed.List(kwargs[arg])

            # Get the output
            ret = fn(**kwargs)
            if keep_torch and dtype and TORCH_AVAILABLE:
                if isinstance(ret, np.ndarray):
                    ret = torch.tensor(ret, dtype=dtype, device=device)
                elif isinstance(ret, list):
                    ret = [torch.tensor(r, dtype=dtype, device=device) for r in ret]
                else:
                    raise TypeError("Return type not recognized, cannot cast to torch.")
            return ret

        return inner

    return outer
