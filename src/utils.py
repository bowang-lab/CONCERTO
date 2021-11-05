from contextlib import contextmanager
import os
import sys
import torch as th
import numpy as np

# from https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_output(stdout=True,stderr=True):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if stdout:
            sys.stdout = devnull
        if stderr:
            sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

@contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextmanager
def th_temp_seed(seed):
    state = th.get_rng_state()
    th.manual_seed(seed)
    try:
        yield
    finally:
        th.set_rng_state(state)
