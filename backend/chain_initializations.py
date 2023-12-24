import pandas as pd
import itertools
from backend.MC import MC
from typing import Callable



def typedInitialization(file_type: str) -> Callable:
    if file_type == "json":
        return pd.read_json
    else:
        raise NotImplementedError

def initializeChainFromStructured(
        file_type: str,
        path_to_initial_distribution: str,
        path_to_transition_probabilities: str
) -> MC:
    initialization_func = typedInitialization(file_type)
    initial_distr_df = initialization_func(path_to_initial_distribution)
    trans_df = initialization_func(path_to_transition_probabilities)
    initial_distr = initial_distr_df.to_dict()
    states = set(initial_distr_df.index)
    trans_dict = dict()
    for state1, state2 in itertools.product(trans_df.index, trans_df.index):
        p = trans_df.loc[state1, state2]
        trans_dict[(state1, state2)] = p
    mc = MC(states, initial_distr, trans_dict)
    return mc