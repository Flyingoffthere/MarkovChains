import numpy as np
import itertools
from typing import Set, Dict, Any, List, Union

class MC:
    def __init__(self,
                 S: Set,
                 initial_distribution: Dict,
                 transition_probs: Dict
                 ) -> None:
        """
        General Markov Chain.
        :param S: state space.
        :param initial_distribution: initial probability distribution of states.
               Must be in the form {state -> probability}
        :param transition_probs: transition probabilities between states.
               Must be in the form {(state_1, state_2) -> probability}
        """

        assert len(S) != 0, "State space must be nonempty"
        assert len(initial_distribution) == len(S), ("Number of states in initial distribution"
                                                     "must match the number of states")
        assert sum(initial_distribution.values()) == 1, "Initial distribution must sum up to 1"

        self._setupComponents(S, initial_distribution, transition_probs)

    def _getStateByIndex(self, idx: int) -> Any:
        return self.state_table[idx]

    def _getIndexByState(self, state: Any) -> int:
        return list(self.state_table.values()).index(state)

    def _convertAlpha(self, prob_vector: np.array) -> Dict:
        indexes = list(range(len(prob_vector)))
        converted = dict()
        for idx in indexes:
            state = self._getStateByIndex(idx)
            p = prob_vector[idx]
            converted[state] = p
        return converted

    def _genStateSpace(self, S: Set) -> None:
        n_states = len(S)
        S_indexes = np.arange(n_states)
        self.state_table = dict(zip(S_indexes, S))

    def _genInitialDistribution(self, initial_distribution: Dict) -> None:
        n_states = len(initial_distribution)
        self.alpha = np.empty(n_states)
        for idx in range(n_states):
            indexed_state = self._getStateByIndex(idx)
            state_prob = initial_distribution[indexed_state]
            self.alpha[idx] = state_prob


    def _genTransitionMatrix(self, transition_probs: Dict) -> None:
        n_states = int(np.sqrt(len(transition_probs)))
        self.P = np.empty((n_states, n_states))
        indexes_pairs = itertools.product(np.arange(n_states), np.arange(n_states))
        for idx_pair in indexes_pairs:
            first_idx, second_idx = idx_pair
            first_state = self._getStateByIndex(first_idx)
            second_state = self._getStateByIndex(second_idx)
            self.P[idx_pair] = transition_probs[(first_state, second_state)]

    def _setupComponents(self,
                         S: Set,
                         initial_distribution: Dict,
                         transition_probs: Dict) -> None:
        self._genStateSpace(S)
        self._genInitialDistribution(initial_distribution)
        self._genTransitionMatrix(transition_probs)

    def _getDistribution(self, n: int) -> np.array:
        return self.alpha @ np.linalg.matrix_power(self.P, n)

    def getDistribution(self, n: int) -> Dict:
        """
        Get the unconditional distribution P(X_n)
        :param n: time
        """
        alpha_n = self._getDistribution(n)
        return self._convertAlpha(alpha_n)

    @staticmethod
    def _sampleChainState(prob_vector: np.array) -> int:
        state_space = np.arange(len(prob_vector))
        state = np.random.choice(state_space, p=prob_vector)
        return state

    def _sampleChainOneStep(self, state: Union[None, int]) -> int:
        if state is None:
            next_state = self._sampleChainState(self.alpha)
            return next_state

        prob_vector = self.P[state, :]
        next_state = self._sampleChainState(prob_vector)
        return next_state

    def _sampleChain(self, n: int, state: Union[None, int] = None) -> List[int]:
        path = []
        for _ in range(n):
            state = self._sampleChainOneStep(state)
            path.append(state)
        return path

    def sampleChain(self, n: int, state: Union[None, Any] = None) -> List[Any]:
        """
        Sample a chain path from X_m = state n steps further
        :param n: number of steps
        :param state: initial state
        """
        state = self._getIndexByState(state) if state else None
        path = self._sampleChain(n, state)
        return list(map(self._getStateByIndex, path))

