import tensorflow as tf


class BayesianWorldModel(tf.Module):
    def __init__(self, config, logger):
        super().__init__()
        self._logger = logger
        self._config = config

    def __call__(self, prev_action, current_observation, belief):
        current_beliefs = self._update_beliefs(
            prev_action, current_observation, belief)
        return current_beliefs

    def generate_sequences_posterior(self, initial_belief, horizon, actor=None, actions=None,
                                     log_sequences=False):
        sequences_posterior = self._generate_sequences_posterior(
            initial_belief, horizon, actor, actions, log_sequences)
        return sequences_posterior

    def reconstruct_sequences_posterior(self, batch):
        sequence_posterior = self._reconstruct_sequences_posterior(batch)
        return sequence_posterior

    def train(self, batch):
        posterior_beliefs = self._training_step(batch)
        return posterior_beliefs

    def reset(self):
        raise NotImplementedError

    def _update_beliefs(self, prev_action, current_observation, belief):
        raise NotImplementedError

    def _generate_sequences_posterior(self, initial_belief, horizon, actor,
                                      actions, log_sequences):
        raise NotImplementedError

    def _reconstruct_sequences_posterior(self, batch):
        raise NotImplementedError

    def _training_step(self, batch):
        raise NotImplementedError
