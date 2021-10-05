import tensorflow as tf

import experiments.train_utils as train_utils
import la_mbda.utils as utils
from cem_actor import CemActor
from la_mbda.la_mbda import LAMBDA


class CemMpc(LAMBDA):
    def __init__(self, config, logger, observation_space, action_space):
        super(CemMpc, self).__init__(config, logger, observation_space, action_space)
        self.actor = CemActor(self.model, config, action_space.shape)

    @tf.function
    def policy(self, observation, training=True):
        if self._config.observation_type == 'dense':
            observation = utils.normalize_clip(
                tf.cast(observation, self._dtype),
                tf.cast(tf.convert_to_tensor(self._experience.observation_mean), self._dtype),
                tf.cast(tf.sqrt(tf.convert_to_tensor(self._experience.observation_variance)),
                        self._dtype),
                10.0)
        current_belief_posterior = tf.reduce_mean(
            self.model(self._prev_action, observation, self._current_belief), 1
        )
        self._current_belief.assign(current_belief_posterior)
        d_t = current_belief_posterior[:, self._config.stochastic_size:]
        z_t = current_belief_posterior[:, :self._config.stochastic_size]
        current_belief = {'stochastic': z_t, 'deterministic': d_t}
        action = self.actor(current_belief)
        self._prev_action.assign(action)
        return action

    @tf.function
    def train(self, batch):
        self.model.train(batch)


if __name__ == '__main__':
    config = train_utils.define_config()
    config.update({'candidates': 150,
                   'cem_iterations': 10,
                   'cem_horizon': 8,
                   'elite': 15})
    config = train_utils.make_config(config)
    train_utils.train(config, CemMpc)
