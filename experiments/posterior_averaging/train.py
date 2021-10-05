import tensorflow as tf

import experiments.train_utils as train_utils
from la_mbda.la_mbda import LAMBDA


class PosteriorAveraging(LAMBDA):
    def __init__(self, config, logger, observation_space, action_space):
        super(PosteriorAveraging, self).__init__(config, logger, observation_space, action_space)

    def _train_actor_critic(self, posterior_beliefs):
        posterior_beliefs = {k: tf.reshape(v, [-1, tf.shape(v)[-1]]) for k, v in
                             posterior_beliefs.items()}
        discount = tf.math.cumprod(
            self._config.discount * tf.ones([self._config.horizon], tf.float32), exclusive=True
        )
        with tf.GradientTape() as actor_tape:
            posterior_sequences = self.model.generate_sequences_posterior(
                posterior_beliefs, self._config.horizon, actor=self.actor)
            mean_sequence = tf.nest.map_structure(lambda x: tf.reduce_mean(
                tf.cast(x, tf.float32), 1), posterior_sequences)
            values = self.critic(mean_sequence['features']).mode()
            lambda_values = self._compute_objective(mean_sequence, values)
            actor_loss = -tf.reduce_mean(lambda_values * discount[:-1])
            if self._config.safety:
                cost_values = self.safety_critic(mean_sequence['features']).mode()
                penalty, lagrange_multiplier, penalty_multiplier, cond = \
                    self._compute_safety_penalty(mean_sequence, cost_values)
                actor_loss += penalty
        actor_grads_norm = self.actor.train(actor_loss, actor_tape)
        critic_loss, critic_grads_norm = self.critic.train(mean_sequence['features'],
                                                           mean_sequence['reward'],
                                                           mean_sequence['terminal'])
        metrics = {'agent/actor_loss': actor_loss, 'agent/actor_grads_norm': actor_grads_norm,
                   'agent/critic_loss': critic_loss, 'agent/critic_grads_norm': critic_grads_norm,
                   'agent/pi_entropy': self.actor(posterior_sequences['features']).entropy()}
        if self._config.safety:
            safety_critic_loss, safety_critic_grads_norm = self.safety_critic.train(
                mean_sequence['features'],
                mean_sequence['cost'] * self._config.action_repeat,
                mean_sequence['terminal'])
            safety_metrics = {'agent/lagrange_multiplier': lagrange_multiplier,
                              'agent/penalty_multiplier': penalty_multiplier,
                              'agent/safety_critic_loss': safety_critic_loss,
                              'agent/safety_critic_grads_norm': safety_critic_grads_norm,
                              'agent/average_safety_cost': tf.reduce_mean(cost_values),
                              'agent/safety_cond': cond}
            metrics.update(safety_metrics)
        self._log_metrics(**metrics)


if __name__ == '__main__':
    config = train_utils.make_config(train_utils.define_config())
    train_utils.train(config, PosteriorAveraging)
