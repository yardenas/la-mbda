import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec


class CemActor(object):
    def __init__(self, world_model, config, action_dim):
        self._world_model = world_model
        self._config = config
        self._dtype = prec.global_policy().compute_dtype
        self._action_dim = action_dim[-1]

    @tf.function
    def __call__(self, observation):
        mu = tf.zeros([self._config.cem_horizon, self._action_dim], dtype=self._dtype)
        sigma = tf.ones_like(mu)
        best_so_far = tf.zeros(self._action_dim, dtype=self._dtype)
        best_so_far_score = -np.inf * tf.ones((), dtype=self._dtype)
        observation_shaped = tf.nest.map_structure(
            lambda x: tf.broadcast_to(x, [self._config.candidates, tf.shape(x)[-1]]),
            observation
        )
        for _ in tf.range(self._config.cem_iterations):
            action_sequences = tf.random.normal(
                shape=[self._config.candidates, self._config.cem_horizon, self._action_dim],
                mean=mu, stddev=sigma, dtype=self._dtype)
            action_sequences = tf.clip_by_value(action_sequences, -1.0, 1.0)
            action_sequences_batch = action_sequences
            predicted_trajectories = self._world_model.generate_sequences_posterior(
                observation_shaped, self._config.cem_horizon, actions=action_sequences_batch
            )
            costs = predicted_trajectories['cost']
            discount = tf.math.cumprod(
                self._config.safety_discount * tf.ones_like(costs), -1, exclusive=False
            )
            predicted_trajectories = {k: tf.cast(tf.reduce_mean(v, 1), self._dtype) for
                                      k, v in predicted_trajectories.items() if k != 'cost'}
            costs = tf.reduce_max(tf.reduce_sum(tf.cast(costs, self._dtype) * discount, 2),
                                  1)
            feasible_set = tf.greater(
                tf.cast(self._config.cost_threshold / self._config.episode_length,
                        self._dtype),
                costs / self._config.cem_horizon)
            omega_cardinality = tf.math.count_nonzero(feasible_set, dtype=tf.int32)
            if tf.equal(omega_cardinality, 0):
                gamma_scores, gamma = tf.nn.top_k(-costs, self._config.elite, sorted=False)
            else:
                discount = tf.math.cumprod(
                    self._config.discount * tf.ones([self._config.cem_horizon], self._dtype),
                    exclusive=True)
                scores = tf.where(feasible_set,
                                  tf.reduce_sum(predicted_trajectories['reward'] * discount, 1),
                                  -np.inf)
                gamma_scores, gamma = tf.nn.top_k(scores, tf.math.minimum(self._config.elite,
                                                                          omega_cardinality),
                                                  sorted=False)
            best_of_elite = tf.cast(tf.argmax(gamma_scores), tf.int32)
            if tf.greater(gamma_scores[best_of_elite], best_so_far_score) and (
                    tf.greater_equal(omega_cardinality, 1)
            ):
                best_so_far = action_sequences[gamma[best_of_elite], 0, :]
                best_so_far_score = gamma_scores[best_of_elite]
                best_so_far.set_shape(self._action_dim)
                best_so_far_score.set_shape([])
            elif not tf.math.is_finite(best_so_far_score):
                best_so_far = action_sequences[gamma[best_of_elite], 0, :]
            elite_actions = tf.gather(action_sequences, gamma, axis=0)
            mean, variance = tf.nn.moments(elite_actions, axes=0)
            mean.set_shape([self._config.cem_horizon, self._action_dim])
            variance.set_shape([self._config.cem_horizon, self._action_dim])
            mu = mean
            sigma = tf.sqrt(variance)
            if tf.less_equal(tf.reduce_mean(sigma), 0.1):
                break
        return tf.clip_by_value(best_so_far, -1.0, 1.0)
