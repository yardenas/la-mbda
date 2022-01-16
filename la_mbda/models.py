import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import la_mbda.building_blocks as blocks
import la_mbda.utils as utils


class WorldModel(tf.Module):
    def __init__(self, observation_type, observation_shape, stochastic_size, deterministic_size,
                 units, safety, cost_weight=1.0, free_nats=1.0, mix=0.8, kl_scale=1.0,
                 observation_layers=3, reward_layers=2, terminal_layers=2, activation=tf.nn.elu,
                 hidden_size=200):
        super().__init__()
        self._dtype = prec.global_policy().compute_dtype
        self._f = tf.keras.layers.GRUCell(deterministic_size)
        self._observation_encoder = blocks.encoder(observation_type, observation_shape,
                                                   observation_layers, units)
        self._observation_decoder = blocks.decoder(observation_type, observation_shape, 3, units)
        self._reward_decoder = blocks.DenseDecoder((), reward_layers, units, activation)
        self._safety = safety
        if self._safety:
            self._cost_weight = cost_weight
            self._cost_decoder = blocks.DenseDecoder((), reward_layers, units, activation,
                                                     'bernoulli')
        self._terminal_decoder = blocks.DenseDecoder(
            (), terminal_layers, units, activation, 'bernoulli')
        self._prior_encoder = tf.keras.layers.Dense(deterministic_size, tf.nn.elu)
        self._posterior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(hidden_size, activation) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._prior_decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(hidden_size, activation) for _ in range(1)] +
            [tf.keras.layers.Dense(2 * stochastic_size)])
        self._stochastic_size = stochastic_size
        self._deterministic_size = deterministic_size
        self._free_nats = tf.constant(free_nats, self._dtype)
        self._mix = tf.constant(mix, self._dtype)
        self._kl_scale = tf.constant(kl_scale, self._dtype)

    def __call__(self, prev_action, current_observation, features):
        d_t = features[:, self._stochastic_size:]
        z_t = features[:, :self._stochastic_size]
        _, _, d_t = self._predict(z_t, d_t, prev_action[None, ...])
        embeddings = self._observation_encoder(current_observation[None, None, ...])
        _, z_t = self._correct(d_t, embeddings[0])
        updated_features = tf.concat([z_t, d_t], -1)
        return updated_features

    def _predict(self, prev_stochastic, prev_deterministic, prev_action):
        d_t_z_t_1 = tf.concat([prev_action, prev_stochastic], -1)
        d_t, _ = self._f(self._prior_encoder(d_t_z_t_1), prev_deterministic)
        prior_mu, prior_stddev = tf.split(self._prior_decoder(d_t), 2, -1)
        prior_stddev = tf.math.softplus(prior_stddev) + 0.1
        prior = tfd.MultivariateNormalDiag(prior_mu, prior_stddev)
        z_t = prior.sample()
        return prior, z_t, d_t

    def _correct(self, prev_deterministic, embeddings):
        posterior_mu, posterior_stddev = tf.split(
            self._posterior_decoder(tf.concat([prev_deterministic, embeddings], -1)), 2, -1)
        posterior_stddev = tf.math.softplus(posterior_stddev) + 0.1
        posterior = tfd.MultivariateNormalDiag(posterior_mu, posterior_stddev)
        z_t = posterior.sample()
        return posterior, z_t

    def reset(self, batch_size):
        initial = {'stochastic': tf.zeros([batch_size, self._stochastic_size], self._dtype),
                   'deterministic': self._f.get_initial_state(None, batch_size, self._dtype)}
        return initial

    def _observe_sequence(self, batch):
        embeddings = self._observation_encoder(batch['observation'][:, 1:])
        actions = batch['action']
        horizon = tf.shape(embeddings)[1]
        inferred = {'stochastics': tf.TensorArray(self._dtype, horizon),
                    'deterministics': tf.TensorArray(self._dtype, horizon),
                    'prior_mus': tf.TensorArray(self._dtype, horizon),
                    'prior_stddevs': tf.TensorArray(self._dtype, horizon),
                    'posterior_mus': tf.TensorArray(self._dtype, horizon),
                    'posterior_stddevs': tf.TensorArray(self._dtype, horizon)}
        state = self.reset(tf.shape(actions)[0])
        d_t = state['deterministic']
        z_t = state['stochastic']
        for t in range(horizon):
            prior, _, d_t = self._predict(z_t, d_t, actions[:, t])
            posterior, z_t = self._correct(d_t, embeddings[:, t])
            inferred['stochastics'] = inferred['stochastics'].write(t, z_t)
            inferred['deterministics'] = inferred['deterministics'].write(t, d_t)
            inferred['prior_mus'] = inferred['prior_mus'].write(t, prior.mean())
            inferred['prior_stddevs'] = inferred['prior_stddevs'].write(t, prior.stddev())
            inferred['posterior_mus'] = inferred['posterior_mus'].write(
                t, posterior.mean())
            inferred['posterior_stddevs'] = inferred['posterior_stddevs'].write(
                t, posterior.stddev())
        stacked_inferred = {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in inferred.items()}
        beliefs = {'stochastic': stacked_inferred['stochastics'],
                   'deterministic': stacked_inferred['deterministics']}
        prior = (stacked_inferred['prior_mus'], stacked_inferred['prior_stddevs'])
        posterior = (stacked_inferred['posterior_mus'], stacked_inferred['posterior_stddevs'])
        return beliefs, prior, posterior

    def inference_step(self, batch):
        beliefs, prior, posterior = self._observe_sequence(batch)
        kl_loss, kl = balanced_kl_loss(posterior, prior, self._free_nats, self._mix)
        features = tf.concat([beliefs['stochastic'],
                              beliefs['deterministic']], -1)
        reconstructed = self._observation_decoder(features)
        log_p_observations = tf.reduce_mean(reconstructed.log_prob(batch['observation'][:, 1:]))
        log_p_rewards = tf.reduce_mean(
            self._reward_decoder(features).log_prob(batch['reward']))
        log_p_terminals = tf.reduce_mean(
            self._terminal_decoder(features).log_prob(batch['terminal']))
        loss = self._kl_scale * kl_loss - log_p_observations - log_p_rewards - log_p_terminals
        results = dict(loss=loss, kl=kl, log_p_observations=log_p_observations,
                       log_p_rewards=log_p_rewards, log_p_terminals=log_p_terminals,
                       reconstructed=reconstructed, beliefs=beliefs)
        if self._safety:
            unsafe = tf.greater_equal(batch['cost'], 1.0)
            log_p_cost = self._cost_decoder(features).log_prob(tf.cast(unsafe, self._dtype))
            # Cost distribution is potentially imbalanced, hence we weight more unsafe states.
            log_p_cost = tf.reduce_mean(tf.where(unsafe, log_p_cost * self._cost_weight,
                                                 log_p_cost))
            results['loss'] -= log_p_cost
            results['log_p_costs'] = log_p_cost
        return results

    def generate_sequence(self, initial_belief, horizon, actor=None, actions=None,
                          log_sequences=False):
        sequence_features = tf.TensorArray(self._dtype, horizon)
        features = tf.concat([initial_belief['stochastic'], initial_belief['deterministic']], -1)
        features = tf.ensure_shape(
            features, [None, self._stochastic_size + self._deterministic_size])
        d_t = initial_belief['deterministic']
        z_t = initial_belief['stochastic']
        for t in range(horizon):
            action = actor(tf.stop_gradient(features)).sample() \
                if actions is None else actions[:, t]
            action = tf.cast(action, self._dtype)
            prior, z_t, d_t = self._predict(z_t, d_t, action)
            features = tf.concat([z_t, d_t], -1)
            sequence_features = sequence_features.write(t, features)
        stacked = {'features': tf.transpose(sequence_features.stack(), [1, 0, 2])}
        if log_sequences:
            stacked['reconstructed_observation'] = self._observation_decoder(
                stacked['features']).mean()
        stacked['reward'] = self._reward_decoder(stacked['features']).mean()
        stacked['terminal'] = self._terminal_decoder(stacked['features']).mean()
        if self._safety:
            stacked['cost'] = self._cost_decoder(stacked['features']).mean()
        return stacked


# https://github.com/danijar/dreamerv2/blob/259e3faa0e01099533e29b0efafdf240adeda4b5/common/nets
# .py#L130
def balanced_kl_loss(posterior, prior, free_nats, mix):
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    mvn = tfd.MultivariateNormalDiag
    lhs = tf.reduce_mean(tfd.kl_divergence(mvn(*posterior), mvn(*sg(prior))))
    rhs = tf.reduce_mean(tfd.kl_divergence(mvn(*sg(posterior)), mvn(*prior)))
    return (1.0 - mix) * tf.maximum(lhs, free_nats) + mix * tf.maximum(rhs, free_nats), lhs


class Critic(tf.Module):
    def __init__(self, config, layers):
        super().__init__()
        self._config = config
        self._value = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.elu, dtype=tf.float32)
             for _ in range(layers)] +
            [tf.keras.layers.Dense(1, dtype=tf.float32)])
        self._delayed_value = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.elu, dtype=tf.float32)
             for _ in range(layers)] +
            [tf.keras.layers.Dense(1, dtype=tf.float32)])
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.critic_learning_rate,
            clipnorm=self._config.critic_grad_clip_norm,
            epsilon=1e-5)
        self._lambda = self._config.lambda_
        self._discount = self._config.discount

    def __call__(self, observation):
        observation = tf.cast(observation, tf.float32)
        mu = tf.squeeze(self._value(observation), axis=2)
        return tfd.Independent(tfd.Normal(loc=mu, scale=1.0), 0)

    def train(self, features, reward, terminal):
        reward = tf.cast(reward, tf.float32)
        terminal = tf.cast(terminal, tf.float32)
        lambda_values = utils.compute_lambda_values(
            tf.squeeze(self._delayed_value(features[:, 1:]), -1), reward, terminal,
            self._discount, self._lambda)
        with tf.GradientTape() as critic_tape:
            critic_loss = -tf.reduce_mean(self.__call__(
                features[:, :-1]).log_prob(tf.stop_gradient(lambda_values)))
        grads = critic_tape.gradient(critic_loss, self._value.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._value.trainable_variables))
        grads_mag = tf.linalg.global_norm(grads)
        return critic_loss, grads_mag

    def clone(self):
        # Clone only after initialization.
        if self._delayed_value.inputs is not None:
            utils.clone_model(self._value, self._delayed_value)


class SafetyCritic(Critic):
    def __init__(self, config, layers):
        super(SafetyCritic, self).__init__(config, layers)
        self._lagrange_multiplier = tf.Variable(self._config.lagrangian_mu, False,
                                                dtype=tf.float32)
        self._penalty_multiplier = tf.Variable(self._config.penalty_mu, False, dtype=tf.float32)
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.safety_critic_learning_rate,
            clipnorm=self._config.safety_critic_grad_clip_norm,
            epsilon=1e-5)
        self._cost_threshold = config.cost_threshold
        self._lambda = self._config.safety_lambda
        self._discount = self._config.safety_discount

    def penalize(self, cost_value):
        # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
        # (with a slight change of notation)
        # Taking the mean value since E[V_c(s)]_p(s) ~= J_c
        g = tf.cast(tf.reduce_mean(cost_value - self._cost_threshold), tf.float32)
        lambda_ = tf.convert_to_tensor(self._lagrange_multiplier)
        c = tf.convert_to_tensor(self._penalty_multiplier)
        cond = lambda_ + c * g
        self._lagrange_multiplier.assign(tf.nn.relu(cond))
        psi = tf.where(tf.greater(cond, 0.0),
                       lambda_ * g + c / 2.0 * g ** 2,
                       -1.0 / (2.0 * c) * lambda_ ** 2)
        # Clip to make sure that c is non-decreasing.
        self._penalty_multiplier.assign(
            tf.clip_by_value(c * (self._config.penalty_power_factor + 1.0), c, 1.0))
        return psi, lambda_, c, tf.greater(cond, 0.0)


class Actor(tf.Module):
    def __init__(self, config, size, layers):
        super().__init__()
        self._config = config
        self._policy = tf.keras.Sequential(
            [tf.keras.layers.Dense(config.units, tf.nn.elu) for _ in range(layers)]
            + [tf.keras.layers.Dense(2 * size)])
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.actor_learning_rate,
            clipnorm=self._config.actor_grad_clip_norm,
            epsilon=1e-5)

    def __call__(self, observation):
        mu, stddev = tf.split(self._policy(observation), 2, -1)
        init_std = np.log(np.exp(5.0) - 1)
        stddev = tf.math.softplus(stddev + init_std) + 1e-4
        multivariate_normal_diag = tfd.Normal(
            loc=tf.cast(5.0 * tf.tanh(mu / 5.0), tf.float32),
            scale=tf.cast(stddev, tf.float32))
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(multivariate_normal_diag, StableTanhBijector())
        dist = tfd.Independent(squashed, 1)
        return SampleDist(dist, self._config.seed)

    def train(self, loss, tape):
        grads = tape.gradient(loss, self.trainable_variables)
        norm = tf.linalg.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, self._config.actor_grad_clip_norm, norm)
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return norm


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
# Following implementation @ https://github.com/danijar/dreamer
class StableTanhBijector(tfp.bijectors.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super(StableTanhBijector, self).__init__(validate_args=validate_args, name=name)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.clip_by_value(y, -0.99999997, 0.99999997)
        y = tf.atanh(y)
        return tf.saturate_cast(y, dtype)


# Following implementation @ https://github.com/danijar/dreamer
class SampleDist(object):
    def __init__(self, dist, seed, samples=100):
        self._dist = dist
        self._samples = samples
        # Use a stateless seed to get the same samples everytime -
        # this simulates the fact that the mean, entropy and mode are deterministic.
        self._seed = (0, seed)

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples, seed=self._seed)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)
