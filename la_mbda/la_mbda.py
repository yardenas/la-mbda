import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tqdm import tqdm

import la_mbda.models as models
import la_mbda.utils as utils
from la_mbda.replay_buffer import ReplayBuffer
from la_mbda.swag_world_model import SwagWorldModel


class LAMBDA(tf.Module):
    def __init__(self, config, logger, observation_space, action_space):
        super(LAMBDA, self).__init__()
        self._config = config
        self._logger = logger
        self._training_step = tf.Variable(0, trainable=False)
        self._pretrained = tf.Variable(False, trainable=False)
        self._experience = ReplayBuffer(config.safety, config.observation_type,
                                        observation_space.shape, action_space.shape,
                                        config.sequence_length, config.batch_size, config.seed,
                                        min(config.total_training_steps // config.episode_length,
                                            int(2e6 // config.episode_length)))
        self._dtype = prec.global_policy().compute_dtype
        self._warmup_policy = lambda: np.random.uniform(action_space.low, action_space.high)
        self.actor = models.Actor(config, action_space.shape[0], 4)
        self.model = SwagWorldModel(self._config, self._logger, observation_space.shape)
        self.critic = models.Critic(config, 3)
        self._prev_action = tf.Variable(tf.zeros(action_space.shape, self._dtype), trainable=False)
        self._current_belief = tf.Variable(self.model.reset(), trainable=False)
        if config.safety:
            self.safety_critic = models.SafetyCritic(config, 3)

    def __call__(self, observation, training=True):
        if not self.warm or not self.pretrained_model:
            action = self._warmup_policy()
        if self.warm:
            if not self.pretrained_model and training:
                self.pretrain_model()
            if self.time_to_update and training:
                print("Updating world model, actor and critic.")
                for batch in tqdm(self._experience.sample(self._config.update_steps),
                                  leave=False, total=self._config.update_steps):
                    self.train(batch)
                self._logger.log_metrics(self.training_step)
            if self.time_to_clone_critic:
                self._clone_critics()
            action = self.policy(tf.constant(observation, self._dtype), training).numpy()
        return np.clip(action, -1.0, 1.0)

    @tf.function
    def policy(self, observation, training=True):
        if self._config.observation_type == 'dense':
            observation = utils.normalize_clip(
                tf.cast(observation, self._dtype),
                tf.cast(tf.convert_to_tensor(self._experience.observation_mean), self._dtype),
                tf.cast(tf.sqrt(tf.convert_to_tensor(self._experience.observation_variance)),
                        self._dtype),
                10.0)
        current_belief_posterior = self.model(self._prev_action, observation, self._current_belief)
        self._current_belief.assign(tf.reduce_mean(current_belief_posterior, 1))
        policy = self.actor(self._current_belief)
        action = policy.sample() if training else policy.mode()
        action = tf.cast(tf.squeeze(action, 0), self._dtype)
        self._prev_action.assign(action)
        return action

    @tf.function
    def train(self, batch):
        posterior_features = self.model.train(batch)
        self._train_actor_critic(posterior_features)

    def observe(self, transition):
        self._training_step.assign_add(self._config.action_repeat)
        self._experience.store(transition)
        if transition['terminal'] or transition['info'].get('TimeLimit.truncated'):
            self._prev_action.assign(tf.zeros_like(self._prev_action))
            self._current_belief.assign(self.model.reset())

    def pretrain_model(self):
        training_steps = max(self._config.pretrain_steps, 1)
        print("Pretraining for {} training steps".format(training_steps))
        dataset = self._experience.sample(training_steps)
        for batch in tqdm(dataset, leave=False, total=training_steps):
            self.model.train(batch)
        self._pretrained.assign(True)

    def _train_actor_critic(self, posterior_beliefs):
        posterior_beliefs = {k: tf.reshape(v, [-1, tf.shape(v)[-1]]) for k, v in
                             posterior_beliefs.items()}
        discount = tf.math.cumprod(
            self._config.discount * tf.ones([self._config.horizon], self._dtype), exclusive=True)
        with tf.GradientTape() as actor_tape:
            posterior_sequences = self.model.generate_sequences_posterior(
                posterior_beliefs, self._config.horizon, actor=self.actor)
            shape = tf.shape(posterior_sequences['features'])
            ravel_features = tf.reshape(posterior_sequences['features'],
                                        tf.concat([[-1], shape[2:]], 0))
            values = tf.reshape(self.critic(ravel_features).mode(), shape[:3])
            optimistic_sample, optimistic_value, pessimistic_sample, _ = \
                gather_optimistic_pessimistic_sample(posterior_sequences, values)
            lambda_values = self._compute_objective(
                optimistic_sample,
                tf.cast(optimistic_value, self._dtype)
            )
            actor_loss = -tf.reduce_mean(lambda_values * discount[:-1])
            if self._config.safety:
                cost_values = tf.reshape(self.safety_critic(ravel_features).mode(), shape[:3])
                pessimistic_cost_sample, pessimistic_cost_value, optimistic_cost_sample, _ = \
                    gather_optimistic_pessimistic_sample(posterior_sequences, cost_values)
                penalty, lagrange_multiplier, penalty_multiplier, cond = \
                    self._compute_safety_penalty(
                        pessimistic_cost_sample,
                        tf.cast(pessimistic_cost_value, self._dtype)
                    )
                actor_loss += tf.saturate_cast(penalty, self._dtype)
        actor_grads_norm = self.actor.train(actor_loss, actor_tape)
        critic_loss, critic_grads_norm = self.critic.train(optimistic_sample['features'],
                                                           optimistic_sample['reward'],
                                                           optimistic_sample['terminal'])
        metrics = {'agent/actor_loss': actor_loss, 'agent/actor_grads_norm': actor_grads_norm,
                   'agent/critic_loss': critic_loss, 'agent/critic_grads_norm': critic_grads_norm,
                   'agent/pi_entropy': self.actor(posterior_sequences['features']).entropy()}
        if self._config.safety:
            safety_critic_loss, safety_critic_grads_norm = self.safety_critic.train(
                pessimistic_cost_sample['features'],
                pessimistic_cost_sample['cost'] * self._config.action_repeat,
                pessimistic_cost_sample['terminal'])
            safety_metrics = {'agent/lagrange_multiplier': lagrange_multiplier,
                              'agent/penalty_multiplier': penalty_multiplier,
                              'agent/penalty': penalty,
                              'agent/safety_critic_loss': safety_critic_loss,
                              'agent/safety_critic_grads_norm': safety_critic_grads_norm,
                              'agent/average_safety_cost': tf.reduce_mean(pessimistic_cost_value),
                              'agent/safety_cond': cond}
            metrics.update(safety_metrics)
        self._log_metrics(**metrics)

    def _compute_objective(self, sequence, values):
        lambda_values = utils.compute_lambda_values(
            values[:, 1:], sequence['reward'], sequence['terminal'],
            self._config.discount, self._config.lambda_)
        return lambda_values

    def _compute_safety_penalty(self, sequence, cost_values):
        cost_lambda_values = utils.compute_lambda_values(
            cost_values[:, 1:], sequence['cost'] * self._config.action_repeat,
            sequence['terminal'], self._config.safety_discount, self._config.safety_lambda)
        return self.safety_critic.penalize(cost_lambda_values)

    def _clone_critics(self):
        self.critic.clone()
        if self._config.safety:
            self.safety_critic.clone()

    def _log_metrics(self, **kwargs):
        for k, v in kwargs.items():
            self._logger[k].update_state(v)

    @property
    def pretrained_model(self):
        return self._pretrained.value()

    @property
    def training_step(self):
        return self._training_step.value()

    @property
    def time_to_update(self):
        return self.training_step and \
               self.training_step % self._config.steps_per_update < self._config.action_repeat

    @property
    def warm(self):
        return self.training_step >= self._config.warmup_training_steps

    @property
    def time_to_clone_critic(self):
        return self.training_step and \
               self.training_step % self._config.steps_per_critic_clone < self._config.action_repeat


def gather_optimistic_pessimistic_sample(posterior, values):
    optimistic_sample, optimistic_values = gather_optimistic_sample(posterior, values)
    pessimistic_sample, pessimistic_values = gather_optimistic_sample(posterior, -values)
    return optimistic_sample, optimistic_values, pessimistic_sample, -pessimistic_values


def gather_optimistic_sample(posterior, values):
    values_summary = tf.reduce_mean(values, 2)
    optimistic_ids = tf.stack([tf.cast(tf.range(tf.shape(values)[0]), tf.int64),
                               tf.argmax(values_summary, 1)], 1)
    optimistic_sample = {k: tf.gather_nd(v, optimistic_ids) for k, v in posterior.items()}
    optimistic_values = tf.gather_nd(values, optimistic_ids)
    return optimistic_sample, optimistic_values
