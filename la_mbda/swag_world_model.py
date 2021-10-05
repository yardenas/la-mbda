import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate

import la_mbda.models as models
from la_mbda.bayesian_world_model import BayesianWorldModel
from la_mbda.swag import SWAG, WeightsSampler


class CyclicLearningRate(CyclicalLearningRate):
    def __init__(
            self,
            initial_learning_rate,
            maximal_learning_rate,
            step_size,
            scale_fn=lambda x: 1.0,
            scale_mode='cycle',
            name='CyclicLearningRate'):
        super(CyclicLearningRate, self).__init__(
            initial_learning_rate,
            maximal_learning_rate,
            step_size,
            scale_fn,
            scale_mode,
            name)

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            mode_step = cycle if self.scale_mode == "cycle" else step

            return maximal_learning_rate - (
                    maximal_learning_rate - initial_learning_rate
            ) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step)


class SwagWorldModel(BayesianWorldModel):
    def __init__(self, config, logger, observation_shape):
        super(SwagWorldModel, self).__init__(config, logger)
        self.optimizer = SWAG(
            tf.optimizers.Adam(
                CyclicLearningRate(config.model_learning_rate,
                                   config.model_learning_rate_factor * config.model_learning_rate,
                                   config.swag_period),
                clipnorm=config.model_grad_clip_norm,
                epsilon=1e-5),
            config.swag_burnin,
            config.swag_period,
            config.swag_models,
            config.swag_decay,
            verbose=False)
        self._model = models.WorldModel(
            config.observation_type,
            observation_shape,
            config.stochastic_size,
            config.deterministic_size,
            config.units,
            config.safety,
            config.cost_imbalance_weight,
            free_nats=config.free_nats,
            mix=config.kl_mix,
            kl_scale=config.kl_scale)
        self.reset()
        self._posterior_samples = config.posterior_samples

    def reset(self):
        belief = self._model.reset(1)
        return tf.concat([belief['stochastic'], belief['deterministic']], -1)

    @property
    def variables(self):
        return self._model.variables + self.optimizer.variables

    @tf.function
    def _update_beliefs(self, prev_action, current_observation, belief):
        features = self._model(prev_action, current_observation, belief)
        return features[:, None]

    @tf.function
    def _generate_sequences_posterior(self, initial_belief, horizon, actor,
                                      actions, log_sequences):
        samples_rollouts = {'features': [],
                            'reward': [],
                            'terminal': []}
        if log_sequences:
            samples_rollouts['reconstructed_observation'] = []
        if self._config.safety:
            samples_rollouts['cost'] = []
        with WeightsSampler(self.optimizer) as sampler:
            for i in range(self._posterior_samples):
                sampler.sample(self._config.sampling_scale)
                sequence = self._model.generate_sequence(initial_belief, horizon, actor=actor,
                                                         actions=actions,
                                                         log_sequences=log_sequences)
                for k, v in samples_rollouts.items():
                    v.append(sequence[k])
        sequence_data = {k: tf.stack(v, 1) for k, v in samples_rollouts.items()}
        return sequence_data

    @tf.function
    def _reconstruct_sequences_posterior(self, batch):
        samples_reconstructed = []
        samples_beliefs = []
        with WeightsSampler(self.optimizer) as sampler:
            for i in range(self._posterior_samples):
                sampler.sample(self._config.sampling_scale)
                results = self._model.inference_step(batch)
                samples_reconstructed.append(results['reconstructed'].mean())
                samples_beliefs.append(results['beliefs'])
        sequence_data = {k: tf.stack(
            [belief[k] for belief in samples_beliefs], 1) for k in
            samples_beliefs[0].keys()}
        sequence_data['reconstructed_observation'] = tf.stack(samples_reconstructed, 1)
        return sequence_data

    @tf.function
    def _training_step(self, batch):
        with tf.GradientTape() as model_tape:
            results = self._model.inference_step(batch)
        grads = model_tape.gradient(results['loss'], self._model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        self._logger['agent/observation_log_p'].update_state(-results['log_p_observations'])
        self._logger['agent/rewards_log_p'].update_state(-results['log_p_rewards'])
        self._logger['agent/terminals_log_p'].update_state(-results['log_p_terminals'])
        self._logger['agent/kl'].update_state(results['kl'])
        self._logger['agent/world_model_loss'].update_state(results['loss'])
        self._logger['agent/world_model_grads'].update_state(tf.linalg.global_norm(grads))
        if self._config.safety:
            self._logger['agent/cost_log_p'].update_state(-results['log_p_costs'])
        return results['beliefs']
