from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tensorflow.keras.mixed_precision import experimental as prec
from tqdm import tqdm


def compute_lambda_values(next_values, rewards, terminals, discount, lambda_):
    lambda_values = []
    v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
    horizon = next_values.shape[1]
    for t in reversed(range(horizon)):
        td = rewards[:, t] + (1.0 - terminals[:, t]) * (
                1.0 - lambda_) * discount * next_values[:, t]
        v_lambda = td + v_lambda * lambda_ * discount
        lambda_values.append(v_lambda)
    return tf.stack(lambda_values, 1)


class TrainingLogger(object):
    def __init__(self, config):
        self._writer = SummaryWriter(config.log_dir)
        self._metrics = defaultdict(tf.metrics.Mean)
        dump_string(pretty_print(config), config.log_dir + '/params.txt')

    def __getitem__(self, item):
        return self._metrics[item]

    def __setitem__(self, key, value):
        self._metrics[key] = value

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_metrics(self, step):
        print("\n----Training step {} summary----".format(step))
        for k, v in self._metrics.items():
            print("{:<40} {:<.2f}".format(k, float(v.result())))
            self._writer.add_scalar(k, float(v.result()), step)
            v.reset_states()
        self._writer.flush()

    # (N, T, C, H, W)
    def log_video(self, images, step=None, name='policy', fps=15):
        self._writer.add_video(name, images, step, fps=fps)
        self._writer.flush()

    def log_images(self, images, step=None, name='policy'):
        self._writer.add_images(name, images, step, dataformats='NHWC')
        self._writer.flush()

    def log_figure(self, figure, step=None, name='policy'):
        self._writer.add_figure(name, figure, step)
        self._writer.flush()


def do_episode(agent, training, environment, config, pbar, render, reset_function=None):
    observation = environment.reset() if reset_function is None else reset_function()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training)
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated', False)
        if training:
            agent.observe(dict(observation=observation.astype(np.float32),
                               next_observation=next_observation.astype(np.float32),
                               action=action.astype(np.float32),
                               reward=np.array(reward, np.float32),
                               terminal=np.array(terminal, np.bool),
                               info=info))
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        pbar.update(config.action_repeat)
        steps += config.action_repeat
    episode_summary['steps'] = [steps]
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True, on_episode_end=None):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = \
            do_episode(agent, training,
                       environment, config,
                       pbar, len(episodes) < config.render_episodes and not training)
        steps_count += episode_steps
        episodes.append(episode_summary)
        if on_episode_end is not None:
            on_episode_end(episode_summary, steps_count)
    pbar.close()
    return steps, episodes


def pretty_print(config, indent=0):
    summary = str()
    align = 30 - indent * 2
    for key, value in vars(config).items():
        summary += '  ' * indent + '{:{align}}'.format(str(key), align=align)
        summary += '{}\n'.format(str(value))
    return summary


def dump_string(string, filename):
    with open(filename, 'w+') as file:
        file.write(string)


def preprocess(image, bias=0.0):
    return image / 255.0 - bias


def normalize_clip(data, mean, stddev, clip):
    stddev = tf.where(stddev < 1e-6, tf.cast(1.0, stddev.dtype), stddev)
    return tf.clip_by_value((data - mean) / stddev,
                            -clip, clip)


def clone_model(a, b):
    for var_a, var_b in zip(a.trainable_variables, b.trainable_variables):
        var_b.assign(var_a)


def split_batch(batch, split_size):
    div = tf.shape(batch)[0] // split_size
    return tf.split(batch, [div] * (split_size - 1) +
                    [div + tf.shape(batch)[0] % split_size])


def standardize_video(sequence, modality, transpose=True):
    shape = tf.shape(sequence)
    if modality == 'binary_image' and shape[-1] != 1:
        standardized = tf.transpose(tf.reshape(tf.transpose(sequence, [0, 2, 3, 1, 4]),
                                               [shape[0], shape[2], shape[3], -1, 1]),
                                    [0, 3, 1, 2, 4])
    elif modality == 'rgb_image':
        standardized = (sequence + 0.5)
        if shape[-1] != 3:
            standardized = tf.transpose(tf.reshape(tf.transpose(sequence, [0, 2, 3, 1, 4]),
                                                   [shape[0], shape[2], shape[3], -1, 3]),
                                        [0, 3, 1, 2, 4])
    else:
        standardized = sequence
    if transpose:
        return tf.transpose(standardized, [0, 1, 4, 2, 3])
    else:
        return standardized


def evaluate_model(ground_truth_sequence, model, logger, observation_type, step):
    dtype = prec.global_policy().compute_dtype
    sequence_length = min(50, len(ground_truth_sequence[0]['next_observation']) + 1)
    gt_episode = {k: tf.convert_to_tensor(v, dtype)[None, :sequence_length]
                  for k, v in ground_truth_sequence[0].items() if k in ['observation',
                                                                        'action', 'terminal',
                                                                        'reward']}
    if 'cost' in ground_truth_sequence[0]['info'][-1].keys():
        gt_episode['cost'] = tf.convert_to_tensor(list(map(
            lambda info: info['cost'], ground_truth_sequence[0]['info'])))
    gt_episode['observation'] = tf.concat(
        [gt_episode['observation'],
         tf.convert_to_tensor(ground_truth_sequence[0]['next_observation'][sequence_length],
                              dtype)
         [None, None]], 1
    )
    reconstructed_sequence = model.reconstruct_sequences_posterior(gt_episode)
    conditioning_length = sequence_length // 5
    horizon = sequence_length - conditioning_length
    beliefs = {'stochastic': reconstructed_sequence['stochastic'],
               'deterministic': reconstructed_sequence['deterministic']}
    last_belief = {k: tf.reduce_mean(v[:, :, conditioning_length - 1], 1)
                   for k, v in beliefs.items()}
    generated = model.generate_sequences_posterior(
        last_belief, horizon, actions=gt_episode['action'][:, conditioning_length:],
        log_sequences=True)
    if observation_type in ['rgb_image', 'binary_image']:
        reconstructed = tf.squeeze(reconstructed_sequence['reconstructed_observation'], 0)
        generated = tf.squeeze(generated['reconstructed_observation'], 0)
        logger.log_video(standardize_video(reconstructed, observation_type).numpy(), step,
                         "reconstructed_sequence")
        logger.log_video(standardize_video(gt_episode['observation'], observation_type).numpy(),
                         step, "true_sequence")
        logger.log_video(standardize_video(generated, observation_type).numpy(), step,
                         "generated_sequence")
    else:
        generated = tf.reduce_mean(generated['reconstructed_observation'], 1)
        predictions = generated.numpy()[0]
        targets = gt_episode['observation'][0, conditioning_length:].numpy()
        logger.log_figure(plot_prediction_ground_truth(targets, predictions),
                          step, "ground_truth_vs_generated")


def plot_prediction_ground_truth(ground_truth, prediction):
    fig = plt.figure(figsize=(11, 3.5), constrained_layout=True)
    n_plots = min(3, ground_truth.shape[-1])
    t = np.arange(0, ground_truth.shape[0] - 1)
    inputs = ground_truth[:-1]
    for n in range(n_plots):
        ax = plt.subplot(3, 1, n + 1)
        ax.plot(t, inputs[:, n], label='Observation', marker='.', zorder=-10)
        ax.scatter(t + 1, prediction[:, n], marker='X', edgecolors='k', label='Predictions',
                   c='#ff7f0e', s=64)
        if n == 0:
            fig.legend()
    return fig
