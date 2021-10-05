import gym
import numpy as np
from PIL import Image
from gym import Wrapper, ObservationWrapper
from gym.spaces.box import Box
from gym.wrappers import RescaleAction

import la_mbda.utils as utils

IMAGE_CROP_ENVS = ['manipulator', 'stacker', 'pointmass']


def make_env(name, episode_length, action_repeat, seed, observation_type):
    suite, name = name.split('_', 1)
    rendered = observation_type in ['rgb_image', 'binary_image']
    if any(env in name for env in IMAGE_CROP_ENVS):
        size = (240, 320)
        crop = (12, 25, 12, 25)
    else:
        size = (64, 64)
        crop = None
    if suite == 'dmc':
        env = make_dm_env(name, episode_length)
        render_kwargs = {'height': size[1],
                         'width': size[0],
                         'camera_id': 0}
    elif suite == 'gym':
        assert not name.startswith('Safe'), 'To use safety gym envs, use the \'sgym\' prefix.'
        env = make_gym_env(name, episode_length)
        render_mode = 'rgb_array' if 'state_pixels' not in env.metadata.get('render.modes') \
            else 'state_pixels'
        render_kwargs = {'mode': render_mode}
    elif suite == 'sgym':
        env = make_safety_gym_env(name, episode_length, rendered)
        render_kwargs = {'mode': 'vision'}
    else:
        raise NotImplementedError
    env = ActionRepeat(env, action_repeat, suite == 'sgym')  # sum costs in suite is safety_gym
    env = RescaleAction(env, -1.0, 1.0)
    if rendered:
        env = RenderedObservation(env, observation_type, (64, 64), render_kwargs, crop)
    env.seed(seed)
    return env


def make_dm_env(name, episode_length):
    domain, task = name.rsplit('.', 1)
    from dm_control import suite
    env = suite.load(domain, task, environment_kwargs={'flat_observation': True})
    env = DeepMindBridge(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    return env


def make_gym_env(name, episode_length):
    env = gym.make(name)
    if not isinstance(env, gym.wrappers.TimeLimit):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    else:
        # https://github.com/openai/gym/issues/499
        env._max_episode_steps = episode_length
    return env


def make_safety_gym_env(name, episode_length, rendered):
    import safety_gym  # noqa
    env = gym.make(name)
    if not isinstance(env, gym.wrappers.TimeLimit):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    else:
        # https://github.com/openai/gym/issues/499
        env._max_episode_steps = episode_length
    # Turning manually on the 'observe_vision' flag so a rendering context gets opened and
    # all object types rendering is on (L.302, safety_gym.world.py).
    env.unwrapped.vision_size = (64, 64)
    env.unwrapped.observe_vision = rendered
    env.unwrapped.vision_render = False
    obs_vision_swap = env.unwrapped.obs_vision

    # Making rendering within obs() function (in safety_gym) not actually render the scene on
    # default so that rendering only occur upon calling to 'render()'.
    from PIL import ImageOps

    def render_obs(fake=True):
        if fake:
            return np.ones(())
        else:
            image = Image.fromarray(np.array(obs_vision_swap() * 255, dtype=np.uint8,
                                             copy=False))
            image = np.asarray(ImageOps.flip(image))
            return image

    env.unwrapped.obs_vision = render_obs

    def safety_gym_render(mode, **kwargs):
        if mode in ['human', 'rgb_array']:
            # Use regular rendering
            return env.unwrapped.render(mode, camera_id=3, **kwargs)
        elif mode == 'vision':
            return render_obs(fake=False)
        else:
            raise NotImplementedError

    env.render = safety_gym_render
    return env


# Copied from https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c
# /wrappers.py
class ActionRepeat(Wrapper):
    def __init__(self, env, repeat, sum_cost=False):
        assert repeat >= 1, 'Expects at least one repeat.'
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat
        self.sum_cost = sum_cost

    def step(self, action):
        done = False
        total_reward = 0.0
        current_step = 0
        total_cost = 0.0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            if self.sum_cost:
                total_cost += info['cost']
            total_reward += reward
            current_step += 1
        if self.sum_cost:
            info['cost'] = total_cost  # noqa
        return obs, total_reward, done, info  # noqa


class DeepMindBridge(gym.Env):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['observations']
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done, {}

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()['observations']
        return gym.spaces.Box(-np.inf, np.inf, spec.shape, dtype=spec.dtype)

    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs.keys():
            kwargs['camera_id'] = 0
        return self._env.physics.render(**kwargs)

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation['observations']
        return obs

    def seed(self, seed=None):
        self._env.task.random.seed(seed)


class RenderedObservation(ObservationWrapper):
    def __init__(self, env, observation_type, image_size, render_kwargs, crop=None):
        super(RenderedObservation, self).__init__(env)
        self._type = observation_type
        self._size = image_size
        if observation_type == 'rgb_image':
            last_dim = 3
        elif observation_type == 'binary_image':
            last_dim = 1
        else:
            raise RuntimeError("Invalid observation type")
        self.observation_space = Box(0.0, 1.0, image_size + (last_dim,), np.float32)
        self._render_kwargs = render_kwargs
        self._crop = crop

    def observation(self, _):
        image = self.env.render(**self._render_kwargs)
        image = Image.fromarray(image)
        if self._crop:
            w, h = image.size
            image = image.crop((self._crop[0], self._crop[1], w - self._crop[2], h - self._crop[3]))
        if image.size != self._size:
            image = image.resize(self._size, Image.BILINEAR)
        if self._type == 'binary_image':
            image = image.convert('L')
        image = np.array(image, copy=False)
        image = np.clip(image, 0, 255).astype(np.float32)
        bias = dict(rgb_image=0.5, binary_image=0.0).get(self._type)
        return utils.preprocess(image, bias)
