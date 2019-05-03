import numpy as np
import os


class Environment():

  def __init__(self, domain, params, random_state=None):

    self.extra_stochasticity = params['extra_stochasticity']
    self.nb_actions = params['nb_actions']
    self.actions = range(self.nb_actions)
    self.state_shape = params['state_shape']

    print('Loading environment with extra stochasticity: {}'.format(self.extra_stochasticity))

    if domain == 'catch':
      from catch import Catch
      self.env = Catch(params['state_shape'], basket_offset=params['paddle_size'], frame_skip=params['frame_skip'],
            rendering=params['rendering'])
    elif domain == 'atari':
      from atari import AtariEnv
      self.env = AtariEnv(frame_skip=params['frame_skip'],
              repeat_action_probability=params['repeat_action_probability'], state_shape=params['state_shape'],
              rom_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), params['rom_path']),
              game_name=params['game_name'], rendering=params['test'], random_state=random_state)
    elif domain == 'dummy':
      from environments.dummy_env import dummy_env
      self.env = dummy_env(time_step = params['dummy_time_step'], size = params['dummy_size'], a_max = params['dummy_a_max'],
              v_max = params['dummy_v_max'], noise = params['dummy_noise'], noise_v = params['dummy_noise_v'],
              log = params['dummy_log'], episode_max_len=params['episode_max_len'], seed=params['seed'],
              noise_factor=params['noise_factor'])
    elif domain == 'gym':
      import gym
      self.env = gym.make(params['game_name'])
      if self.env.action_space.shape == (1,):
        self.actions = [self.env.action_space.low + i * (self.env.action_space.high - self.env.action_space.low) / (self.nb_actions - 1) for i in range(self.nb_actions)]
      self.env.seed(params['seed'])
    else:
      raise ValueError('domain must be either catch or atari')

  def step(self, action):
    if np.random.binomial(1, self.extra_stochasticity):
      played_action = np.random.randint(self.nb_actions)
    else:
      played_action = action
    return self.env.step(self.actions[played_action])

  def reset(self):
    return self.env.reset()