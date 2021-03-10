import gym
from gym import spaces
import numpy as np

from libarry import Library
from main import extract_line_data


class LibrariesEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': []}

    def __init__(self, data_file):
        super(LibrariesEnv, self).__init__()
        self.is_signing = False
        self.lib_currently_signing = None
        self.days_left_signing = None
        self.total_score = 0
        self.data_file = data_file
        self.num_books, self.num_libraries, self.total_days, self.scores, self.libraries = \
            self.get_problem_info(self.data_file)
        self.action_space = spaces.Discrete(self.num_libraries)
        self.reward_range = (0, np.array(self.scores).sum())

    @staticmethod
    def get_problem_info(data_file):
        with open(data_file, 'r') as f:
            data = [l for l in f.read().split('\n') if len(l)]

        B, L, D = extract_line_data(data[0])
        scores = extract_line_data(data[1])
        libraries = []
        for lib_num in range(L):
            lib_idx = 2 + 2 * lib_num
            num_books, signup_len, num_ship = extract_line_data(data[lib_idx])
            books_in_lib = extract_line_data(data[lib_idx + 1])
            libraries.append(Library(id=lib_num,
                                     book_ids=books_in_lib,
                                     signup_days=signup_len,
                                     max_books_scanned_per_day=num_ship))
        return B, L, D, scores, libraries

    def step(self, action, current_day):
        """Run one timestep of the environment's dynamics. When end of
                episode is reached, you are responsible for calling `reset()`
                to reset this environment's state.

                Accepts an action and returns a tuple (observation, reward, done, info).

                Args:
                    action (object): an action provided by the agent

                Returns:
                    observation (object): agent's observation of the current environment
                    reward (float) : amount of reward returned after previous action
                    done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                    info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
                """
        self.update_libraries(action)
        self.total_score += np.array([lib.scan_books() for lib in self.libraries])
        observations = np.array([self.libraries,
                                 self.is_signing,
                                 self.total_days,
                                 self.days_left_signing])
        return observations, self.total_score, False, None

    def update_libraries(self, action):
        chosen_lib = self.libraries[action]
        if not self.is_signing and not chosen_lib.is_signed:
            self.days_left_signing = chosen_lib.signup_days
            self.lib_currently_signing = action
        else:
            if self.days_left_signing == 0:
                self.libraries[self.lib_currently_signing].is_signed = True
                self.days_left_signing = None
        if self.days_left_signing is not None:
            self.days_left_signing -= 1

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        _, _, _, _, self.libraries = self.get_problem_info(self.data_file)
        self.is_signing = False
        self.lib_currently_signing = None
        self.days_left_signing = None
        self.total_score = 0

    def render(self, mode=None):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """

    pass


class MyTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(MyTimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action, self._elapsed_steps)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
