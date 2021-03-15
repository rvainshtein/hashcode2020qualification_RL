import gym
from gym import spaces
import numpy as np

from libarry import Library


def extract_line_data(line):
    return list(map(int, line.split()))


class UniqueAction(gym.spaces.Discrete):
    def __init__(self, n):
        super(UniqueAction, self).__init__(n)
        self.taken_actions = []

    def sample(self):
        action = self.np_random.choice(set(np.arange(self.n)) - set(self.taken_actions))
        return action


class LibrariesEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': []}

    def __init__(self, data_file):
        super(LibrariesEnv, self).__init__()
        self.data_file = data_file
        self.num_books, self.num_libraries, self.total_days, self.scores, self.libraries = \
            self.get_problem_info(self.data_file)
        self.action_space = UniqueAction(self.num_libraries)
        self.observation_space = spaces.Box(low=0, high=self.scores.sum(), shape=(self.num_libraries, self.total_days))
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
        return B, L, D, np.array(scores), libraries

    def get_initial_observation(self):
        observation = []
        for l in self.libraries:
            ordered_b_score = sorted(self.scores[l.book_ids], reverse=True)
            cumm_score = [sum(ordered_b_score[:l.max_books_scanned_per_day * (self.total_days - idx)])
                          for idx in range(self.total_days)]
            observation.append(cumm_score)
        return np.array(observation)

    def step(self, action):
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
        self.total_score += self.last_observation[action, 0]
        self.last_observation = np.roll(self.last_observation, shift=-1, axis=1)
        self.last_observation[:, -1] = 0
        self.chosen_libraries.append(action)
        self.action_space.taken_actions.append(action)
        if len(self.action_space.taken_actions) == self.num_libraries:
            return self.last_observation, self.total_score, True, dict()
        else:
            return self.last_observation, self.total_score, False, dict()

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
        self.chosen_libraries = []
        self.total_score = 0
        self.last_observation = self.get_initial_observation()
        self.action_space.taken_actions = []
        return self.last_observation

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
