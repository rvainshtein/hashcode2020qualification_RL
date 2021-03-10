from glob import glob

from gym.wrappers import TimeLimit

from env import LibrariesEnv, MyTimeLimit

if __name__ == '__main__':
    data_files = glob('data/qualification_round_2020.in/*.txt')
    data_file = data_files[0]  # example

    env = LibrariesEnv(data_file)
    env.reset()
    wrapped_env = TimeLimit(env, max_episode_steps=env.total_days)
    pass
