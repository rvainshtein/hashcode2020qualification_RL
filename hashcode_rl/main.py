from glob import glob

from env import LibrariesEnv, MyTimeLimit


def extract_line_data(line):
    return list(map(int, line.split()))


if __name__ == '__main__':
    data_files = glob('RL/data/qualification_round_2020.in/*.txt')
    data_file = data_files[0]  # example

    env = LibrariesEnv(data_file)
    wrapped_env = MyTimeLimit(env, max_episode_steps=env.total_days)
    pass
