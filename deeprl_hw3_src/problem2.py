from deeprl_hw3 import imitation
import gym
import argparse
import os

expert_yaml = os.path.join(os.getcwd(), 'CartPole-v0_config.yaml')
expert_h5f = os.path.join(os.getcwd(), 'CartPole-v0_weights.h5f')


expert = imitation.load_model(expert_yaml, expert_h5f)
env = gym.make('CartPole-v0')
cmdline = argparse.ArgumentParser()
cmdline.add_argument("-e", "--episodes", dest="num_episodes", default=100, help="Number of episodes from expert")

if __name__ == '__main__':
    args = cmdline.parse_args()
    # Problem 2.
    print("===== Problem 2.1 =====")
    obz, act = imitation.generate_expert_training_data(expert, env, num_episodes=int(args.num_episodes), render=False)
    model = imitation.load_model(expert_yaml)
    imitation.behavior_cloning(model, obz, act)

    print("===== Problem 2.2 =====")
    imitation.test_cloned_policy(env, model, render=False)

    print("===== Problem 2.3 =====")
    harder_env = imitation.wrap_cartpole(env)
    print("> evaluate cloned model")
    imitation.test_cloned_policy(harder_env, model, render=False)
    print("> evaluate expert model")
    imitation.test_cloned_policy(harder_env, expert, render=False)

    print("===== DAGGER =====")
    model = imitation.load_model(expert_yaml)
    imitation.dagger(env, model, expert)

