from deeprl_hw3 import imitation
import gym
import argparse


expert = imitation.load_model('CartPole-v0_config.ymal', 'CartPole-v0_weights.h5f')
env = gym.make('CartPole-v0')
cmdline = argparse.ArgumentParser()
cmdline.add_argument("-e", "--episodes", dest="num_episodes", default=100, help="Number of episodes from expert")

if __name__ == '__main__':
    args = cmdline.parse_args()
    # Problem 2.1
    obz, act = imitation.generate_expert_training_data(expert, env, num_episodes=int(args.num_episodes), render=False)
    model = imitation.load_model('CartPole-V0_config.yaml')
    imitation.behavior_cloning(model, obz, act)

