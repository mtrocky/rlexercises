from deeprl_hw3 import imitation, reinforce
import gym
import os

expert_yaml = os.path.join(os.getcwd(), 'CartPole-v0_config.yaml')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = imitation.load_model(expert_yaml)
    reinforce.reinforce(env, model)