from deeprl_hw3 import imitation
import gym
import numpy as np
import tensorflow as tf

def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    return 0.0


def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    return .5, 0


def reinforce(env, model):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """

    # comstruct graph
    opt = tf.train.AdamOptimizer()
    Gt = tf.placeholder(tf.float32, shape=[None, 1], name='Gt')
    At = tf.placeholder(tf.float32, shape=[None, 2], name='At')

    t_ = tf.squeeze(model.output)
    target = -Gt * tf.log(tf.reduce_sum(tf.multiply(t_, At), axis=[1]))
    grad_step = opt.compute_gradients(target, model.weights)
    update_weights = opt.apply_gradients(grad_step)

    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        states, actions, total_rewards = generate_episodes(env, model)
        sess.run(update_weights, feed_dict={model.input: states, Gt:total_rewards, At:actions})
        model.set_weights(sess.run(model.weights))

        # check for each 20 rounds
        if i % 20 == 0:
            imitation.test_cloned_policy(env, model, render=False)


def generate_episodes(env, model):
    """
    generate episodes of stats, actions and rewards
    :param env:
    :param model:
    :return: states, actions, rewards
    """

    states = []
    actions = []
    rewards = []
    total_rewards = []

    state = env.reset()
    is_done = False
    time_step = 0

    while not is_done:
        time_step += 1
        states.append(state)
        action_prob = model.predict(np.reshape(state, (-1, 4)))
        action = np.random.choice(np.arange(len(action_prob[0])), p=action_prob[0])
        one_hot_vec = np.zeros(2)
        one_hot_vec[action] = 1
        actions.append(one_hot_vec)
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)

    for i in range(time_step):
        total_rewards.append([np.sum(rewards[i:])])

    return np.array(states), np.array(actions), np.array(total_rewards)
