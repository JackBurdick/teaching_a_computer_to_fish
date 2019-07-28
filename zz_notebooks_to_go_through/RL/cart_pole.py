import numpy as np

# import cPickle as pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import math

import gym

env = gym.make("CartPole-v0")

env.reset()

# Hyperparameters
HYP_DICT = {}
HYP_DICT["INPUT_DIM"] = 4  # Input dimensions
HYP_DICT["H_SIZE"] = 16  # Number of hidden layer neurons
HYP_DICT["BATCH_SIZE"] = 5  # Update Params after every 5 episodes
HYP_DICT["LR"] = 0.05  # Learning Rate
HYP_DICT["EPISODES"] = 500
GAMMA = 0.99  # Discount factor
INPUT_DIM = 4

# Initializing
tf.reset_default_graph()

# Network to define moving left or right
x_input = tf.placeholder(tf.float32, [None, HYP_DICT["INPUT_DIM"]], name="input_x")
layer_one = tf.layers.dense(x_input, units=HYP_DICT["H_SIZE"], activation=tf.nn.elu)
layer_two = tf.layers.dense(
    layer_one, units=HYP_DICT["H_SIZE"] // 2, activation=tf.nn.elu
)
layer_three = tf.layers.dense(
    layer_two, units=HYP_DICT["H_SIZE"] // 4, activation=tf.nn.elu
)
score = tf.layers.dense(layer_three, units=1)  # , activation=None)
probability = tf.nn.sigmoid(score)

# From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(
    input_y * (input_y - probability) + (1 - input_y) * (input_y + probability)
)
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)


optimizer = tf.train.AdamOptimizer(learning_rate=HYP_DICT["LR"])  # Adam optimizer
W1Grad = tf.placeholder(
    tf.float32, name="batch_grad1"
)  # Placeholders for final gradients once update happens
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = optimizer.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, drs, ys = [], [], [], []  # Arrays to store parameters till an update happens
running_reward = None
reward_sum = 0
episode_number = 1

init = tf.initialize_all_variables()

# Training
with tf.Session() as sess:
    rendering = True
    sess.run(init)
    input_initial = env.reset()  # Initial state of the environment

    # Array to store gradients for each min-batch step
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= HYP_DICT["EPISODES"]:

        if (
            reward_sum / HYP_DICT["BATCH_SIZE"] > 100 or rendering == True
        ):  # Render environment only after avg reward reaches 100
            env.render()
            rendering = True

        # Format the state for placeholder
        x = np.reshape(input_initial, [1, INPUT_DIM])

        # Run policy network
        tfprob = sess.run(probability, feed_dict={x_input: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)  # Store x
        y = 1 if action == 0 else 0
        ys.append(y)

        # take action for the state
        input_initial, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # store reward after action is taken

        if done:
            episode_number += 1
            # Stack the memory arrays to feed in session
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)

            xs, hs, drs, ys = [], [], [], []  # Reset Arrays

            # Compute the discounted reward
            discounted_epr = discount_rewards(epr)

            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get and save the gradient
            tGrad = sess.run(
                newGrads,
                feed_dict={x_input: epx, input_y: epy, advantages: discounted_epr},
            )
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # Update Params after Min-Batch number of episodes
            if episode_number % HYP_DICT["BATCH_SIZE"] == 0:
                sess.run(
                    updateGrads,
                    feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]},
                )
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Print details of the present model
                running_reward = (
                    reward_sum
                    if running_reward is None
                    else running_reward * 0.99 + reward_sum * 0.01
                )
                print(
                    "Average reward for episode {}.  Total average reward {}".format(
                        reward_sum / HYP_DICT["BATCH_SIZE"],
                        running_reward / HYP_DICT["BATCH_SIZE"],
                    )
                )

                if reward_sum / HYP_DICT["BATCH_SIZE"] > 200:
                    print("Task solved in", episode_number, "episodes")
                    break

                reward_sum = 0

            input_initial = env.reset()

print(episode_number, "Episodes completed.")

