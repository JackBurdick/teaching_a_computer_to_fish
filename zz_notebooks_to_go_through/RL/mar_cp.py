import gym
import tensorflow as tf

import numpy as np
import random
import math
import time

env = gym.make("CartPole-v0")


bucket_n = (1, 1, 6, 3)
actions_n = env.action_space.n


def create_state_bounds():
    # create bounds
    state_value_bounds = list(
        zip(env.observation_space.low, env.observation_space.high)
    )

    # manual override
    state_value_bounds[1] = [-0.5, 0.5]
    state_value_bounds[3] = [-math.radians(50), math.radians(50)]
    return state_value_bounds


state_value_bounds = create_state_bounds()

action_index = len(bucket_n)

q_value_table = np.zeros(bucket_n + (actions_n,))

# hyp
HYP_DICT = {}
HYP_DICT["min_explore_rate"] = 0.01
HYP_DICT["min_learning_rate"] = 0.1
HYP_DICT["max_episodes"] = 250
HYP_DICT["max_time_steps"] = 250
HYP_DICT["streak_to_end"] = 120
HYP_DICT["solved_time"] = 199
HYP_DICT["discount"] = 0.99
HYP_DICT["streaks_n"] = 0


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_value_table[state_value])
    return action


def select_explore_rate(x):
    return max(HYP_DICT["min_explore_rate"], min(1, 1.0 - math.log10((x + 1) / 25)))


def select_learning_rate(x):
    return max(HYP_DICT["min_learning_rate"], min(0.5, 1.0 - math.log10((x + 2) / 25)))


def bucketize_state_value(state_value):
    bucket_indexes = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = bucket_n[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (bucket_n[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (bucket_n[i] - 1) / bound_width
            bucket_index = int(round(scaling * state_value[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


# train
for episode in range(HYP_DICT["max_episodes"]):

    explore_rate = select_explore_rate(episode)
    learning_rate = select_learning_rate(episode)

    observation = env.reset()

    # start episode
    print(observation)
    start_state_value = bucketize_state_value(observation)
    previous_state_value = start_state_value
    print(start_state_value)
    break

    for time_step in range(HYP_DICT["max_time_steps"]):

        selected_action = select_action(previous_state_value, explore_rate)
        observation, reward_gain, completed, _ = env.step(selected_action)
        env.render()
        # time.sleep(0.02)

        state_value = bucketize_state_value(observation)
        best_q_value = np.amax(q_value_table[state_value])

        q_value_table[previous_state_value + (selected_action,)] += learning_rate * (
            reward_gain
            + HYP_DICT["discount"] * (best_q_value)
            - q_value_table[previous_state_value + (selected_action,)]
        )

        # print("Episode number : %d" % episode)
        # print("Time step : %d" % time_step)
        # print("Selection action : %d" % selected_action)
        # print("Current state : %s" % str(state_value))
        # print("Reward obtained : %f" % reward_gain)
        # print("Best Q value : %f" % best_q_value)
        # print("Learning rate : %f" % learning_rate)
        # print("Explore rate : %f" % explore_rate)
        # print("Streak number : %d" % HYP_DICT["streaks_n"])

        if completed:
            print("Episode %d finished after %f time steps" % (episode, time_step))
            if time_step >= HYP_DICT["solved_time"]:
                HYP_DICT["streaks_n"] += 1
            else:
                HYP_DICT["streaks_n"] = 0
            env.reset()
            break

        previous_state_value = state_value

    if HYP_DICT["streaks_n"] > HYP_DICT["streak_to_end"]:
        break

env.close()
