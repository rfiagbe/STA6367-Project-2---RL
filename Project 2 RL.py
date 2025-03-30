#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 10:28:13 2025

@author: roland
"""

import os

# Set the directory path (replace with your desired path)
directory = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 2 RL'

# Change the working directory
os.chdir(directory)

# Verify that the working directory has changed
print("Current working directory:", os.getcwd())


#import sys
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from lake_envs import *



# complete policy evaluation function
def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)

    while True:
        delta = 0
        for s in range(nS):
            v = 0
            a = policy[s]
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * value_function[next_state] * (not done))
            delta = max(delta, np.abs(v - value_function[s]))
            value_function[s] = v

        if delta < tol:
            break

    return value_function


## Complete Policy improvement function
def policy_improvement(P, nS, nA, value_from_policy, gamma):
    new_policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in P[s][a]:
                q_sa[a] += prob * (reward + gamma * value_from_policy[next_state] * (not done))
        new_policy[s] = np.argmax(q_sa)
    return new_policy



## Complete policy iteration function
def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    policy = np.zeros(nS, dtype=int)
    while True:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, gamma)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return value_function, policy



## Complete value iteration function
def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    while True:
        delta = 0
        for s in range(nS):
            A = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in P[s][a]:
                    A[a] += prob * (reward + gamma * value_function[next_state] * (not done))
            best_action_value = np.max(A)

            delta = max(delta, np.abs(best_action_value - value_function[s]))
            value_function[s] = best_action_value

        if delta < tol:
            break

    for s in range(nS):
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in P[s][a]:
                A[a] += prob * (reward + gamma * value_function[next_state] * (not done))
        best_action = np.argmax(A)
        policy[s] = best_action

    return value_function, policy



## Function to visualize the performance of policy
def render_single(env, policy, max_steps=100):
    episode_reward = 0
    ob, _ = env.reset()
    ob = int(ob)
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        ob = int(ob)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False) 

    nS = env.observation_space.n
    nA = env.action_space.n

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.unwrapped.P, nS, nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.unwrapped.P, nS, nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)



if __name__ == "__main__":
    deterministic_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    stochastic_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    deterministic_nS = deterministic_env.observation_space.n
    deterministic_nA = deterministic_env.action_space.n
    stochastic_nS = stochastic_env.observation_space.n
    stochastic_nA = stochastic_env.action_space.n

    print("\n" + "-" * 25 + "\nDeterministic FrozenLake-v1\n" + "-" * 25)
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi_det, p_pi_det = policy_iteration(
        deterministic_env.unwrapped.P, deterministic_nS, deterministic_nA, gamma=0.9, tol=1e-3
    )
    print("Optimal Value Function (Policy Iteration):")
    print(V_pi_det)
    print("Optimal Policy (Policy Iteration):")
    print(p_pi_det)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi_det, p_vi_det = value_iteration(
        deterministic_env.unwrapped.P, deterministic_nS, deterministic_nA, gamma=0.9, tol=1e-3
    )
    print("Optimal Value Function (Value Iteration):")
    print(V_vi_det)
    print("Optimal Policy (Value Iteration):")
    print(p_vi_det)

    print("\n" + "-" * 25 + "\nStochastic FrozenLake-v1\n" + "-" * 25)
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi_sto, p_pi_sto = policy_iteration(
        stochastic_env.unwrapped.P, stochastic_nS, stochastic_nA, gamma=0.9, tol=1e-3
    )
    print("Optimal Value Function (Policy Iteration):")
    print(V_pi_sto)
    print("Optimal Policy (Policy Iteration):")
    print(p_pi_sto)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi_sto, p_vi_sto = value_iteration(
        stochastic_env.unwrapped.P, stochastic_nS, stochastic_nA, gamma=0.9, tol=1e-3
    )
    print("Optimal Value Function (Value Iteration):")
    print(V_vi_sto)
    print("Optimal Policy (Value Iteration):")
    print(p_vi_sto)




np.set_printoptions(precision=3)

def draw_grid_environment(environment, title):
    env_unwrapped = environment.unwrapped  # Get the raw environment
    grid_size = env_unwrapped.desc.shape[0]  # Assume square grid
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))

    ax.matshow(np.zeros((grid_size, grid_size)), cmap='Wistia')
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    for y in range(grid_size):
        for x in range(grid_size):
            char = env_unwrapped.desc[y, x].decode('utf-8')
            color = 'green' if char == 'G' else 'blue' if char == 'S' else 'white'
            ax.text(x, y, char, ha='center', va='center', color=color, fontsize=20, weight='bold')

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def illustrate_agent_path(environment, path):
    env_unwrapped = environment.unwrapped  # Get the base environment
    nrows, ncols = env_unwrapped.desc.shape
    trajectory_img = np.zeros((nrows, ncols))
    plt.imshow(trajectory_img, cmap='Greys', origin='upper', extent=(0, ncols, 0, nrows))

    for r in range(nrows):
        for c in range(ncols):
            if env_unwrapped.desc[r, c] == b'H':
                plt.gca().add_patch(plt.Rectangle((c, r), 1, 1, color='darkred'))
            if env_unwrapped.desc[r, c] == b'G':
                plt.text(c + 0.5, r + 0.5, 'G', color='gold', ha='center', va='center', fontsize=14)

    for index, (state, action, _, next_state, done) in enumerate(path):
        r, c = state // ncols, state % ncols
        nr, nc = next_state // ncols, next_state % ncols
        plt.plot([c + 0.5, nc + 0.5], [r + 0.5, nr + 0.5], 'y', linewidth=2)
        if not done:
            plt.text(c + 0.5, r + 0.5, str(index), color='lime', ha='center', va='center', fontsize=10)

    plt.title('Path of the Agent', fontsize=16)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()


def show_policy_execution(environment, policy, max_steps=100):
    episode_reward = 0
    state = environment.reset()
    state = state[0] if isinstance(state, tuple) else state
    path_taken = []

    for _ in range(max_steps):
        action = policy[state]
        new_state, reward, finished, *_ = environment.step(action)
        new_state = new_state[0] if isinstance(new_state, tuple) else new_state
        path_taken.append((state, action, reward, new_state, finished))
        episode_reward += reward
        state = new_state
        if finished:
            break

    illustrate_agent_path(environment, path_taken)
    print("Total reward received: ", episode_reward)



##===================== Main execution =============================
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)

    nS = env.observation_space.n
    nA = env.action_space.n

    # Running Policy Iteration
    print("\n" + "-" * 25 + "\nRunning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env.unwrapped.P, nS, nA, gamma=0.9, tol=1e-3)
    draw_grid_environment(env, title="Optimal Policy Grid - Policy Iteration")
    show_policy_execution(env, p_pi, max_steps=100)

    # Running Value Iteration
    print("\n" + "-" * 25 + "\nRunning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.unwrapped.P, nS, nA, gamma=0.9, tol=1e-3)
    draw_grid_environment(env, title="Optimal Policy Grid - Value Iteration")
    show_policy_execution(env, p_vi, max_steps=100)