import numpy as np
import pandas as pd
import hiive.mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import gym


def gen_R_T_matrices(env):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    R = np.zeros((num_states, num_actions))
    T = np.zeros((num_actions, num_states, num_states))

    for state in range(num_states):
        for action in range(num_actions):
            for transition in env.env.P[state][action]:
                probability, next_state, reward, _ = transition
                R[state, action] = reward
                T[action, state, next_state] = probability

            T[action, state, :] /= np.sum(T[action, state, :])
    return R, T


"""
This function takes an OpenAI gym environment and a policy and returns the mean value obtained by executing that policy
"""
def get_rewards(env, policy):

    all_rewards = []
    for episode in range(1000):
        obs = env.reset()
        while True:
            action = policy[obs]
            obs, reward, done, info = env.step(action)
            if done:
                all_rewards.append(reward)
                break
    return np.mean(all_rewards)


env = gym.make("FrozenLake-v0")

R, P = gen_R_T_matrices(env)


np.random.seed(22)
# P, R = hiive.mdptoolbox.example.forest(is_sparse=False)
result_list =[]
result_dict = {}
methods = ['softmax']
# methods = ['e-greedy', 'softmax', 'thompson']
taus = [0.25]
for i in range(50000, 510000, 50000):
    for m in methods:
        for tau in taus:
            ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.95, n_iter=i, alpha=0.8, tau=tau)

            ql.run(samp_method=m)
            avg_rwd = get_rewards(env, ql.policy)
            result_dict['iterations'] = i
            result_dict['time'] = ql.time
            print(f"Tau: {tau}")
            print(f"Iterations: {i}")
            print(f"Method: {m}")
            print(f"Time: {ql.time}")
            print(f"Avg Reward: {avg_rwd}")
            print(f"Policy : {ql.policy}\n")
