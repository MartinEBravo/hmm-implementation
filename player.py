#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import time
import sys

epsilon = sys.float_info.epsilon  # Prevent division by zero or underflow

def create_matrix(m, n, values):
    idx = 0
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            matrix[i][j] = values[idx]
            idx += 1
    return matrix

def alpha_pass(A, B, pi, O):
    N = len(A)  # Number of states
    T = len(O)  # Length of observation sequence

    alpha = [[0 for _ in range(N)] for _ in range(T)]
    scalers = [0 for _ in range(T)]

    # Initialization step
    for i in range(N):
        alpha[0][i] = pi[0][i] * B[i][O[0]]
        scalers[0] += alpha[0][i]

    if scalers[0] == 0:
        scalers[0] = epsilon  # Avoid division by zero
    scalers[0] = 1 / scalers[0]
    for i in range(N):
        alpha[0][i] *= scalers[0]

    # Recursion step
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum(alpha[t - 1][j] * A[j][i] for j in range(N)) * B[i][O[t]]
            scalers[t] += alpha[t][i]

        if scalers[t] == 0:
            scalers[t] = epsilon  # Avoid division by zero
        scalers[t] = 1 / scalers[t]
        for i in range(N):
            alpha[t][i] *= scalers[t]

    return alpha, scalers

def beta_pass(A, B, O, scalers):
    N = len(A)
    T = len(O)

    beta = [[0 for _ in range(N)] for _ in range(T)]

    # Initialization step
    for i in range(N):
        beta[T - 1][i] = scalers[T - 1]

    # Recursion step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t][i] = sum(beta[t + 1][j] * A[i][j] * B[j][O[t + 1]] for j in range(N))
            beta[t][i] *= scalers[t]

    return beta

def compute_log_likelihood(scalers):
    return -sum(math.log(s + epsilon) for s in scalers)

def gamma_pass(A, B, pi, O):
    N = len(A)
    T = len(O)

    alpha, scalers = alpha_pass(A, B, pi, O)
    beta = beta_pass(A, B, O, scalers)

    gamma = [[0 for _ in range(N)] for _ in range(T)]
    digamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T - 1)]

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                digamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]
            gamma[t][i] = sum(digamma[t][i][j] for j in range(N))

    for i in range(N):
        gamma[T - 1][i] = alpha[T - 1][i]

    return gamma, digamma

def reestimate(A, B, pi, O):
    N = len(A)
    T = len(O)
    K = len(B[0])

    gamma, digamma = gamma_pass(A, B, pi, O)

    new_A = [[0 for _ in range(N)] for _ in range(N)]
    new_B = [[0 for _ in range(K)] for _ in range(N)]
    new_pi = [[0 for _ in range(N)]]

    for i in range(N):
        new_pi[0][i] = gamma[0][i]

        for j in range(N):
            numerator = sum(digamma[t][i][j] for t in range(T - 1))
            denominator = sum(gamma[t][i] for t in range(T - 1)) + epsilon
            new_A[i][j] = numerator / denominator

        for k in range(K):
            numerator = sum(gamma[t][i] for t in range(T) if O[t] == k)
            denominator = sum(gamma[t][i] for t in range(T)) + epsilon
            new_B[i][k] = numerator / denominator

    return new_A, new_B, new_pi

def forward_algorithm(A, B, pi, O):
    _, scalers = alpha_pass(A, B, pi, O)
    return compute_log_likelihood(scalers)

def viterbi(A, B, pi, O):
    N = len(A)
    T = len(O)

    delta = [[0 for _ in range(N)] for _ in range(T)]
    delta_idx = [[0 for _ in range(N)] for _ in range(T)]

    for i in range(N):
        delta[0][i] = pi[0][i] * B[i][O[0]]

    for t in range(1, T):
        for i in range(N):
            max_val, max_idx = max(
                (delta[t - 1][j] * A[j][i], j) for j in range(N))
            delta[t][i] = max_val * B[i][O[t]]
            delta_idx[t][i] = max_idx

    path = [0] * T
    path[T - 1] = max(range(N), key=lambda i: delta[T - 1][i])
    for t in range(T - 2, -1, -1):
        path[t] = delta_idx[t + 1][path[t + 1]]

    return path

def generate_row_stochastic_matrix(m, n):
    matrix = [[random.random() for _ in range(n)] for _ in range(m)]
    for i in range(m):
        row_sum = sum(matrix[i])
        for j in range(n):
            matrix[i][j] /= row_sum
    return matrix

class HMM:
    def __init__(self, n_states, n_emissions):
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.PI = generate_row_stochastic_matrix(1, n_states)
        self.A = generate_row_stochastic_matrix(n_states, n_states)
        self.B = generate_row_stochastic_matrix(n_states, n_emissions)

    def update_model(self, observations):
        self.A, self.B, self.PI = reestimate(self.A, self.B, self.PI, observations)

    def get_probability(self, observations):
        return forward_algorithm(self.A, self.B, self.PI, observations)

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        self.models = [HMM(1, N_EMISSIONS) for _ in range(N_SPECIES)]
        self.fishes_obs = [[] for _ in range(N_FISH)]
        self.fished_tested = [False] * N_FISH

    def guess(self, step, observations):
        for i in range(N_FISH):
            if not self.fished_tested[i]:
                self.fishes_obs[i].append(observations[i])

        if step < 110:
            return None

        fish_id = random.choice([i for i in range(N_FISH) if not self.fished_tested[i]])
        best_prob, fish_type = max(
            (model.get_probability(self.fishes_obs[fish_id]), species)
            for species, model in enumerate(self.models)
        )
        return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        self.fished_tested[fish_id] = True
        if not correct:
            self.models[true_type].update_model(self.fishes_obs[fish_id])
