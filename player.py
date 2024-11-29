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

    if scalers[0] <= 0:
        scalers[0] = epsilon  # Avoid division by zero
    scalers[0] = 1 / scalers[0]
    for i in range(N):
        alpha[0][i] *= scalers[0]

    # Recursion step
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum(alpha[t - 1][j] * A[j][i] for j in range(N)) * B[i][O[t]]
            scalers[t] += alpha[t][i]

        if scalers[t] <= 0:
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
    return -sum(math.log(max(s, epsilon)) for s in scalers)

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

def solve(A, B, pi, O):
    log_likelihood = float("-inf")
    threshold = 1e-6
    time_start = time.time()
    timeout = 0.5
    logs = []

    iter = 0
    while True:
        new_A, new_B, new_pi = reestimate(A, B, pi, O)
        new_log_likelihood = compute_log_likelihood(alpha_pass(A, B, pi, O)[1])
        logs.append(new_log_likelihood)

        if abs(new_log_likelihood - log_likelihood) < threshold or time.time() - time_start > timeout:
            break

        A, B, pi = new_A, new_B, new_pi
        log_likelihood = new_log_likelihood
        iter += 1

    return new_A, new_B, new_pi

def generate_row_stochastic_matrix(m, n):
    values = [random.random() for _ in range(m * n)]
    for i in range(m):
        row_sum = sum(values[i * n:(i + 1) * n])
        for j in range(n):
            values[i * n + j] /= row_sum
    return create_matrix(m, n, values)

class HMM:
    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.pi = generate_row_stochastic_matrix(1, N)
        self.A = generate_row_stochastic_matrix(N, N)
        self.B = generate_row_stochastic_matrix(N, K)

    def update_model(self, observations):
        self.A, self.B, self.pi = solve(self.A, self.B, self.pi, observations)

    def get_probability(self, observations):
        return compute_log_likelihood(alpha_pass(self.A, self.B, self.pi, observations)[1])

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        self.models = [HMM(2, N_EMISSIONS) for _ in range(N_SPECIES)]
        self.fishes_obs = [[] for _ in range(N_FISH)]
        self.fished_tested = [False] * N_FISH

    def guess(self, step, observations):
        for i in range(N_FISH):
            if not self.fished_tested[i]:
                self.fishes_obs[i].append(observations[i])

        if step < 110:
            return None

        fish_id = random.choice([i for i in range(N_FISH) if not self.fished_tested[i]])
        _, fish_type = max(
            (model.get_probability(self.fishes_obs[fish_id]), species)
            for species, model in enumerate(self.models)
        )
        return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        self.fished_tested[fish_id] = True
        if not correct:
            self.models[true_type].update_model(self.fishes_obs[fish_id])
