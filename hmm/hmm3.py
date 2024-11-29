import time
import math

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

def test():
    test = """
    4 4 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 
    4 4 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 
    1 4 0.241896 0.266086 0.249153 0.242864 
    1000 0 1 2 3 3 0 0 1 1 1 2 2 2 3 0 0 0 1 1 1 2 3 3 0 0 0 1 1 1 2 3 3 0 1 2 3 0 1 1 1 2 3 3 0 1 2 2 3 0 0 0 1 1 2 2 3 0 1 1 2 3 0 1 2 2 2 2 3 0 0 1 2 3 0 1 1 2 3 3 3 0 0 1 1 1 1 2 2 3 3 3 0 1 2 3 3 3 3 0 1 1 2 2 3 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 3 3 3 3 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 0 1 2 3 0 1 1 1 2 3 0 1 1 2 2 2 2 2 3 0 1 1 1 2 2 2 2 3 0 0 0 0 0 1 1 1 1 2 2 3 3 0 1 2 3 3 0 0 0 0 0 0 1 1 2 2 3 0 0 1 1 1 1 1 1 2 3 3 0 0 1 1 1 2 3 0 0 1 2 3 0 1 1 2 3 3 0 0 0 1 2 3 3 3 0 1 1 1 1 2 3 3 3 3 3 3 0 1 2 2 2 2 2 2 3 0 1 1 1 2 2 3 3 3 3 0 1 2 3 0 0 0 1 1 2 2 3 0 0 0 0 0 0 0 1 2 2 2 3 3 3 3 0 0 1 2 2 2 3 3 3 0 0 1 2 2 3 0 0 0 0 1 1 1 2 3 3 3 3 3 3 3 3 0 1 2 3 0 0 1 2 3 3 3 0 0 0 0 0 1 1 1 1 2 3 0 0 0 1 2 2 3 3 0 0 0 1 1 1 1 1 2 3 3 3 3 0 1 1 1 2 2 3 0 1 2 3 3 3 3 0 0 0 0 1 2 3 3 0 1 2 2 3 3 0 0 1 1 2 3 3 0 1 2 2 3 3 3 0 0 1 1 2 3 3 3 3 0 0 1 1 2 3 3 0 1 2 3 0 1 1 2 2 3 0 1 2 3 3 0 1 1 1 2 2 2 3 3 0 0 1 1 1 1 1 2 3 3 3 0 1 1 2 2 2 2 3 3 0 0 1 2 3 0 1 1 2 2 2 2 3 0 0 1 2 2 3 0 0 0 0 0 1 1 1 2 3 0 0 1 2 3 3 0 0 0 1 2 2 2 3 3 0 0 0 1 2 2 2 2 2 3 0 1 1 2 3 0 0 1 1 1 2 2 3 0 0 0 0 1 1 1 2 2 3 0 1 1 1 2 2 2 3 3 0 0 1 2 2 3 3 3 0 1 1 2 3 0 0 0 0 0 1 2 2 2 3 3 3 0 0 0 1 2 3 0 1 1 2 3 3 3 0 1 2 2 2 3 0 0 1 1 1 1 2 3 3 0 0 0 0 1 2 3 3 3 0 0 0 1 1 2 3 0 1 1 1 1 2 2 2 2 2 2 3 0 0 0 0 1 2 2 2 2 3 0 1 2 2 3 0 1 2 3 0 1 2 3 0 0 0 1 1 2 2 3 3 0 1 1 1 1 2 2 3 3 0 1 1 1 2 2 2 3 3 3 0 1 1 2 3 3 0 1 2 3 0 0 0 0 1 2 3 0 0 0 0 0 0 1 2 2 3 3 0 0 1 2 3 0 1 2 2 3 0 0 0 1 1 2 2 2 2 2 3 3 3 3 3 0 1 2 2 3 3 3 3 3 0 0 1 1 2 2 3 0 0 1 2 2 3 3 3 0 0 0 1 2 2 2 2 3 3 0 1 2 3 0 0 1 1 1 2 2 3 0 0 1 1 2 2 2 3 3 0 0 1 1 1 1 1 2 3 3 3 0 1 2 2 2 2 3 3 3 3 3 3 0 0 0 0 0 0 1 2 3 0 0 1 1 1 2 3 0 0 1 1 2 2 2 2 3 3 3 0 1 1 2 2 2 3 3 0 0 0 0 0 0 1 2 2 3 3 0 0 0 0 0 0 1 2 3 3 3 0 1 1 1 2 2 2 2 2 3 3 3 0 1 2 2 2 3 3 3 3 0 0 0 0 1 2 3 3 3 3 3 3 0 0 1 1 1 1 2 3 0 1 2 3 0 1 1 2 3 3 3 0 0 0 0 1 1 2 3 3 3 3 0 0 1 1 1 2 2 2 2 2 2 3 3 0 0 0 1 2 3 0 0 1 1 2 2 3 3 3 3 3 0 0 1 2 2 2 2 3 0 0 1 1 1 1 1 2 3 3 0 0 1 1 1 2 3 3 3 0 0 
    """

    lines = test.strip().split("\n")
    first_line = lines[0].strip().split()
    m, n = int(first_line[0]), int(first_line[1])
    A = create_matrix(m, n, list(map(float, first_line[2:])))
    second_line = lines[1].strip().split()
    m, n = int(second_line[0]), int(second_line[1])
    B = create_matrix(m, n, list(map(float, second_line[2:])))
    third_line = lines[2].strip().split()
    m, n = int(third_line[0]), int(third_line[1])
    pi = create_matrix(m, n, list(map(float, third_line[2:])))
    fourth_line = lines[3].strip().split()
    sequence = list(map(int, fourth_line[1:]))

    transition_matrix, emission_matrix = solve(A, B, pi, sequence)
    print(transition_matrix)
    print(emission_matrix)


if __name__ == "__main__":
    import sys

    # test()

    #transition matrix
    first_line = sys.stdin.readline().strip().split()
    m, n = int(first_line[0]), int(first_line[1])
    A = create_matrix(m, n, list(map(float, first_line[2:])))

    #emission matrix
    second_line = sys.stdin.readline().strip().split()
    m, n = int(second_line[0]), int(second_line[1])
    B = create_matrix(m, n, list(map(float, second_line[2:])))

    #initial state probability distribution
    third_line = sys.stdin.readline().strip().split()
    m, n = int(third_line[0]), int(third_line[1])
    pi = create_matrix(m, n, list(map(float, third_line[2:])))

    #sequence
    fourth_line = sys.stdin.readline().strip().split()
    sequence = list(map(int, fourth_line[1:]))

    #solving the problem
    transition_matrix, emission_matrix = solve(A, B, pi, sequence)
    # Print dimensions
    print(f"{len(transition_matrix)} {len(transition_matrix[0])}", end=" ")
    for row in transition_matrix:
        print(" ".join(str(x) for x in row))
    print(f"{len(emission_matrix)} {len(emission_matrix[0])}", end=" ")
    for row in emission_matrix:
        print(" ".join(str(x) for x in row))