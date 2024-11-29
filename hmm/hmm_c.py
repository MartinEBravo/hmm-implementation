import time
import math

def create_matrix(m, n, values):
    idx = 0
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            matrix[i][j] = values[idx]
            idx += 1
    return matrix

def alpha_pass(A, B, pi, O):

    # Number of states
    N = len(A)
    # Length of the observation O
    T = len(O)

    # Initialize alpha matrix (forward probabilities)
    alpha = [[0 for _ in range(N)] for _ in range(T)]
    # Initialize scalers
    scalers = [0 for _ in range(T)]

    # Initialization step
    for i in range(N):
        alpha[0][i] = pi[0][i] * B[i][O[0]]
        scalers[0] += alpha[0][i]

    # Scale the alpha matrix
    scalers[0] = 1 / scalers[0]
    for i in range(N):
        alpha[0][i] *= scalers[0]

    # Recursion step
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum(alpha[t-1][j] * A[j][i] for j in range(N)) * B[i][O[t]]
            scalers[t] += alpha[t][i]

        # Scale the alpha matrix
        scalers[t] = 1 / scalers[t]
        for i in range(N):
            alpha[t][i] *= scalers[t]

    return alpha, scalers

def beta_pass(A, B, O, scalers):
    
    # Number of states
    N = len(A)
    # Length of the observation O
    T = len(O)

    # Initialize beta matrix (backward probabilities)
    beta = [[0 for _ in range(N)] for _ in range(T)]

    # Initialization step
    for i in range(N):
        beta[T-1][i] = scalers[T-1]

    # Recursion step
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = sum(beta[t+1][j] * B[j][O[t+1]] * A[i][j] for j in range(N))
            beta[t][i] *= scalers[t]

    return beta

def compute_log_likelihood(scalers):
    return sum(-1 * math.log(scaler) for scaler in scalers)

def gamma_pass(A, B, pi, O):
    
    # Number of states
    N = len(A)
    # Length of the observation O
    T = len(O)

    # Initialize alpha and beta matrix
    alpha, scalers = alpha_pass(A, B, pi, O)
    beta = beta_pass(A, B, O, scalers)

    # Initialize gamma and digamma matrix
    gamma = [[0 for _ in range(N)] for _ in range(T)]
    digamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T-1)]

    # Compute denominator
    denominator = sum(alpha[T-1][k] for k in range(N))

    # Compute gamma and digamma matrix
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                digamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j] / denominator
            gamma[t][i] = sum(digamma[t][i][j] for j in range(N))

    # Normalize gamma matrix
    for t in range(T-1):
        for i in range(N):
            gamma[t][i] /= sum(gamma[t][j] for j in range(N))

    return gamma, digamma

def reestimate(A, B, pi, O):

    # Number of states
    N = len(A)
    # Length of the observation O
    T = len(O)
    # Number of possible observations
    K = len(B[0])

    # Initialize new_A and new_B
    new_A = [[0 for _ in range(N)] for _ in range(N)]
    new_B = [[0 for _ in range(K)] for _ in range(N)]
    new_pi = [[0 for _ in range(N)]]

    # Initialize alpha and beta matrix
    gamma, digamma = gamma_pass(A, B, pi, O)

    # Re-estimate A, B, and pi
    for i in range(N):
        for j in range(N):
            new_A[i][j] = sum(digamma[t][i][j] for t in range(T-1)) / sum(gamma[t][i] for t in range(T-1))

    for i in range(N):
        for k in range(K):
            new_B[i][k] = sum(gamma[t][i] for t in range(T-1) if O[t] == k) / sum(gamma[t][i] for t in range(T-1))

    for i in range(N):
        new_pi[0][i] = gamma[0][i]

    return new_A, new_B, new_pi

# Output the estimated transition matrix and emission matrix
def solve(A, B, pi, O):

    log_likelihood = -1e9
    threshold = 1e-6
    time_start = time.time()
    timeout = 0.8
    logs = []

    iter = 0

    # Initialize new_A and new_B
    while True:
        new_A, new_B, new_pi = reestimate(A, B, pi, O)
        new_log_likelihood = compute_log_likelihood(alpha_pass(A, B, pi, O)[1])
        logs.append(new_log_likelihood)

        if new_log_likelihood - log_likelihood < threshold or time.time() - time_start > timeout:
            break

        A = new_A
        B = new_B
        pi = new_pi
        log_likelihood = new_log_likelihood
        iter += 1

    print("Number of iterations:", iter)
    print("Time taken:", time.time() - time_start)
    return new_A, new_B


def test():

    file_1 = "hmm_c_N1000.in"

    # Import .in file
    with open(file_1, "r") as f:
        sequence = f.readline().strip().split()
        O = [int(x) for x in sequence[1:500]]

    A = [[0.54, 0.26, 0.20], [0.19, 0.53, 0.28], [0.22, 0.18, 0.60]]
    B = [[0.5, 0.2, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45]]
    pi = [[0.3, 0.2, 0.5]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))

    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))

# Question 7 Train an HMM with the same parameter dimensions as above, i.e. A should be a 3
# times 3 matrix, etc. Initialize your algorithm with the following matrices:
# A =
# 
# 
# 0.54 0.26 0.20
# 0.19 0.53 0.28
# 0.22 0.18 0.6
# 
#  B =
# 
# 
# 0.5 0.2 0.11 0.19
# 0.22 0.28 0.23 0.27
# 0.19 0.21 0.15 0.45
# 
#  π= (0.3 0.2 0.5)
# Does the algorithm converge? How many observations do you need for the algorithm to con-
# verge? How can you define convergence?

def test_7():

    file_1 = "hmm_c_N1000.in"

    # Import .in file
    with open(file_1, "r") as f:
        sequence = f.readline().strip().split()
        O = [int(x) for x in sequence[1:500]]

    A = [[0.54, 0.26, 0.20], [0.19, 0.53, 0.28], [0.22, 0.18, 0.60]]
    B = [[0.5, 0.2, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45]]
    pi = [[0.3, 0.2, 0.5]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))

    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))

# Question 8 Train an HMM with the same parameter dimensions as above, i.e. A is a 3x3 matrix
# etc. The initialization is left up to you.
# How close do you get to the parameters above, i.e. how close do you get to the generating
# parameters in Eq. 3.1? What is the problem when it comes to estimating the distance between
# these matrices? How can you solve these issues?

def test_8():

    file_1 = "hmm_c_N1000.in"

    # Import .in file
    with open(file_1, "r") as f:
        sequence = f.readline().strip().split()
        O = [int(x) for x in sequence[1:500]]

    A = [[0.1, 0.3, 0.6], [0.3, 0.5, 0.2], [0.4, 0.1, 0.5]]
    B = [[0.1, 0.4, 0.5, 0.0], [0.6, 0.3, 0.1, 0.0], [0.0, 0.0, 0.0, 1.0]]
    pi = [[0.3, 0.2, 0.5]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))

    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))

# Question 9 Train an HMM with different numbers of hidden states.
# What happens if you use more or less than 3 hidden states? Why?
# Are three hidden states and four observations the best choice? If not, why? How can you deter-
# mine the optimal setting? How does this depend on the amount of data you have?

def test_9():
    
        file_1 = "hmm_c_N1000.in"
    
        # Import .in file
        with open(file_1, "r") as f:
            sequence = f.readline().strip().split()
            O = [int(x) for x in sequence[1:500]]
    
        A = [[0.54, 0.26, 0.20, 0.00], [0.19, 0.53, 0.28, 0.00], [0.22, 0.18, 0.60, 0.00], [0.00, 0.00, 0.00, 1.00]]
        B = [[0.5, 0.2, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45], [0.00, 0.00, 0.00, 1.00]]
        pi = [[0.3, 0.2, 0.1, 0.4]]
    
        new_A, new_B = solve(A, B, pi, O)
    
        # Output the estimated transition matrix
        for i in range(len(new_A)):
            print(" ".join(str(x) for x in new_A[i]))
    
        # Output the estimated emission matrix
        for i in range(len(new_B)):
            print(" ".join(str(x) for x in new_B[i]))

# Question 10 Initialize your Baum-Welch algorithm with a uniform distribution. How does this
# affect the learning?
# Initialize your Baum-Welch algorithm with a diagonal A matrix and π= [0, 0, 1]. How does this
# affect the learning?
# Initialize your Baum-Welch algorithm with a matrices that are close to the solution. How does
# this affect the learning?

def test_10():

    file_1 = "hmm_c_N1000.in"

    # Import .in file
    with open(file_1, "r") as f:
        sequence = f.readline().strip().split()
        O = [int(x) for x in sequence[1:500]]

    # baum-welch algorithm with uniform distribution
    print("Uniform distribution")
    A = [[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]
    B = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    pi = [[0.33, 0.33, 0.33]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))

    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))

    # baum-welch algorithm with diagonal A matrix and π= [0, 0, 1]
    print("Diagonal A matrix and π= [0, 0, 1]")
    A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    B = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    pi = [[0, 0, 1]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))

    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))

    # baum-welch algorithm with matrices that are close to the solution
    print("Matrices that are close to the solution")
    A = [[0.65, 0.25, 0.10], [0.15, 0.65, 0.20], [0.20, 0.15, 0.65]]
    B = [[0.45, 0.25, 0.15, 0.15], [0.20, 0.40, 0.20, 0.20], [0.15, 0.15, 0.35, 0.35]]

    new_A, new_B = solve(A, B, pi, O)

    # Output the estimated transition matrix
    for i in range(len(new_A)):
        print(" ".join(str(x) for x in new_A[i]))
    
    # Output the estimated emission matrix
    for i in range(len(new_B)):
        print(" ".join(str(x) for x in new_B[i]))


if __name__ == "__main__":
    import sys

    print("Test 7")
    test_7()
    print("Test 8")
    test_8()

    print("Test 10")
    test_10()


