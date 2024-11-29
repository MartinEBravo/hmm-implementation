def matrix_multiply(A, B):
    m1 = len(A)
    n1 = len(A[0])
    m2 = len(B)
    n2 = len(B[0])

    if n1 != m2:
        raise ValueError(f"Cannot multiply matrices of dimensions {m1}x{n1} and {m2}x{n2}")
    
    result = [[0 for _ in range(n2)] for _ in range(m1)]

    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                result[i][j] += A[i][k] * B[k][j]

    return result

def matrix_dimensions(matrix):
    return len(matrix), len(matrix[0])

def create_matrix(m, n, values):
    idx = 0
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            matrix[i][j] = values[idx]
            idx += 1
    return matrix

def solve_viterbi(A, B, pi, sequence):
    num_states = len(A)  #number of states
    num_observations = len(sequence)  #length of the observation sequence

    #initialize delta and psi matrices
    delta = [[0 for _ in range(num_states)] for _ in range(num_observations)]
    psi = [[0 for _ in range(num_states)] for _ in range(num_observations)]

    #initialization
    for i in range(num_states):
        delta[0][i] = pi[0][i] * B[i][sequence[0]]
        psi[0][i] = 0

    #recursion
    for t in range(1, num_observations):
        for j in range(num_states):
            max_val = max(delta[t-1][i] * A[i][j] for i in range(num_states))
            delta[t][j] = max_val * B[j][sequence[t]]
            psi[t][j] = max(range(num_states), key=lambda i: delta[t-1][i] * A[i][j])

    #finalization
    last_state = max(range(num_states), key=lambda i: delta[num_observations-1][i])

    # Path backtracking
    path = [0] * num_observations
    path[num_observations-1] = last_state
    for t in range(num_observations-2, -1, -1):
        path[t] = psi[t+1][path[t+1]]

    return path

if __name__ == "__main__":
    import sys

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
    path = solve_viterbi(A, B, pi, sequence)
    print(" ".join(map(str, path)))