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

# Output the estimated transition matrix and emission matrix
def solve(A, B, pi, sequence):

    # Forward algorithm
    def forward_algorithm(A, B, pi, sequence):
        N = len(A)
        T = len(sequence)
        alpha = [[0 for _ in range(T)] for _ in range(N)]

        # Initialization
        for i in range(N):
            alpha[i][0] = pi[i][0] * B[i][sequence[0]]

        # Induction
        for t in range(1, T):
            for j in range(N):
                alpha[j][t] = sum(alpha[i][t-1] * A[i][j] for i in range(N)) * B[j][sequence[t]]

        return alpha
    
    # Backward algorithm
    def backward_algorithm(A, B, pi, sequence):
        N = len(A)
        T = len(sequence)
        beta = [[0 for _ in range(T)] for _ in range(N)]

        # Initialization
        for i in range(N):
            beta[i][T-1] = 1

        # Induction
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i][t] = sum(A[i][j] * B[j][sequence[t+1]] * beta[j][t+1] for j in range(N))

        return beta
    
    # Compute the forward and backward probabilities
    alpha = forward_algorithm(A, B, pi, sequence)
    beta = backward_algorithm(A, B, pi, sequence)

    # Compute the probability of the sequence
    P = sum(alpha[i][-1] for i in range(len(A)))

    # Compute the estimated transition matrix
    estimated_transition_matrix = [[0 for _ in range(len(A))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A)):
            numerator = sum(alpha[i][t] * A[i][j] * B[j][sequence[t+1]] * beta[j][t+1] for t in range(len(sequence)-1))
            denominator = sum(alpha[i][t] * beta[i][t] for t in range(len(sequence)))
            estimated_transition_matrix[i][j] = numerator / denominator

    # Compute the estimated emission matrix
    estimated_emission_matrix = [[0 for _ in range(len(B[0]))] for _ in range(len(B))]

    for j in range(len(B)):
        for k in range(len(B[0])):
            numerator = sum(alpha[i][t] * beta[i][t] for t in range(len(sequence)) if sequence[t] == k)
            denominator = sum(alpha[i][t] * beta[i][t] for t in range(len(sequence)))
            estimated_emission_matrix[j][k] = numerator / denominator

        m1, n1 = matrix_dimensions(estimated_emission_matrix)
        m2, n2 = matrix_dimensions(B)

    return (m1, n1, estimated_transition_matrix), (m2, n2, estimated_emission_matrix)

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
    transition_matrix, emission_matrix = solve(A, B, pi, sequence)
    # PRint m n and the matrix
    print(f"{transition_matrix[0]} {transition_matrix[1]} {' '.join(str(x) for row in transition_matrix[2] for x in row)}")
    print(f"{emission_matrix[0]} {emission_matrix[1]} {' '.join(str(x) for row in emission_matrix[2] for x in row)}")