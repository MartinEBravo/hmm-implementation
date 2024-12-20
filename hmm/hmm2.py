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

def solve_viterbi(A, B, pi, O):
    N = len(A)  #number of states
    K = len(O)  #length of the observation O

    # initialize delta
    delta = [[0 for _ in range(N)] for _ in range(K)]
    delta_idx = [[0 for _ in range(N)] for _ in range(K)]

    # initialization step
    for i in range(N):
        delta[0][i] = B[i][O[0]] * pi[0][i]
        delta_idx[0][i] = 0

    # recursion step
    for t in range(1, K):
        for i in range(N):
            delta[t][i] = max(A[j][i] * delta[t-1][j] * B[i][O[t]] for j in range(N))
            delta_idx[t][i] = max(range(N), key=lambda j: A[j][i] * delta[t-1][j] * B[i][O[t]])
    
    # termination step
    path = [max(range(N), key=lambda i: delta[K-1][i])]
    for t in range(K-1, 0, -1):
        path.insert(0, delta_idx[t][path[0]])

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