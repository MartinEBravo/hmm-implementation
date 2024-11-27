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

# Output the emission probability distribution on a single line in the same matrix format, including the dimensions
def solve(A, B, pi):

    # Multiply matrices in the order: pi * A * B
    result = matrix_multiply(pi, A)
    result = matrix_multiply(result, B)

    # Dimensions of the resulting matrix
    m, n = matrix_dimensions(result)

    return m, n, result

if __name__ == "__main__":
    import sys

    # Transition matrix
    first_line = sys.stdin.readline().strip().split()
    m, n = int(first_line[0]), int(first_line[1])
    A = create_matrix(m, n, list(map(float, first_line[2:])))

    # Emission matrix
    second_line = sys.stdin.readline().strip().split()
    m, n = int(second_line[0]), int(second_line[1])
    B = create_matrix(m, n, list(map(float, second_line[2:])))

    # Initial state probability distribution
    third_line = sys.stdin.readline().strip().split()
    m, n = int(third_line[0]), int(third_line[1])
    pi = create_matrix(m, n, list(map(float, third_line[2:])))

    # Solve the problem
    m, n, matrix = solve(A, B, pi)
    print(m, n, end=" ")
    for i in range(m):
        for j in range(n):
            print(matrix[i][j], end=" ")
        print()
