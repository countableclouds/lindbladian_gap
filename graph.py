import numpy as np
import itertools

class Graph:
    def from_name(type, adj):
        if type == "complete":
            return CompleteGraph
        if type =='cyclic':
            return CyclicGraph
        if type =='path':
            return PathGraph
        if type =='hypercube':
            return HypercubeGraph
        else:
            return lambda n, jumps: GenericGraph(adj, n, jumps)
        
class GenericGraph:
    def __init__(self, M, n, jumps):
        self.n = n
        self.hamiltonian = M

        self.jumps = np.array(jumps)
        self.energies = np.array(eig.eigenvalues)
        self.eigenbasis = np.array(eig.eigenvectors)
        jump_nonzeros = np.array([self.jumps[np.nonzero(self.jumps)]]).transpose()
        self.jump_rows = np.sqrt(jump_nonzeros) * self.eigenbasis[np.nonzero(self.jumps)[0], :]
        self.jump_columns = np.sqrt(jump_nonzeros) * self.eigenbasis[np.nonzero(self.jumps)[1], :]

    def jumps_transition(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        S = np.sum(self.jump_rows[:, a].conjugate() * # this is c_{ia}, since each column is an eigenvector
                 self.jump_rows[:, b] * 
                 self.jump_columns[:, l] * 
                 self.jump_columns[:, m].conjugate(), axis=0)

        return S

    def jumps_decay(self, b, m, j):
        return self.jumps_transition(j,j,b, m)
        
class HypercubeGraph:
    def __init__(self, n, jumps):
        self.n = 2**n
        self.hamiltonian = np.matrix(HypercubeGraph.adj_matrix(n))
        self.jumps = jumps
        energies= []
            
        for i in itertools.product([-1, 1], repeat=n):
            energies.append(sum(i))
            
        self.energies = np.array(energies)
        
    def adj_matrix(n):
        binary = lambda x: bin(x)[2:].rjust(n, '0')
        return [
            [1 if sum ( binary(i)[k]!= binary(j)[k] for k in range(n) )==1 else 0 for j in range(2**n)] for i in range(2**n)]

    def jumps_transition(self, a, b, l, m):
        if self.jumps == "diagonal":
            return self.jumps_diag_trans(a, b, l, m)

    def jumps_decay(self, a, l, j):
        if self.jumps == "diagonal":
            return self.jumps_diag_decay(a, l, j)

    def jumps_diag_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        return ((a + m - l - b) % n == 0) * 1 / n

    def jumps_diag_decay(self, a, l, j):
        coeff = 0
        n = self.n
        if l == a:
            coeff = 1 /n

        return coeff

class CompleteGraph:
    def __init__(self, n, jumps):
        self.n = n
        self.hamiltonian = np.matrix(CyclicGraph.adj_matrix(n))
        self.jumps = jumps
        self.energies = np.array([n-1]+[-1]*(n-1))

    def adj_matrix(n):
        return [
            [1 if i!=j else 0 for j in range(n)] for i in range(n)
        ]

    def jumps_transition(self, a, b, l, m):
        if self.jumps == "diagonal":
            return self.jumps_diag_trans(a, b, l, m)
        if self.jumps == "adjacent":
            return self.jumps_adjacent_trans(a, b, l, m)

    def jumps_decay(self, a, l, j):
        if self.jumps == "diagonal":
            return self.jumps_diag_decay(a, l, j)
        if self.jumps == "adjacent":
            return self.jumps_adjacent_decay(a, l, j)

    def jumps_diag_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        return ((a + m - l - b) % n == 0) * 1 / n

    def jumps_diag_decay(self, a, l, j):
        coeff = 0
        if l == a:
            coeff = 1 / self.n

        return coeff

    def jumps_adjacent_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        z_n = np.exp(complex(0, 2 * np.pi / n))

        return ((a + m - l - b) % n == 0) * (z_n ** (l - m) + z_n ** (m - l)) / (2 * n)

    def jumps_adjacent_decay(self, a, l, j):
        coeff = 0
        if l == a:
            coeff = 1 / self.n
        return coeff


class CyclicGraph:
    def __init__(self, n, jumps):
        self.n = n
        self.hamiltonian = np.matrix(CyclicGraph.adj_matrix(n))
        self.jumps = jumps
        self.energies = np.array([2 * np.cos(2 * np.pi * i / n) for i in range(0, n)])

    def adj_matrix(n):
        if n==2:
            return [[0, 2], [2, 0]]
        return [
            [1 if abs(i - j) in [1, n - 1] else 0 for j in range(n)] for i in range(n)
        ]

    def jumps_transition(self, a, b, l, m):
        if self.jumps == "diagonal":
            return self.jumps_diag_trans(a, b, l, m)
        if self.jumps == "adjacent":
            return self.jumps_adjacent_trans(a, b, l, m)

    def jumps_decay(self, a, l, j):
        if self.jumps == "diagonal":
            return self.jumps_diag_decay(a, l, j)
        if self.jumps == "adjacent":
            return self.jumps_adjacent_decay(a, l, j)

    def jumps_diag_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        return (((a + m - l - b) % n) == 0) * 1 / n

    def jumps_diag_decay(self, a, l, j):
        coeff = 0
        if l == a:
            coeff = 1 / self.n 

        return coeff

    def jumps_adjacent_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps to adjacent vertices
        n = self.n
        z_n = np.exp(complex(0, 2 * np.pi / n))

        return ((a + m - l - b) % n == 0) * (z_n ** (m - l) + z_n ** (l - m)) / (2 * n)
    
    def jumps_adjacent_decay(self, b, m, j):
        return self.jumps_adjacent_trans(j, j, b, m)



class PathGraph:
    def __init__(self, n, jumps):
        self.n = n
        self.hamiltonian = np.matrix(PathGraph.adj_matrix(n))
        self.jumps = jumps
        self.adj_matrix = PathGraph.adj_matrix(n)
        self.energies = np.array(
            [2 * np.cos(np.pi * i / (n + 1)) for i in range(1, n + 1)]
        )

    def adj_matrix(n):
        return [[1 if abs(i - j) == 1 else 0 for j in range(n)] for i in range(n)]

    def jumps_transition(self, a, b, l, m):
        if self.jumps == "diagonal":
            return self.jumps_diag_trans(a, b, l, m)

    def jumps_decay(self, a, l, j):
        if self.jumps == "diagonal":
            return self.jumps_diag_decay(a, l, j)

    def first_sum(
        self, j, k, l, m
    ):  # computes the sum from i = 1 to n of sin(ij*pi/(n+1))*sin(ik*pi/(n+1))*sin(il*pi/(n+1))*sin(im*pi/(n+1))
        n = self.n
        total = 0
        j_ = j + 1
        k_ = k + 1
        l_ = l + 1
        m_ = m + 1
        for p2, p3 in itertools.product(range(0, 2), repeat=2):
            if (j_ + (-1) ** p2 * k_ + (-1) ** p3 * m_) % (2 * n + 2) == l_:
                total += (-1) ** ((p2, p3).count(1) + 1) * 2
            if (j_ + (-1) ** p2 * k_ + (-1) ** p3 * m_ + l_) % (2 * n + 2) == 0:
                total += (-1) ** ((p2, p3).count(1)) * 2
                
        # if (j-l-k+m)%n==0:
        #     total = 4
        # else:
        #     total = 0
        return total * (n + 1) / 16

    def first_sum_test(self, j, k, l, m):
        n = self.n
        j_ = j + 1
        k_ = k + 1
        l_ = l + 1
        m_ = m + 1
        h = lambda n, o, p: np.sin(np.pi * o * p / (n + 1))
        out = (
            round(
                16
                / (n + 1)
                * sum(
                    [
                        h(n, i, j_) * h(n, i, k_) * h(n, i, l_) * h(n, i, m_)
                        for i in range(1, n + 1)
                    ]
                )
            )
            * (n + 1)
            / 16
        )
        assert out == self.first_sum(j, k, l, m)

    def second_sum(
        self, j, k, l
    ):  # computes the sum from i = 1 to n of sin(ij*pi/(n+1))^2*sin(ik*pi/(n+1))*sin(il*pi/(n+1))
        n = self.n
        j_ = j + 1
        k_ = k + 1
        l_ = l + 1
        total = 0
        if k_ == l_:
            total += 4
        if k_ + l_ == 2 * j_:
            total += 2
        if 2 * j_ + k_ + l_ == 2 * n + 2:
            total += 2
        if (2 * j_ - k_ + l_) % (2 * n + 2) == 0:
            total -= 2
        if (2 * j_ + k_ - l_) % (2 * n + 2) == 0:
            total -= 2
        # if k==l:
        #     total = 4
        # else:
        #     total = 0
        return total * (n + 1) / 16

    def second_sum_test(self, j, k, l):
        j_ = j + 1
        k_ = k + 1
        l_ = l + 1
        n = self.n
        h = lambda n, o, p: np.sin(np.pi * o * p / (n + 1))
        out = (
            round(
                16
                / (n + 1)
                * sum(
                    [
                        h(n, i, j_) ** 2 * h(n, i, k_) * h(n, i, l_)
                        for i in range(1, n + 1)
                    ]
                )
            )
            * (n + 1)
            / 16
        )
        assert out == self.second_sum(j, k, l), (out, self.second_sum(j, k, l), j, k, l)

    def jumps_diag_trans(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps |i><i|
        coeff = self.first_sum(a, b, l, m) * (2 / (self.n + 1)) ** 2
        return coeff

    def jumps_diag_decay(
        self, a, l, j
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, using jumps |i><i|
        coeff = self.second_sum(j, a, l) * (2 / (self.n + 1)) ** 2
        return coeff
    

