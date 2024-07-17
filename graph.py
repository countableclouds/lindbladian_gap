import numpy as np
import itertools
from dataclasses import dataclass

@dataclass
class EigResult:
    eigenvalues: int
    eigenvectors: int

class Graph:
    def from_name(type):
        if type == "complete":
            return CompleteGraph
        if type =='cyclic':
            return CyclicGraph
        if type =='path':
            return PathGraph
        if type =='hypercube':
            return HypercubeGraph
        
    def from_adjacency(adj):
        return lambda jumps, eig=None: GenericGraph(adj, adj.shape[0], jumps, eig)
        
class GenericGraph:
    def __init__(self, M, n, jumps, eig = None): 
        ## jumps is a list of 3-tuples of the rows index of the jumps, the column 
        ## index of the jumps, and the magnitude of the jump operator. These are all
        ## 1d arrays, where the list enumerates the jump operator, and the 1d 
        ## arrays are for a given jump operator
        
        self.n = n
        self.hamiltonian = M

        self.jumps = np.array(jumps)
        
        if eig is None:
            eig = np.linalg.eigh(M)
        self.energies = np.array(eig.eigenvalues)
        self.eigenbasis = np.array(eig.eigenvectors)
        
        self.jump_coeffs= np.zeros((len(jumps), n, n), dtype=complex)
        for j, jump in enumerate(jumps):
            rows, columns, mags = jump
            a, l = np.indices((n, n))
            i = np.array(range(0, len(rows)))
            A = self.eigenbasis[rows[i][:, np.newaxis, np.newaxis], a]
            B = mags*self.eigenbasis[columns[i][:, np.newaxis, np.newaxis], l]
            jump_coeff = np.sum(A.conjugate()* B, axis=0)
            
            self.jump_coeffs[j, :, :] = jump_coeff

        

    def jumps_transition(
        self, a, b, l, m
    ):  # returns the ((a,b), (l,m)) jump coefficient of the Lindbladian, for the ith jump operator
        n = self.n
        # S = np.sum(self.eigenbasis[:, a] * # this is c_{ia}, since each column is an eigenvector
        #     self.eigenbasis[:, b] * 
        #     self.eigenbasis[:, l] * 
        #     self.eigenbasis[:, m], axis=0)
        
        S = np.sum(self.jump_coeffs[:, a, l]*self.jump_coeffs[:, b, m].conjugate(), axis=0)

        return S

    def jumps_decay(self, b, m, j):
        return self.jumps_transition(j,j,b, m)
        
class HypercubeGraph:
    def __init__(self, d, jumps):
        self.n = 2**d
        self.hamiltonian = np.matrix(HypercubeGraph.adj_matrix(d))
        self.jumps = jumps            
        self.energies = np.array(HypercubeGraph.energies(d))
        n = self.n
        
        
    def energies(d):
        energies= []
        for i in itertools.product([1, -1], repeat=d):
            energies.append(sum(i))
        return energies
    
    def eigenvectors(d):
        H = np.matrix([[1, 1], [1, -1]])
        M = np.matrix([[1, 1], [1, -1]])
        for i in range(d-1):
            M = np.kron(H, M)
        M = np.array(M)
        return 2**(-d/2) * M
            
    
    def adj_matrix(d):
        binary = lambda x: bin(x)[2:].rjust(d, '0')
        
        X = np.matrix([[0, 1], [1, 0]])
        I_2 = np.eye(2, 2)
        I = np.eye(2, 2)
        for _ in range(d-2):
            I = np.tensordot(I, I_2, axes=0)
        
        M = np.zeros((2**d, 2**d)).reshape([2]*(2*d))
        X_0 = np.tensordot(X, I, axes=0)
        M += X_0

        for i in range(1, d):
            order= list(range(2, 2*(i+1))) + [0, 1]+list(range(2*(i+1), 2*d))
            X_i =np.transpose(X_0, order)
            M += X_i

        order = list(range(0, 2*d, 2))+list(range(1, 2*d, 2))
        M = np.transpose(M, order)
        M = M.reshape(2**d, 2**d)
        
        return np.array(M)

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
        return ((a ^ m ^ l ^ b) % n == 0) * 1 / n

    def jumps_diag_decay(self, a, l, j):
        n = self.n
        return (l == a) * 1 /n

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
        return np.array([
            [1 if abs(i - j) in [1, n - 1] else 0 for j in range(n)] for i in range(n)
        ])

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
        n = self.n
        return (l == a) * 1 /n
    
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
        return np.array([[1 if abs(i - j) == 1 else 0 for j in range(n)] for i in range(n)])

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
    

