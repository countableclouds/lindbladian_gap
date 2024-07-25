import numpy as np
import itertools
from scipy import linalg
import time

def sparsify_jump(jump):
    nonzero = np.nonzero(jump)
    mag = jump[nonzero]
    rows = nonzero[0]
    columns =  nonzero[1]
    return rows, columns, mag
    

class Lindbladian:
    def __init__(self, graph, filter):
        self.graph = graph
        self.filter = filter
        H = np.matrix(np.diag(self.graph.energies))
        self.rho = linalg.expm(-filter.beta * H)  # the steady state in the energy basis
        self.rho /= self.rho.trace()

    def initialize(self):
        self.M_d, self.M_c = self.lindbladian_matrix()
        self.M = self.M_d + self.M_c
        self.discriminant()
        
    def discriminant(self):
        n = self.graph.n
        s = np.diag(self.rho)**(1/4)
        a, b, l, m = np.indices((n, n, n, n))
        rescale_factor = np.array((s[a]**(-1) * s[b]**(-1) * s[l] * s[m]).reshape(n**2, n**2))
        self.D = np.multiply(rescale_factor, self.M)        

    def cyclic_reshape(self):
        n = self.graph.n
        indices = []
        for i in range(n):
            for j in range(n):
                indices.append(j * n + (i + j) % n)
                
        self.D = self.D[np.ix_(indices, indices)]
        self.M_d = self.M_d[np.ix_(indices, indices)]
        self.M_c = self.M_c[np.ix_(indices, indices)]
        self.M = self.M[np.ix_(indices, indices)]

    def mat_spectral_gap(M,  nullity, assertion=True, eigs=False):
        nullity = int(nullity)
        assert np.round(abs(M - M.transpose().conjugate()).max(), 10)==0
        eig = np.linalg.eigvalsh(M)
        
        np.matrix.sort(eig)
        if eigs:
            return np.round(np.real(eig), 10)
        rounded_eig = np.matrix.round(eig.real, 10)
        if assertion:
            if nullity!=0:
                assert rounded_eig[nullity-1]==0
        gap = eig[nullity].real
        
        return gap

    def spectral_gap(self, exact=True, nullity = 1, assertion=True, eigs=False):
        M = -self.D
        gap = Lindbladian.mat_spectral_gap(M, nullity, assertion, eigs)
        return gap

    def v(self, i, j):
        return self.graph.energies[i] - self.graph.energies[j]
    
    def lindbladian_matrix(
        self,
    ):  
        n = self.graph.n
        graph, filter = self.graph, self.filter
        v = self.v

        C_bohr = np.zeros((n, n), dtype=complex)
        C_coh = np.zeros((n, n), dtype=complex)

        for j, k_1, k_2 in itertools.product(range(0, n), repeat=3):
            decay_coeff = graph.jumps_decay(k_1, k_2, j)
            C_bohr[k_1, k_2] += (
                filter.bohr_coefficient(v(j, k_1), v(j, k_2)) * decay_coeff
            )
            C_coh[k_1, k_2] += (
                filter.coherent_coefficient(v(j, k_1), v(j, k_2))* decay_coeff
            )

        M = np.zeros((n, n, n, n), dtype=complex)
        C = np.zeros((n, n, n, n), dtype=complex)
        
        a, b, l, m = np.indices((n, n, n, n))
        M += graph.jumps_transition(a, b, l, m) * filter.bohr_coefficient(v(a, l), v(b, m))

        for a, b, l in itertools.product(range(0, n), repeat=3):
            m = b
            M[a, b, l, m] += -1 / 2 * C_bohr[l, a]
            C[a, b, l, m] += -1 / 2 * C_coh[l, a]
                    
        for a, b, m in itertools.product(range(0, n), repeat=3):
            l = a
            M[a, b, l, m] += -1 / 2 * C_bohr[b, m]
            C[a, b, l, m] += 1 / 2 * C_coh[b, m]

        D = np.matrix(M.reshape(n**2, n**2))
        M_c = np.matrix(C.reshape(n**2, n**2))
        
        return D, M_c
    
    def block(
        self, indices
    ):  # Return the matrix for the dissipative part of the Lindbladian with respect to the energy basis
        n = self.graph.n
        assert len(indices)==n
        graph, filter = self.graph, self.filter
        v = self.v
        # v = lambda j, k: j-k

        C_bohr = np.zeros((n,), dtype=complex)
        C_coh = np.zeros((n,), dtype=complex)

        j, k = np.indices((n, n))
        decay_coeff = graph.jumps_decay(k, k, j)
        C_bohr += (filter.bohr_coefficient(v(j, k), v(j, k)) * decay_coeff).sum(axis=0)
        C_coh += (filter.coherent_coefficient(v(j, k), v(j, k))* decay_coeff).sum(axis=0)

        M = np.zeros((n, n), dtype=complex)
        C = np.zeros((n, n), dtype=complex)
        
        i, j = np.indices((n, n))
        a, b = indices[i, 0], indices[i, 1]
        l, m = indices[j, 0], indices[j, 1]
        M += graph.jumps_transition(a, b, l, m)* filter.bohr_coefficient(v(a, l), v(b, m))

        i = np.indices((n,))[0]
        a, b = indices[i, 0], indices[i, 1]
        M += np.diag(-1 / 2 * (C_bohr[a]+C_bohr[b]))
        C += np.diag(-1 / 2 * (C_coh[a]-C_coh[b]))
        
        M= M+C
        
        n = self.graph.n
        s = np.diag(self.rho)**(1/4)
        i, j = np.indices((n, n))
        a, b = indices[i, 0], indices[i, 1]
        l, m = indices[j, 0], indices[j, 1]
        
        rescale_factor = np.array((s[a]**(-1) * s[b]**(-1) * s[l] * s[m]).reshape(n, n))
        D = np.multiply(rescale_factor, M)
        
        return M, D
