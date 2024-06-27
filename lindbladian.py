import numpy as np
import itertools
from scipy import linalg

class Lindbladian:
    def __init__(self, graph, filter):
        self.graph = graph
        self.filter = filter
        H = np.matrix(np.diag(self.graph.energies))
        self.rho = linalg.expm(-filter.beta * H) # the steady state in the energy basis
        self.rho /= self.rho.trace()
        self.M_d, self.M_c = self.lindbladian_matrix()
        self.M = self.M_d + self.M_c
        self.D = self.discriminant()

    # def lindbladian_matrix(self):
    #     return self.dissipative_matrix() + self.coherent_matrix()

    def discriminant(self):
        n = self.graph.n
        s = linalg.sqrtm(linalg.sqrtm(self.rho))
        O_t = np.zeros((n, n, n, n))
        for (l, m) in itertools.product(range(0, n), repeat = 2):
            L = np.matrix(np.zeros((n, n)))
            L[l, m] = 1
            
            O_t[:, :, l, m] = s * L * s

        O = np.matrix(O_t.reshape(n**2, n**2))
        
        P_t = np.zeros((n, n, n, n))
        for (l, m) in itertools.product(range(0, n), repeat = 2):
            L = np.matrix(np.zeros((n, n)))
            L[l, m] = 1
        
            P_t[:, :, l, m] = np.linalg.inv(s) * L * np.linalg.inv(s)
        P = np.matrix(P_t.reshape(n**2, n**2))

        sgn = lambda x: 0 if x >= 0 else 1
        v = lambda i, j: G.energies[i]- G.energies[j]
        indices = []
        for i in range(n):
            for j in range(n):
                indices.append(j * n + (i + j)%n)


        return (P * self.M * O)[np.ix_(indices, indices)]
    

    def mat_spectral_gap(M, exact = True, verbose = False, assertion = True):
        if exact:
            eig = np.linalg.eigvals(M)
        else:
            eig = np.linalg.eigvalsh(M)
        np.matrix.sort(eig)
        if verbose:
            print(np.round(np.real(eig), 10))

        
        rounded_eig = np.matrix.round(eig.real, 10)
        if assertion:
            assert rounded_eig[0].real==0
        gap = (eig[1]-eig[0]).real
        return gap
        
    def spectral_gap(self, exact = True, verbose = False, assertion = True):
        M = -self.D

        return Lindbladian.mat_spectral_gap(M, exact, verbose, assertion)

    def lindbladian_matrix(self): # Return the matrix for the dissipative part of the Lindbladian with respect to the energy basis
        n = self.graph.n
        graph, filter = self.graph, self.filter
        v = lambda i, j: graph.energies[i]- graph.energies[j]

        C_bohr = np.zeros((n, n), dtype=complex)
        C_coh = np.zeros((n, n), dtype=complex)
        
        for (j, k_1, k_2) in itertools.product(range(0, n), repeat = 3):
            decay_coeff = graph.jumps_decay(k_1, k_2, j)
            C_bohr[k_1, k_2] += filter.bohr_coefficient(v(j, k_1), v(j, k_2))*decay_coeff
            C_coh[k_1, k_2] += filter.coherent_coefficient(v(j, k_1), v(j, k_2))*decay_coeff

        
        M = np.zeros((n, n, n, n), dtype=complex)
        C = np.zeros((n, n, n, n), dtype=complex)
        
        for (a, b, l, m) in itertools.product(range(0, n), repeat = 4):
            transition_coeff = graph.jumps_transition(a, b, l, m)
            if transition_coeff:
                M[a, b, l, m] += transition_coeff * filter.bohr_coefficient(v(a, l), v(b, m))
                
            if m==b:
                M[a, b, l, m] += -1/2  * C_bohr[l, a]
                C[a, b, l, m] += -1/2  * C_coh[l, a]
            if l==a:
                M[a, b, l, m] += -1/2  *C_bohr[b, m]
                C[a, b, l, m] += 1/2* C_coh[b,m]

        D = np.matrix(M.reshape(n**2, n**2))
        M_c = np.matrix(C.reshape(n**2, n**2))
                
        return D, M_c

