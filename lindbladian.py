import numpy as np
import itertools
from scipy import linalg


class Lindbladian:
    def __init__(self, graph, filter):
        self.graph = graph
        self.filter = filter
        H = np.matrix(np.diag(self.graph.energies))
        self.rho = linalg.expm(-filter.beta * H)  # the steady state in the energy basis
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
        for l, m in itertools.product(range(0, n), repeat=2):
            L = np.matrix(np.zeros((n, n)))
            L[l, m] = 1

            O_t[:, :, l, m] = s * L * s

        O = np.matrix(O_t.reshape(n**2, n**2))

        P_t = np.zeros((n, n, n, n))
        for l, m in itertools.product(range(0, n), repeat=2):
            L = np.matrix(np.zeros((n, n)))
            L[l, m] = 1
            t = np.linalg.inv(s)
            P_t[:, :, l, m] = t * L * t
        P = np.matrix(P_t.reshape(n**2, n**2))

        def sgn(x):
            return 0 if x >= 0 else 1

        indices = []
        for i in range(n):
            for j in range(n):
                indices.append(j * n + (i + j) % n)

        return (P * self.M * O)[np.ix_(indices, indices)]

    def mat_spectral_gap(M, exact=True, eigs=False, assertion=True):
        if exact:
            eig = np.linalg.eigvals(M)
        else:
            eig = np.linalg.eigvalsh(M)
        np.matrix.sort(eig)
        if eigs:
            return np.round(np.real(eig), 10)

        rounded_eig = np.matrix.round(eig.real, 10)
        if assertion:
            assert rounded_eig[0].real == 0
        gap = (eig[1] - eig[0]).real
        return gap

    def spectral_gap(self, exact=True, eigs=False, assertion=True):
        M = -self.D

        return Lindbladian.mat_spectral_gap(M, exact, eigs, assertion)

    def lindbladian_matrix(
        self,
    ):  # Return the matrix for the dissipative part of the Lindbladian with respect to the energy basis
        n = self.graph.n
        graph, filter = self.graph, self.filter

        def v(i, j):
            return graph.energies[i] - graph.energies[j]

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
        
        indices = []
        for i in range(n):
            for j in range(n):
                indices.append(j * n + (i + j) % n)
        # n = 4
        D = D[np.ix_(indices, indices)]
        M_c = M_c[np.ix_(indices, indices)]

        return D, M_c
