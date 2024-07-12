import numpy as np
import scipy.integrate as integrate
import scipy
import random

def gauss_int(a, b, c, d_0, d_1): # calculating the Gaussian integral of exp(-(ax^2+bx+c)) from d_0 to d_1 (none if infinity)
    f = lambda d:scipy.special.erf((d *a + b/2)/np.sqrt(a))+1
    f_d1 = 2 if (d_1 is None) else f(d_1)
    f_d0 = 0 if (d_0 is None) else f(d_0)
    res =f_d1 - f_d0
    res *= np.sqrt(np.pi/a)/2 * np.exp(b**2/(4*a) - c)
    return res

class Filter:
    def from_name(type):
        if type == 'ckg_metropolis':
            return MetropolisFilterCKG
        if type == 'davies_metropolis':
            return MetropolisFilter
        if type == 'ckg_gaussian':
            return GaussianFilterCKG
        if type=='davies_glauber':
            return GlauberFilter

class MetropolisFilterCKG:
    def __init__(self, beta, s_e = None): 
        if s_e is None:
            s_e = 1/beta
        self.beta = beta
        self.s_e = s_e # s_e is the standard deviation of the Gaussian filter on the operator Fourier transform

    def weight(self, w): 
        s_e, beta = self.s_e, self.beta
        return np.exp(-beta * max(w + beta * s_e**2/2, 0))

    
    def bohr_coefficient(self, v_1, v_2): # given two Bohr frequencies v_1 and v_2, get the filter scaling
        s_e, beta = self.s_e, self.beta
        a = 1/(2 * s_e**2)
        b = -(v_1+v_2)/(2*s_e**2)
        c = (v_1**2+v_2**2)/(4 * s_e**2) 
        res =  gauss_int(a,b,c, None, -s_e**2*beta/2)
        b = -(v_1+v_2)/(2*s_e**2)+beta
        c = (v_1**2+v_2**2)/(4 * s_e**2) + beta**2 * s_e**2/2
        res += gauss_int(a, b,c, -s_e**2*beta/2, None)
        res *= 1/(s_e*np.sqrt(8*np.pi))
        # return 1
        # random.seed(t)
        # return random.random()
        # return np.exp(-beta**2*(v_1 - v_2)**2/8)
        return res

    def test_bohr(self, v_1, v_2): # Return error of the optimized Bohr coefficients with the manual calculation. 
        s_e, beta = self.s_e, self.beta
        p = lambda w: self.weight(w) * 1/(s_e * np.sqrt(8*np.pi)) * np.exp(-1/(4*s_e**2) * ((w-v_1)**2+(w-v_2)**2))
        out = integrate.quad(p, -np.inf, np.inf)
        return (out, self.bohr_coefficient(v_1, v_2))
        

    def coherent_coefficient(self, v_1, v_2):
        beta = self.beta
        return np.tanh(-beta * (v_1 - v_2)/4) * self.bohr_coefficient(v_1, v_2)


        
class MetropolisFilter:
    def __init__(self, beta): 
        self.beta = beta

    def weight(self, w): 
        beta = self.beta
        return np.exp(-beta * (w>=0) * w)

    def bohr_coefficient(self, v_1, v_2): # given two Bohr frequencies v_1 and v_2, get the filter scaling of the jump from v_1 to v_2
        beta = self.beta
        return (v_1 == v_2) * self.weight(v_1)


    def coherent_coefficient(self, v_1, v_2):
        beta = self.beta
        return np.tanh(-beta * (v_1 - v_2)/4) * self.bohr_coefficient(v_1, v_2)
    
    def __str__(self):
        return 'davies_metropolis'
    
class GlauberFilter:
    def __init__(self, beta): 
        self.beta = beta

    def weight(self, w): 
        beta = self.beta
        return 1/(1+np.exp(beta * w))

    def bohr_coefficient(self, v_1, v_2): # given two Bohr frequencies v_1 and v_2, get the filter scaling of the jump from v_1 to v_2
        beta = self.beta
        return (v_1 == v_2) * self.weight(v_1)


    def coherent_coefficient(self, v_1, v_2):
        beta = self.beta
        return np.tanh(-beta * (v_1 - v_2)/4) * self.bohr_coefficient(v_1, v_2)
    



class GaussianFilterCKG: # The filter is a Gaussian on the frequencies of the jump operators
    def __init__(self, beta, s_e, w_y, s_y):
        self.beta = 2 * w_y/(s_e**2 + s_y**2)
        self.s_e = s_e # s_e is the standard deviation of the Gaussian filter on the operator Fourier transform
        self.w_y = w_y # w_y is the mean frequency of the Gaussian on frequencies
        self.s_y = s_y # s_e is the standard deviation of this Gaussian
        
        assert self.beta==beta

    def bohr_coefficient(self, v_1, v_2):
        s_y, s_e, w_y = self.s_y, self.s_e, self.w_y
        res = s_y/(2*np.sqrt(s_e**2 + s_y**2))
        res *= np.exp(-(v_1 + v_2+2*w_y)**2/(8 * (s_y**2 + s_e**2)))
        res *= np.exp(-(v_1 - v_2)**2/(8 * s_e**2))
        return res

        
    def coherent_coefficient(self, v_1, v_2):
        beta = self.beta
        return np.tanh(-beta * (v_1 - v_2)/4) * self.bohr_coefficient(v_1, v_2)
    
    def from_beta(beta):
        return GaussianFilterCKG(beta, 1/beta, 1/beta, 1/beta)
    


        