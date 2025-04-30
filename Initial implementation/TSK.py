#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:42:21 2021

@author: arslan
"""

from numpy import (exp, linspace, array, meshgrid, zeros, minimum, maximum, 
                   reshape, arange, sin, cos, gradient, trapz, )
from numpy.random import (rand, )
from numpy.linalg import (norm, )
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import (LinearLocator, FormatStrFormatter, )



class Gaussian:

    def d0(self, x, c, v):
        return exp(- ((x - c) / v) ** 2)

    def d1(self, x, c, v):
        return - 2 * ((x - c) / v ** 2) * exp(- ((x - c) / v) ** 2)


class LGaussian:

    def d0(self, x, c, v):
        return 1.0 * (x >= c) + exp(- ((x - c) / v) ** 2) * (x < c)
    
    def d1(self, x, c, v):
        return 0.0 * (x >= c) - 2 * ((x - c) / v ** 2) * exp(- ((x - c) / v) ** 2) * (x < c)


class RGaussian:

    def d0(self, x, c, v):
        return 1.0 * (x <= c) + exp(- ((x - c) / v) ** 2) * (x > c)
    
    def d1(self, x, c, v):
        return 0.0 * (x <= c) - 2 * ((x - c) / v ** 2) * exp(- ((x - c) / v) ** 2) * (x > c)


rgaussian = RGaussian()
gaussian = Gaussian()
lgaussian = LGaussian()


class TSK:
    
    def __init__(self, P, TSKN, TSKM, mf, c=1.0):
        self.p = reshape(P[:-TSKM], (TSKM, TSKN, 2, ))
        self.q = P[-TSKM:]
        self.N = TSKN
        self.M = TSKM
        self.mf = mf
        self.c = c
    
    def d0(self, d0x):
        s = 1.
        d = 0.
        n = 0.
        for l in range(self.M):
            s = 1.
            for i in range(self.N):
                s *= self.mf[l][i].d0(d0x[i], self.p[l][i][0], self.p[l][i][1])
            n += self.q[l] * s
            d += s
        return self.c * n / d
    
    def d1(self, d0x, d1x):
        s1 = 0.
        s2 = 0.
        s3 = 0.
        s4 = 0.
        
        for l in range(self.M):
            s5 = 1.
            for i in range(self.N):
                s5 *= self.mf[l][i].d0(d0x[i], self.p[l][i][0], self.p[l][i][1])
            s1 += self.q[l] * s5
            s2 += s5
        
        for j in range(self.N):
            s7 = 0.
            s8 = 0.
            for l in range(self.M):
                s6 = 1.
                for i in range(self.N):
                    if i != j:
                        s6 *= self.mf[l][i].d0(d0x[i], self.p[l][i][0], self.p[l][i][1])
                
                s7 += self.q[l] * self.mf[l][j].d1(d0x[j], self.p[l][j][0], self.p[l][j][1]) * s6
                s8 += self.mf[l][j].d1(d0x[j], self.p[l][j][0], self.p[l][j][1]) * s6
            
            s3 += d1x[j] * s7
            s4 += d1x[j] * s8
        
        return (s3 * s2 - s1 * s4) / s2 ** 2


class PSO:
    
    def __init__(self, N, M, func, args=()):
        self.func = func
        self.args = args
        self.N  = N
        self.M  = M
        self.X  = 4.0 * (rand(self.N, self.M) - 0.5)
        self.V  = 8.0 * (rand(self.N, self.M) - 0.5)
        self.Xb = self.X.copy()
        self.Fb = zeros(shape=(self.N, ))
        
        self.Fb[0] = self.func(self.X[0, :], *self.args)
        
        self.xb = self.X[0, :].copy()
        self.fb = self.Fb[0]
        
        self.iterNum = 0
        
        for i in range(1, self.N):
            self.Fb[i] = self.func(self.X[i, :], *self.args)
                
            if self.Fb[i] < self.fb:
                self.fb = self.Fb[i]
                self.xb = self.X[i, :].copy()
        
        self.saveData()
                
    
    def saveData(self):
        dataFile = open(str(self.iterNum) + ".txt", "w")
        for i in range(self.N):
            for j in range(self.M):
                dataFile.write(str(self.X[i, j]) + "\n")
            
            for j in range(self.M):
                dataFile.write(str(self.V[i, j]) + "\n")
            
            for j in range(self.M):
                dataFile.write(str(self.Xb[i, j]) + "\n")
            
            dataFile.write(str(self.Fb[i]) + "\n")
        
        for j in range(self.M):
            dataFile.write(str(self.xb[j]) + "\n")
        
        dataFile.write(str(self.fb) + "\n")
        
        dataFile.close()
    
    
    def iterate(self, omega=0.3, phi_p=0.3, phi_g=2.1):
        self.iterNum += 1
        r_p = rand(self.N, self.M)
        r_g = rand(self.N, self.M)
        self.V = omega * self.V + \
                 phi_p * r_p * (self.Xb - self.X) + \
                 phi_g * r_g * (self.xb - self.X)
        self.X = self.X + self.V
        
        for i in range(self.N):
            tmp = self.func(self.X[i, :], *self.args)
                
            if tmp < self.Fb[i]:
                self.Xb[i, :] = self.X[i, :].copy()
                self.Fb[i] = tmp
                if tmp < self.fb:
                    self.xb = self.X[i, :].copy()
                    self.fb = tmp
        
        self.saveData()
   
        
def ITAE(X, t, dt):
    return trapz(norm(X, axis=1) * t, dx=dt)


def IAE(X, dt):
    return trapz(norm(X, axis=1), dx=dt)


def saveSolution(P0, error, filename="solutions.txt"):
    dataFile = open(filename, "a")
    for i in range(len(P0)):
        dataFile.write(str(P0[i]) + "\n")
    dataFile.write(str(error) + "\n")
    dataFile.close()


def stringSolution(P0):
    s = "["
    for p in P0:
        s += str(round(p, 4)) + ", "
    s += "]"
    return s

if __name__ == "__main__":
    TSKN1 = 1
    TSKM1 = 3
    P1 = [ 0.1, 0.1, # I1 is P
          #--------------------
           0.00, 0.1, # I1 is Z
          #--------------------
          -0.10, 0.1, # I1 is N
          #--------------------
          1.0, 4.0, 1.0, ]
    
    MFList = [[lgaussian, ], 
              [ gaussian, ], 
              [rgaussian, ], ]
    
    Ktsmc2 = TSK(P1, TSKN1, TSKM1, 
                 MFList,  )
    
    t = arange(-0.5, 0.5, 0.001)
    d0X = t
    d1X = (t + 1) - t
    d0 = array([Ktsmc2.d0([d0x, ]) for d0x in d0X])
    d1 = array([Ktsmc2.d1([d0x, ], [d1x, ]) for d0x, d1x in zip(d0X, d1X)])
    
    plt.figure()
    plt.plot(t, d0, label="d0")
    plt.plot(t, 1.0 + 1.75 * exp(-t ** 2. / 0.1 ** 2.), label="Fake d0")
    plt.plot(t, d1, label="d1")
    plt.plot(t, gradient(d0, 0.001), label="grad")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    # TSKN2 = 2
    # TSKM2 = 9
    # P2 = [ 0.2, 0.1, # I1 is P
    #        0.2, 0.1, # I2 is P
    #       #--------------------
    #        0.2, 0.1, # I1 is P
    #        0.0, 0.1, # I2 is Z
    #       #--------------------
    #        0.2, 0.1, # I1 is P
    #       -0.2, 0.1, # I2 is N
    #       #--------------------
    #        0.0, 0.1, # I1 is Z
    #        0.2, 0.1, # I2 is P
    #       #--------------------
    #        0.0, 0.1, # I1 is Z
    #        0.0, 0.1, # I2 is Z
    #       #--------------------
    #        0.0, 0.1, # I1 is Z
    #       -0.2, 0.1, # I2 is N
    #       #--------------------
    #       -0.2, 0.1, # I1 is N
    #        0.2, 0.1, # I2 is P
    #       #--------------------
    #       -0.2, 0.1, # I1 is N
    #        0.0, 0.1, # I2 is Z
    #       #--------------------
    #       -0.2, 0.1, # I1 is N
    #       -0.2, 0.1, # I2 is N
    #       #--------------------
    #       1.0, 0.5, 0.25, 0.5, 2.0, 0.5, 0.25, 0.5, 1.0]
    
    # MFList = [[lgaussian, lgaussian, ], 
    #           [lgaussian, gaussian, ], 
    #           [lgaussian, rgaussian, ], 
    #           [gaussian, lgaussian, ], 
    #           [gaussian, gaussian, ], 
    #           [gaussian, rgaussian, ], 
    #           [rgaussian, lgaussian, ], 
    #           [rgaussian, gaussian, ], 
    #           [rgaussian, rgaussian, ], ]
    
    # myTSK = TSK(P2, TSKN2, TSKM2, 
    #             MFList,  )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    