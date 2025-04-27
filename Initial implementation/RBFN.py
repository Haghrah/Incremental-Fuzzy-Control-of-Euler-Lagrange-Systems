#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:42:21 2021

@author: arslan
"""

from numpy import (exp, zeros, 
                   reshape, trapz, tensordot, )
from numpy.random import (rand, )
from numpy.linalg import (norm, )



class RBFN:
    
    def __init__(self, N, M, P):
        self.N = N
        self.M = M
        self.P = reshape(P[1:], (N, M + 2))
        self.w0 = P[0]
        self.w  = self.P[:, 0]
        self.u  = self.P[:, 1:M + 1]
        self.sigma2 = self.P[:, M + 1] ** 2
        
    def d0(self, d0x):
        return self.w0 + sum(self.w * exp(- norm(d0x - self.u, axis=1) ** 2 / (2. * self.sigma2)))
    
    def d1(self, d0x, d1x):
        return sum(- self.w * (tensordot(d0x - self.u, d1x, axes=1) / self.sigma2) 
                   * exp(- norm(d0x - self.u, axis=1) ** 2 / (2. * self.sigma2)))


class PSO:
    
    def __init__(self, N, M, func, args=()):
        self.func = func
        self.args = args
        self.N  = N
        self.M  = M
        self.X  = 1.0 * (rand(self.N, self.M) - 0.5)
        self.V  = 16.0 * (rand(self.N, self.M) - 0.5)
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
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    