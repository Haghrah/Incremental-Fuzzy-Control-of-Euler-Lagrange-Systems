#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:42:21 2021

@author: arslan
"""

from numpy import (array, zeros, trapz, tanh, multiply, )
from numpy.random import (rand, )
from numpy.linalg import (norm, )


def d0sign(x, c=10.):
    return tanh(c * x)

def d1sign(x, c=10.):
    return c * (1 - multiply(tanh(c * x), tanh(c * x)))

class ActivationFunc:
    
    def __init__(self, d0phi, d1phi):
        self.d0phi = d0phi
        self.d1phi = d1phi
    
    def d0(self, d0x):
        return self.d0phi(d0x)
    
    def d1(self, d0x):
        return self.d1phi(d0x)

d0tgh = lambda x: 8. * d0sign(x, 1.)
d1tgh = lambda x: 8. * d1sign(x, 1.)
tanhActi = ActivationFunc(d0tgh, d1tgh)

class MLP:
    
    def __init__(self, layers, phi, I, P):
        # Number of neurons in each layer
        self.layers = layers
        
        # Activation function
        self.phi = phi
        
        # Number of layers
        self.L = len(layers)
        
        # Number of inputs
        self.I = I
        
        # Layer parameters
        self.W = []
        self.B = []
        
        index = 0
        for i in range(self.L):
            if i == 0:
                b = array(P[index:index + self.layers[i]]).reshape((self.layers[i], 1))
                index += self.layers[i]
                w = array(P[index:index + I * self.layers[i]]).reshape((I, self.layers[i]))
                index += I * self.layers[i]
                
                self.B.append(b)
                self.W.append(w)
            else:
                b = array(P[index:index + self.layers[i]]).reshape((self.layers[i], 1))
                index += self.layers[i]
                w = array(P[index:index + self.layers[i - 1] * self.layers[i]]).reshape((self.layers[i - 1], self.layers[i]))
                index += self.layers[i - 1] * self.layers[i]
                
                self.B.append(b)
                self.W.append(w)
        
    
    def d0(self, d0x, l=None):
        if l is None:
            l = self.L - 1
        
        if l == 0:
            return self.phi.d0(self.B[l] + self.W[l].T @ d0x)
        else:
            return self.phi.d0(self.B[l] + self.W[l].T @ self.d0(d0x, l - 1))
    
    def d1(self, d0x, d1x, l=None):
        if l is None:
            l = self.L - 1
        
        if l == 0:
            return self.phi.d1(self.B[l] + self.W[l].T @ d0x) * (self.W[l].T @ d1x)
        else:
            return self.phi.d1(self.B[l] + self.W[l].T @ self.d0(d0x, l - 1)) * (self.W[l].T @ self.d1(d0x, d1x, l - 1))


class MISOMLP:
    
    def __init__(self, layers, phi, I, P):
        self.mlp = MLP(layers, phi, I, P)
    
    def d0(self, d0x):
        return self.mlp.d0(d0x)[0, 0]
    
    def d1(self, d0x, d1x):
        return self.mlp.d1(d0x, d1x)[0, 0]


class PSO:
    
    def __init__(self, N, M, func, args=()):
        self.func = func
        self.args = args
        self.N  = N
        self.M  = M
        self.X  = 0.1 * (rand(self.N, self.M) - 0.5)
        self.V  = 0.2 * (rand(self.N, self.M) - 0.5)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    