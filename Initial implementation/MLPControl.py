#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:21:29 2024

@author: arslan
"""


from numpy import (array, arange, cos, sin, )
from numpy.linalg import (inv, )
from numpy.random import (rand, )
from scipy.integrate import (solve_ivp, )
import matplotlib.pyplot as plt

from MLP import (PSO, MISOMLP, stringSolution, IAE, saveSolution, tanhActi, )


def sat(x, l, h):
    return max(l, min(x, h))

def angle_error(θ1, θ2):
    return θ1 - θ2

class EulerLagrange:
    
    Iz1 = 0.027
    Iz2 = 0.027
    m1  = 0.506
    m2  = 0.506
    l1  = 0.305
    l2  = 0.305
    k1 = 0.01625
    k2 = 0.01625
    g = 9.81
    
    def __init__(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
    
    def __call__(self, t, X, ):
        u1 = self.u1.d0(array([X[0], X[1], X[4], ]))
        u2 = self.u2.d0(array([X[2], X[3], X[5], ]))
        dx1 = X[1]
        dx3 = X[3]
        
        m11 = self.Iz1 + self.Iz2 + self.m1 * self.l1 ** 2 / 4 + \
            self.m2 * (self.l1 ** 2 + self.l2 ** 2 / 4 + self.l1 * self.l2 * cos(X[2]))
        m12 = self.Iz2 + self.m2 * (self.l2 ** 2 /4 + self.l1 * self.l2 * cos(X[2]) / 2)
        m21 = m12
        m22 = self.Iz2 + self.m2 * self.l2 ** 2 / 4
        M = array([[m11, m12, ], 
                   [m21, m22, ], ])
        c = 0.5 * self.m2 * self.l1 * self.l2 * sin(X[2])
        h11 = -c * X[3] + self.k1
        h12 = -c * (X[1] + X[3])
        h21 = c * X[1]
        h22 = self.k2
        H = array([[h11, h12, ], 
                   [h21, h22, ], ])
        d1 = 0.5 * self.m1 * self.g * self.l1 * cos(X[0]) + \
            self.m2 * self.g * (self.l1 * cos(X[0]) + 0.5 * self.l2 * cos(X[0] + X[2]))
        d2 = 0.5 * self.m2 * self.g * self.l2 * cos(X[0] + X[2])
        D = array([d1, d2, ])
        u = array([u1, u2, ])
        ΔM = 0.25 * M
        ΔH = 0.25 * H
        ΔD = 0.25 * D
        y = inv(M + ΔM) @ (u - ΔD - ΔH @ array([X[1], X[3], ]))
        dx2 = y[0]
        dx4 = y[1]
        
        dx5 = X[0]
        dx6 = X[2]
        
        return array([dx1, dx2, dx3, dx4, dx5, dx6, ])
    

if __name__ == "__main__":
    T  = 10.
    S  = 0.01
    
    def errorFunc(P, layers1, layers2, P1Len, P2Len):
        MLPCtrl1 = MISOMLP(layers1, tanhActi, 3, P[:P1Len])
        MLPCtrl2 = MISOMLP(layers2, tanhActi, 3, P[P1Len:P1Len + P2Len])
        myRobot = EulerLagrange(MLPCtrl1, MLPCtrl2)
        
        e = 0.
        for X0 in [array([-0.8, -0.8,  0.8,  0.8, 0., 0., ]), 
                   array([-0.8, -0.8, -0.8, -0.8, 0., 0., ]), 
                   array([ 0.8,  0.8,  0.8,  0.8, 0., 0., ]), 
                   array([ 0.8,  0.8, -0.8, -0.8, 0., 0., ]), ]:
            tt = arange(0., T, S)
            Result = solve_ivp(myRobot, [0., T], X0, 
                               t_eval=tt, dense_output=True, 
                               method="RK45")
            tt = Result.t
            X = Result.y.T
            
            e += (IAE(X, S) * 10. / tt[-1])
            
        print("This turn:", e)
        saveSolution(P, e)
        return e
    
    
    layers1 = [3, 9, 1]
    I1 = 3
    paramNum1 = layers1[0] * (1 + I1)
    for i in range(1, len(layers1)):
        paramNum1 += layers1[i] * (1 + layers1[i - 1])
    
    layers2 = [3, 9, 1]
    I2 = 3
    paramNum2 = layers2[0] * (1 + I2)
    for i in range(1, len(layers2)):
        paramNum2 += layers2[i] * (1 + layers2[i - 1])
    
    myPSO = PSO(25, paramNum1 + paramNum2, 
                errorFunc, args=(layers1, layers2, paramNum1, paramNum2, ))
    for i in range(50):
        myPSO.iterate()
        print("Iteration ", i+1, ".", myPSO.fb)
    P0 = myPSO.xb
    print(stringSolution(P0))
    MLPCtrl1 = MISOMLP(layers1, tanhActi, 3, P0[:paramNum1])
    MLPCtrl2 = MISOMLP(layers2, tanhActi, 3, P0[paramNum1:paramNum1 + paramNum2])
    
    myRobot = EulerLagrange(MLPCtrl1, MLPCtrl2, )
    
    X0 = 2.0 * (rand(6) - 0.5) * array([1., 1., 1., 1., 0., 0., ])
    tt = arange(0., T, S)
    Result = solve_ivp(myRobot, [0., T], X0, 
                       t_eval=tt, dense_output=True, 
                       method="RK45")
    tt = Result.t
    X = Result.y.T
    
    plt.figure(figsize=(8, 4))
    plt.plot(tt, X[:, 0], label=r"$x_{1}$", linestyle="solid")
    plt.plot(tt, X[:, 1], label=r"$x_{2}$", linestyle="dashed")
    plt.plot(tt, X[:, 2], label=r"$x_{3}$", linestyle="dotted")
    plt.plot(tt, X[:, 3], label=r"$x_{4}$", linestyle="dashdot")
    plt.legend()
    plt.grid(True)
    plt.show()






















