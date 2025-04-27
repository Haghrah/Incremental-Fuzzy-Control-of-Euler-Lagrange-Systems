#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:28:37 2024

@author: arslan
"""

from math import (isclose, )
from numpy import (array, arange, exp, reshape, zeros, sign, cos, sin, arccos, log, 
                   argmax, tanh, mean, average, std, trapz, sum, diag, zeros_like, 
                   round, ones_like, pi, )
from numpy.linalg import (inv, )
from numpy.random import (rand, randn, )
from scipy.integrate import (solve_ivp, BDF, )
from numpy.linalg import (norm, )
from copy import (deepcopy, )
import matplotlib.pyplot as plt


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
    
    def __call__(self, t, X):
        u1 = self.u1(t, array([X[0], X[1], X[4], ]))[0]
        u2 = self.u2(t, array([X[2], X[3], X[5], ]))[0]
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
        y = inv(M) @ u
        dx2 = y[0]
        dx4 = y[1]
        
        dx5 = X[0]
        dx6 = X[2]
        
        return array([dx1, dx2, dx3, dx4, dx5, dx6, ])
    
    
def gaussian_mf(x, params):
    return params[2] * exp(-(((params[0] - x) ** 2) / (2 * params[1] ** 2)))

def lgaussian_mf(x, params):
    return (x > params[0]) * params[2] + (x <= params[0]) * gaussian_mf(x, params)

def rgaussian_mf(x, params):
    return (x < params[0]) * params[2] + (x >= params[0]) * gaussian_mf(x, params)

def sigmoid_mf(x, params):
    return params[2] / (1. + params[0] * exp(- params[1] * x))

def inverse_sigmoid_mf(x, params):
    return sigmoid_mf(-x, params)


class FILSigmoidMF:
    
    def __init__(self, params, name):
        self.params = params
        self.name = name
    
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, x):
        return sigmoid_mf(x, self.params)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name


class FILInvSigmoidMF:
    
    def __init__(self, params, name):
        self.params = params
        self.name = name
    
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, x):
        return inverse_sigmoid_mf(x, self.params)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name


class FILGaussianMF:
    
    def __init__(self, params, name):
        self.params = params
        self.name = name
    
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, x):
        return gaussian_mf(x, self.params)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name
        


class FILRGaussianMF:
    
    def __init__(self, params, name):
        self.params = params
        self.name = name
    
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, x):
        return rgaussian_mf(x, self.params)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name


class FILLGaussianMF:
    
    def __init__(self, params, name):
        self.params = params
        self.name = name
    
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, x):
        return lgaussian_mf(x, self.params)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name


class FILRule:
    
    def __init__(self, inputs):
        self.inputs = inputs
        
    def resolveContradiction(self, ):
        pass
    
    def __call__(self, X):
        f = 1.
        for inp, x in zip(self.inputs, X):
            f *= inp(x)
        return f



class FILControl:
    
    
    def __init__(self, sets, R0, X0, W0, Y0, 
                 Tu, l, m, run, 
                 experienceListDict, 
                 name, 
                 Gain=8., 
                 training=False):
        
        self.R = R0
        self.W = W0
        self.X = [X0,  ]
        self.T = [ 0., ]
        self.sets = sets
        self.N = len(self.sets)
        A0 = self.getActiveRule(X0)
        self.A = [A0,  ]
        self.Tu = Tu
        self.l = l
        self.m = m
        self.Gain = Gain
        self.run = run
        self.experienceListDict = experienceListDict
        self.name = name
        Y0 = []
        for i in range(len(R0)):
            Y0.append(self.experienceBasedDecision(i))
        C0 = [self.experienceListDict[i][-1][-1] for i in range(len(self.experienceListDict))]
        self.Y = [Y0,  ]
        self.C = [C0, ]
        self.Yb = zeros_like(Y0, dtype="float") + Y0
        self.Cb = [self.experienceListDict[i][-1][-1] for i in range(len(self.experienceListDict))]
        self.training = training
    
    
    def criteria(self, t, X):
        return 2. * t * (X.T @ self.W @ X) ** 0.5
    
    
    def getActiveRule(self, X):
        activeRule = []
        for i in range(self.N):
            strength = [fuzzySet(X[i]) for fuzzySet in self.sets[i]]
            activeRule.append(self.sets[i][argmax(strength)])
        ruleIndex = -1
        for i in range(len(self.R)):
            if all(set1 == set2 for set1, set2 in zip(self.R[i].inputs, activeRule)):
                ruleIndex = i
                break
        return ruleIndex
    
    
    def updateExperienceList(self, t, X):
        if len(self.Y) > 1:
            lastActiveRule = self.A[-1]
            c1 = self.criteria(self.T[-1], self.X[-1], )
            c0 = self.criteria(t, X, )
            c = c1 - c0
            if c >= 0.:
                experience = (deepcopy(self.Y[-1][lastActiveRule]), c0, )
                self.experienceListDict[lastActiveRule].append(experience)
    
    
    def experienceBasedDecision(self, activeRuleIndex):
        decisionList = []
        weightList = []
        for experience in self.experienceListDict[activeRuleIndex][-20:]:
            if experience[1] > 0:
                decisionList.append(experience[0])
                weightList.append(experience[1])
                
        if len(decisionList) > 0:
            return average(decisionList, axis=0, weights=weightList)
        else:
            return [0., ] * self.m * self.l
    
    
    def regulateRules(self, t, X, K=0.1):
        self.updateExperienceList(t, X)
        
        Y = deepcopy(self.Y[-1])
        C = deepcopy(self.C[-1])
        ruleIndex = self.A[-1]
        experienceMean = self.experienceBasedDecision(ruleIndex)
        
        c0 = self.criteria(t, X)
        c1 = self.criteria(self.T[-1], self.X[-1])
        c = c1 - c0
            
        if c <= 0:
            for i in range(self.m):
                for j in range(self.l):
                    Y[ruleIndex][i * self.l + j] += 0.1 * rand() * (self.Yb[ruleIndex][i * self.l + j] \
                                                                    -  Y[ruleIndex][i * self.l + j]) \
                        + 0.1 * rand() * (experienceMean[i * self.l + j] - Y[ruleIndex][i * self.l + j]) \
                        + K * Y[ruleIndex][i * self.l + j] * (rand() - 0.5)
        
        if c > 0:
            for i in range(self.m):
                for j in range(self.l):
                    self.Yb[ruleIndex][i * self.l + j] = Y[ruleIndex][i * self.l + j]
            self.Cb[ruleIndex] = c0
            
            print("\a")
            print(self.name[self.name.find("/") + 1:] + ". " + "Run.", self.run)
            print(self.name[self.name.find("/") + 1:] + ". " + "Time.", round(t, 5))
            print(self.name[self.name.find("/") + 1:] + ". " + "Criteria:", round(c, 5))
            print(self.name[self.name.find("/") + 1:] + ". " + "Y:", round(Y[ruleIndex], 2))
        
        C[ruleIndex] = c
        return Y, C
    
    
    def update(self, t, X):
        if t >= self.T[-1] + self.Tu:
            y, c = self.regulateRules(t, X)
            self.X.append(X)
            self.Y.append(y)
            self.C.append(c)
            self.A.append(self.getActiveRule(X))
            self.T.append(t)
        elif self.getActiveRule(X) != self.A[-1]:
            y, c = self.regulateRules(t, X)
            self.X.append(X)
            self.Y.append(y)
            self.C.append(c)
            self.A.append(self.getActiveRule(X))
            self.T.append(t)
    
    
    def evaluate(self, t, X):
        o = zeros(shape=(self.m, self.l, ))
        
        for k in range(self.m):
            s1, s2 = [0., ] * self.l , 0.
            for rule, Y in zip(self.R, self.Y[-1]):
                f = rule(X)
                s2 += f
                for i, y in zip(range(self.l), Y[k * self.l:(k + 1) * self.l]):
                    s1[i] += y * f
            for i in range(self.l):
                o[k, i] = (s1[i] / s2) * X[i]
        return self.Gain * tanh(sum(o, axis=1))
    
    
    def __call__(self, t, X):
        if self.training:
            self.update(t, X)
        return self.evaluate(t, X)
    

def saveExperience(FIL:FILControl):
    solFile = open(FIL.name + ".txt", "w")
    for i in range(len(FIL.R)):
        experienceStr = ""
        for experience in FIL.experienceListDict[i]:
            for j in range(FIL.m):
                for k in range(FIL.l):
                    experienceStr += str(experience[0][j * FIL.l + k]) + ", "
            experienceStr += str(experience[1]) + ", "
        solFile.write(experienceStr[:-2])
        solFile.write("\n")
    solFile.close()


def loadExperience(l, m, filename):
    solFile = open(filename + ".txt", "r")
    experienceListDict = {}
    ruleIndex = 0
    for line in solFile:
        line = line.split(",")
        experienceListDict[ruleIndex] = []
        for experienceIndex in range(len(line) // (m * l + 1)):
            experienceList = []
            for i in range(m):
                for j in range(l):
                    experienceList.append(float(line[(m * l + 1) * experienceIndex + i * l + j]))
            experienceListDict[ruleIndex].append((experienceList, 
                                                  float(line[(m * l + 1) * experienceIndex + m * l])))
        ruleIndex += 1
    solFile.close()
    return experienceListDict


def plotFuzzySet(F, Labels):
    lineStyles = ["solid", "dashed", "dotted", ]
    e = arange(-1.5, 1.5, 0.001)
    plt.figure()
    for f, l, ls in zip(F, Labels, lineStyles):
        plt.plot(e, f(e), label=l, linestyle=ls)
    
    plt.title("Input Fuzzy Sets")
    plt.xlabel("Domain")
    plt.ylabel("Membership Degree")
    plt.ylim((-0.05, 1.05))
    
    plt.grid(which="major", linestyle="-", linewidth=0.75, color="gray", alpha=0.7)
    plt.minorticks_on() 
    plt.grid(which="minor", linestyle=":", linewidth=0.5, color="lightgray", alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.savefig("inFuzzySets.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


def solveODE(f, X0, step, ts, te):
    o = [X0, ]
    for t in arange(ts, te - step, step):
        o.append(o[-1] + step * f(t, o[-1]))
    return array(o)


def IAE(X, dt):
    return trapz(norm(X, axis=1), dx=dt)


P1 = FILLGaussianMF([ 0.1, 0.05, 1.0], "P1")
P2 = FILLGaussianMF([ 0.1, 0.05, 1.0], "P2")
P3 = FILLGaussianMF([ 0.1, 0.05, 1.0], "P3")
P4 = FILLGaussianMF([ 0.1, 0.05, 1.0], "P4")
Z1 =  FILGaussianMF([ 0.0, 0.05, 1.0], "Z1")
Z2 =  FILGaussianMF([ 0.0, 0.05, 1.0], "Z2")
Z3 =  FILGaussianMF([ 0.0, 0.05, 1.0], "Z3")
Z4 =  FILGaussianMF([ 0.0, 0.05, 1.0], "Z4")
N1 = FILRGaussianMF([-0.1, 0.05, 1.0], "N1")
N2 = FILRGaussianMF([-0.1, 0.05, 1.0], "N2")
N3 = FILRGaussianMF([-0.1, 0.05, 1.0], "N3")
N4 = FILRGaussianMF([-0.1, 0.05, 1.0], "N4")

setsDict = {"P1":P1, "P2":P2, "P3":P3, "P4":P4, 
            "Z1":Z1, "Z2":Z2, "Z3":Z3, "Z4":Z4, 
            "N1":N1, "N2":N2, "N3":N3, "N4":N4, }

Sets1 = [[P1, Z1, N1, ], 
         [P2, Z2, N2, ], 
         [P3, Z3, N3, ], ]

Sets2 = [[P1, Z1, N1, ], 
         [P2, Z2, N2, ], 
         [P3, Z3, N3, ], ]

R01 = [FILRule([N1, N2, N3,], ), FILRule([Z1, N2, N3,], ), FILRule([P1, N2, N3,], ), 
       FILRule([N1, N2, Z3,], ), FILRule([Z1, N2, Z3,], ), FILRule([P1, N2, Z3,], ), 
       FILRule([N1, N2, P3,], ), FILRule([Z1, N2, P3,], ), FILRule([P1, N2, P3,], ), 
       FILRule([N1, Z2, N3,], ), FILRule([Z1, Z2, N3,], ), FILRule([P1, Z2, N3,], ), 
       FILRule([N1, Z2, Z3,], ), FILRule([Z1, Z2, Z3,], ), FILRule([P1, Z2, Z3,], ), 
       FILRule([N1, Z2, P3,], ), FILRule([Z1, Z2, P3,], ), FILRule([P1, Z2, P3,], ), 
       FILRule([N1, P2, N3,], ), FILRule([Z1, P2, N3,], ), FILRule([P1, P2, N3,], ), 
       FILRule([N1, P2, Z3,], ), FILRule([Z1, P2, Z3,], ), FILRule([P1, P2, Z3,], ), 
       FILRule([N1, P2, P3,], ), FILRule([Z1, P2, P3,], ), FILRule([P1, P2, P3,], ), ]

R02 = [FILRule([N1, N2, N3,], ), FILRule([Z1, N2, N3,], ), FILRule([P1, N2, N3,], ), 
       FILRule([N1, N2, Z3,], ), FILRule([Z1, N2, Z3,], ), FILRule([P1, N2, Z3,], ), 
       FILRule([N1, N2, P3,], ), FILRule([Z1, N2, P3,], ), FILRule([P1, N2, P3,], ), 
       FILRule([N1, Z2, N3,], ), FILRule([Z1, Z2, N3,], ), FILRule([P1, Z2, N3,], ), 
       FILRule([N1, Z2, Z3,], ), FILRule([Z1, Z2, Z3,], ), FILRule([P1, Z2, Z3,], ), 
       FILRule([N1, Z2, P3,], ), FILRule([Z1, Z2, P3,], ), FILRule([P1, Z2, P3,], ), 
       FILRule([N1, P2, N3,], ), FILRule([Z1, P2, N3,], ), FILRule([P1, P2, N3,], ), 
       FILRule([N1, P2, Z3,], ), FILRule([Z1, P2, Z3,], ), FILRule([P1, P2, Z3,], ), 
       FILRule([N1, P2, P3,], ), FILRule([Z1, P2, P3,], ), FILRule([P1, P2, P3,], ), ]

Y0 = array([[-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], 
            [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], [-2.0, -0.5, -0.5], ])

W1 = diag([4., 1., 0., ])
W2 = diag([2., 1., 0., ])

if __name__ == "__main__":
    Tu = 0.1
    T  = 5.
    S  = 0.01
    Run = 200
    training = True
    
    t = arange(0., T, S)
    
    def testRun(runIndex, X0):
        try:
            experienceListDict1 = loadExperience(3, 1, "FFILC training Data/IntegralNFIL1" + str(runIndex))
        except:
            experienceListDict1 = {i:[(Y0[i], 0.1), ] for i in range(len(R01))}
    
        try:
            experienceListDict2 = loadExperience(3, 1, "FFILC training Data/IntegralNFIL2" + str(runIndex))
        except:
            experienceListDict2 = {i:[(Y0[i], 0.1), ] for i in range(len(R02))}
        
        myFIL1 = FILControl(Sets1, R01, X0[[0, 1, 4]], W1, Y0, Tu, 
                            3, 1, runIndex, 
                            experienceListDict1, "FFILC training Data/IntegralNFIL1" + str(runIndex), 
                            training=training)
        myFIL2 = FILControl(Sets2, R02, X0[[2, 3, 5]], W2, Y0, Tu, 
                            3, 1, runIndex, 
                            experienceListDict2, "FFILC training Data/IntegralNFIL2" + str(runIndex),
                            training=training)
        myRobot = EulerLagrange(myFIL1, myFIL2)
        Result = solve_ivp(myRobot, 
                            [0., T], X0, 
                            t_eval=t, 
                            dense_output=True, 
                            max_step = min(S, Tu), 
                            method="BDF")
        return Result.t, Result.y.T
    
    
    if training:
        
        for run in range(0, Run):
            print("Run " + str(run + 1) + ".")
            X0 = (rand(6) - 0.5) * array([1., 1., 1., 1., 0., 0.])
            X0 = X0 / norm(X0)
            try:
                experienceListDict1 = loadExperience(3, 1, "FFILC training Data/IntegralNFIL1" + str(run))
            except:
                experienceListDict1 = {}
                for i in range(len(R01)):
                    experienceListDict1[i] = [(Y0[i], 0.1), ]
    
            try:
                experienceListDict2 = loadExperience(3, 1, "FFILC training Data/IntegralNFIL2" + str(run))
            except:
                experienceListDict2 = {}
                for i in range(len(R02)):
                    experienceListDict2[i] = [(Y0[i], 0.1), ]
            
            myFIL1 = FILControl(Sets1, R01, X0[[0, 1, 4]], W1, Y0, Tu, 
                                3, 1, run + 1, 
                                experienceListDict1, "FFILC training Data/IntegralNFIL1" + str(run + 1), 
                                training=training)
            myFIL2 = FILControl(Sets2, R02, X0[[2, 3, 5]], W2, Y0, Tu, 
                                3, 1, run + 1, 
                                experienceListDict2, "FFILC training Data/IntegralNFIL2" + str(run + 1), 
                                training=training)
            myRobot = EulerLagrange(myFIL1, myFIL2)
            
            t = arange(0., T, S)
            Result = solve_ivp(myRobot, 
                                [0., T], X0, 
                                t_eval=t, 
                                dense_output=True, 
                                max_step = min(S, Tu), 
                                method="BDF")
            t = Result.t
            X = Result.y.T
            
            saveExperience(myFIL1)
            saveExperience(myFIL2)
            
            plt.figure(figsize=(8, 4))
            plt.plot(t, angle_error(X[:, 0], 0), label=r"$x_{1}$", linestyle="solid")
            plt.plot(t, X[:, 1], label=r"$x_{2}$", linestyle="dashed")
            plt.plot(t, angle_error(X[:, 2], 0), label=r"$x_{3}$", linestyle="dotted")
            plt.plot(t, X[:, 3], label=r"$x_{4}$", linestyle="dashdot")
            plt.legend()
            plt.title("Run " + str(run + 1))
            plt.grid(True)
            plt.show()
    
    else:
        T = 5.0
        t = arange(0., T, S)
        exprList = [1, 50, 100, 150, 200]
        exprError = []
        runNum = 5
        X0 = []
        for i in range(runNum):
            X0.append(2.0 * (rand(6) - 0.5) * array([1., 1., 1., 1., 0., 0.]))
            X0[-1] = X0[-1] / norm(X0[-1])
        
        # for expr in exprList:
        for expr in range(1, 201, 1):
            X = zeros((int(T / S), 6))
            E = 0.
            for i in range(runNum):
                t_, X_ = testRun(expr, X0[i])
                e__ = trapz(t * norm(X_[:,[0, 2, ]], axis=1), t)
                E += e__
                X += X_
                print(f"Expr {expr}. Run {i + 1}:{runNum} done: {e__}")
                
                
            
            X /= runNum
            exprError.append(E / runNum)
            print(f"Average IAE over {runNum} runs: {exprError[-1]}")
            
            plt.figure(figsize=(8, 4))
            plt.plot(t, X[:, 0], label=r"$x_{1}$", linestyle="solid")
            plt.plot(t, X[:, 1], label=r"$x_{2}$", linestyle="dashed")
            plt.plot(t, X[:, 2], label=r"$x_{3}$", linestyle="dotted")
            plt.plot(t, X[:, 3], label=r"$x_{4}$", linestyle="dashdot")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(8, 4))
            plt.plot(t, norm(X[:, :3], axis=1))
            plt.grid(True)
            plt.show()
    


