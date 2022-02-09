# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:34:32 2021

@author: andre
"""

from gekko import GEKKO
import numpy as np

import pyomo.environ as pyo
import pyomo.util.infeasible as pyo_util
from pyomo.opt import SolverStatus, TerminationCondition

def visible_basestation_constraint_rule(m, i, j):
    return m.u_k[i,j] <= m.P[i,j]*m.u_k[i,j]

def max_U_constraint_rule(m, j):
    return sum(m.u_k[i, j] for i in m.N)<=m.U_max

def queue_constraint_lower_rule(m, i):
    return m.q[i] + m.T_s*(m.w[i] - sum(m.u_k[i,j] for j in m.M))>=0
def queue_constraint_upper_rule(m, i):
    return m.q[i] + m.T_s*(m.w[i] - sum(m.u_k[i,j] for j in m.M))<=m.Q_max

def obj_rule(m):
    load_balancing = 0
    for j in m.M:
        for k in m.M:
            if k>j:
                load_balancing += (sum(m.u_k[i,j] for i in m.N) - sum(m.u_k[i,k] for i in m.N))**2
    queue_empty = sum((m.q[i] + m.T_s*(m.w[i] - sum(m.u_k[i,j] for j in m.M))) for i in m.N)
    return load_balancing + queue_empty

def output_datarate_optimization_PYOMO(q, w, N, M, P, T_s, U_max=1000, Q_max=1, tol=1e-7):
    if N == 0:
        return []
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, N-1)
    m.M = pyo.RangeSet(0, M-1)
    m.q = pyo.Param(m.N, initialize = {i: q[i] for i in m.N}, within = pyo.Reals)
    m.w = pyo.Param(m.N, initialize = {i: w[i] for i in m.N}, within = pyo.NonNegativeReals)
    m.P = pyo.Param(m.N, m.M, initialize = {(i,j): P[i][j] for j in m.M for i in m.N}, within = pyo.NonNegativeReals)
    m.T_s = pyo.Param(initialize = T_s, within = pyo.NonNegativeReals)
    m.U_max = pyo.Param(initialize = U_max, within = pyo.NonNegativeReals)
    m.Q_max = pyo.Param(initialize = Q_max, within = pyo.NonNegativeReals)
    
    m.u_k = pyo.Var(m.N, m.M, domain=pyo.NonNegativeReals)

    m.visible_basestation_constraint = pyo.Constraint(m.N, m.M, rule = visible_basestation_constraint_rule)
    m.max_U_constraint = pyo.Constraint(m.M, rule = max_U_constraint_rule)
    m.queue_constraint_lower = pyo.Constraint(m.N, rule = queue_constraint_lower_rule)
    m.queue_constraint_upper = pyo.Constraint(m.N, rule = queue_constraint_upper_rule)
    m.obj = pyo.Objective(rule = obj_rule)
    opt = pyo.SolverFactory("ipopt", executable="ipopt-win64\\ipopt.exe")
    ret = opt.solve(m)
    u_final = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            u_final[i,j] = pyo.value(m.u_k[i,j])
    return u_final


def output_datarate_optimization_GEKKO(q, w, N, M, P, T_s, U_max=1000, Q_max=1, tol=1e-7):

    '''
    inputs: q(k) and w(k) and P(k) 
    set of base stations visible by sensors
    '''
    if N == 0:
        return []
    
    
    #initialize the model
    m = GEKKO(remote=False)
    
    
    q_k = m.Array(m.Param, N)
    for i in range(N):
        q_k[i].value = q[i]
    
    
    w_k = m.Array(m.Param, N)
    for i in range(N):
        w_k[i].value = w[i]
        
        
    u_k = m.Array(m.Var, (N,M), value=0, lb=0)
    P_k = m.Array(m.Param, (N,M))
    for i in range(N):
        for j in range(M):
            P_k[i,j].value = P[i,j]
    for i in range(N):
        for j in range(M):
            m.Equation((1-P_k[i,j])*u_k[i,j] <= 0.0)
            
            
    for j in range(M):
        m.Equation(sum(u_k[:,j])<=U_max)
    
    
    for i in range(N):
        m.Equation(q_k[i] + T_s*(w_k[i] - sum(u_k[i,:]))>=0)
        m.Equation(q_k[i] + T_s*(w_k[i] - sum(u_k[i,:]))<=Q_max)
    
    #Load balancing
       
    m.Obj(sum([(sum(u_k[:,i])-sum(u_k[:,j])) for i in range(M-1) for j in range(i+1,M)]) + 100*sum([(q_k[i] + T_s*(w_k[i] - sum(u_k[i,:]))) for i in range(N)]))                                                
    m.solve(disp=False)
    u_final = np.zeros((N,M)) 
    for i in range(N):
        for j in range(M):
            u_final[i,j] = u_k[i,j].value[0]
            if u_final[i,j] < tol:
                u_final[i,j] = 0
    return u_final

def output_datarate_optimization(q, w, N, M, P, T_s, U_max=1000, Q_max=1, tol=1e-7):
    return output_datarate_optimization_PYOMO(q, w, N, M, P, T_s, U_max=1000, Q_max=1, tol=1e-7)

def build_drone_pos_ref(u, x, y, u_0, x_0, y_0):
    if u_0 + sum(u) == 0:
        return [x_0, y_0]
    ref_x = (u_0*x_0 + np.dot(u,x))/(u_0 + sum(u))
    ref_y = (np.dot(u,y) + u_0*y_0)/(u_0 + sum(u))
    droneRef = [ref_x,ref_y]
    
    return droneRef