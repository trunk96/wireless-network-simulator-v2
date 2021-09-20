# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:34:32 2021

@author: andre
"""

from gekko import GEKKO
import numpy as np

def output_datarate_optimization(q, w, N, M, P, T_s, U_max=1000, Q_max=1, tol=1e-7):

    '''
    inputs: q(k) and w(k) and P(k) 
    set of base stations visible by sensors
    '''
    
    
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
       
    m.Obj(sum([(sum(u_k[:,i])-sum(u_k[:,j]))**2 for i in range(M) for j in range(M)]) + 100*sum([(q_k[i] + T_s*(w_k[i] - sum(u_k[i,:])))**2 for i in range(N)]))                                                
    m.solve(disp=False)
    u_final = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            u_final[i,j] = u_k[i,j].value[0]
            if u_final[i,j] < tol:
                u_final[i,j] = 0
    return u_final

def build_drone_pos_ref(u, x, y, u_0, x_0, y_0):
    
    ref_x = (u_0*x_0 + np.dot(u,x))/(u_0 + sum(u))
    ref_y = (np.dot(u,y) + u_0*y_0)/(u_0 + sum(u))
    droneRef = [ref_x,ref_y]
    
    return droneRef