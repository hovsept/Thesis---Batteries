# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:30:33 2023

@author: Hovsep Touloujian
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Adapted from
Created on Fri Jun  1 17:08:57 2018

NMC-Pouch cell

@author: shpark
"""
import numpy as np
from numpy import random as rnd
import scipy.optimize
p={}


#==============================================================================
# Geometric params
#==============================================================================

# Thickness of each layer
p['L_n'] = 5e-5      # Thickness of negative electrode [m]
p['L_s'] = 8.2e-5       # Thickness of separator [m]
p['L_p'] = 3.6e-5     # Thickness of positive electrode [m]
p['A'] = 0.0284     # Electrode Contact Surface Area [m^2]

L_ccn = 25e-6;    # Thickness of negative current collector [m]
L_ccp = 25e-6;    # Thickness of negative current collector [m]


# Particle Radii
p['R_s'] = 1e-5 # Radius of solid particles in positive and negative electrode [m]
p['V_s'] = 4*np.pi*p['R_s']**3/3 # Volume of solid particles [m^3]

# Volume fractions
p['epsilon_s_n'] = 0.58 # Volume fraction in solid for neg. electrode
p['epsilon_s_p'] = 0.5 # Volume fraction in solid for pos. electrode

p['epsilon_e_n'] = 0.3   # Volume fraction in electrolyte for neg. electrode
p['epsilon_e_s'] = 0.4	  # Volume fraction in electrolyte for separator
p['epsilon_e_p'] = 0.3   # Volume fraction in electrolyte for pos. electrode

p['epsilon_f_n'] = 1 - p['epsilon_s_n'] - p['epsilon_e_n']  # Volume fraction of filler in neg. electrode
p['epsilon_f_p'] = 1 - p['epsilon_s_p'] - p['epsilon_e_p']  # Volume fraction of filler in pos. electrode


# Specific interfacial surface area
p['a_s_n'] = 3*p['epsilon_s_n'] / p['R_s']  # Negative electrode [m^2/m^3]
p['a_s_p'] = 3*p['epsilon_s_p'] / p['R_s']  # Positive electrode [m^2/m^3]

#==============================================================================
# Discretization params
#==============================================================================
p['nrn'] = 3
p['nrp'] =3
p['n_sei'] = 3
p['dr_n'] = p['R_s']/(p['nrn']-1)
p['dr_p'] = p['R_s']/(p['nrp']-1)

p['dt'] = 0.1
p['N'] = 100001
p['Tf'] = (p['N']-1)*p['dt']

#==============================================================================
# Miscellaneous
#==============================================================================
p['Rg'] = 8.314472;      # Gas constant, [J/mol-K]
p['F'] = 96487  # Faraday constant [Coulombs/mol]
p['alph'] = 0.5         # Charge transfer coefficients
p['t_plus'] = 0.45		# Transference number
p['brug'] = 1.8		# Bruggeman porosity

#==============================================================================
# Transport params
#==============================================================================

def D_s_n(Tc):
    return 2e-12*np.exp(1/p['Rg'] * (1/298 - 1/(Tc+273)))

def D_s_p(Tc):
    return 3.7e-12*np.exp(1/p['Rg'] * (1/298 - 1/(Tc+273)))

# Conductivity of solid
p['sig_n'] = 100    # Conductivity of solid in neg. electrode, [1/Ohms*m]
p['sig_p'] = 0.1  # Conductivity of solid in pos. electrode, [1/Ohms*m]

#==============================================================================
# Concentrations
#==============================================================================

p['cs_max_n'] = 23230 # Max concentration in anode, [mol/m^3]
p['cs_max_p'] = 27362 # Max concentration in cathode, [mol/m^3]
p['n_Li_s'] = 3.0 # Total moles of lithium in solid phase [mol]
p['ce_avg'] = 1.2e3    # Electrolyte concentration [mol/m^3]

# Stoichimetry points
p['theta_0_n'] = 0.126
p['theta_100_n'] = 0.676 # x100, Cell SOC 100
p['theta_0_p'] = 0.936   # y0, Cell SOC 0
p['theta_100_p'] = 0.442 # y100, Cell SOC 100

#==============================================================================
# Kinetic params
#==============================================================================
p['R_f_n'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
p['R_f_p'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
#p.R_c = 2.5e-03;%5.1874e-05/p.Area; % Contact Resistance/Current Collector Resistance, [Ohms-m^2]

def alpha_s_n(Tc):
    return D_s_n(Tc)/p['dr_n']**2
def alpha_s_p(Tc):
    return D_s_p(Tc)/p['dr_p']**2

p['beta_s_n'] = -1/(p['A']*p['L_n']*p['F']*p['a_s_n']*p['dr_n'])
p['beta_s_p'] = 1/(p['A']*p['L_p']*p['F']*p['a_s_p']*p['dr_p'])


# Nominal Reaction rates
p['k_p_ref'] = 1.4302e-12
p['k_n_ref'] = 6.633e-13
def k_p(Tc):
    return p['k_p_ref']*np.exp(1/p['Rg'] * (1/298 - 1/(Tc+273)))
def k_n(Tc):
    return p['k_n_ref']*np.exp(1/p['Rg'] * (1/298 - 1/(Tc+273)))

# Exchange Current Density
def i0_n(cs_surf_n,Tc):
    return k_n(Tc)*p['F']*np.sqrt(p['ce_avg']*cs_surf_n*(p['cs_max_n']-cs_surf_n))
def i0_p(cs_surf_p,Tc):
    return k_p(Tc)*p['F']*np.sqrt(p['ce_avg']*cs_surf_p*(p['cs_max_p']-cs_surf_p))

# Butler-Volmer Kinetic Equations
def eta_n(I,Tc,cs_surf_n):
    return p['Rg']*(Tc+273)/(0.5*p['F']) * np.arcsinh(I/(2*p['A']*p['a_s_n']*p['L_n']*i0_n(cs_surf_n,Tc)))
def eta_p(I,Tc,cs_surf_p):
    return p['Rg']*(Tc+273)/(0.5*p['F']) * np.arcsinh(I/(2*p['A']*p['a_s_p']*p['L_p']*i0_p(cs_surf_p,Tc)))

# Open-Circuit Potentials
def theta_n(cs_surf_n):
    return cs_surf_n/p['cs_max_n']
def theta_p(cs_surf_p):
    return cs_surf_p/p['cs_max_p']
def U_n(cs_surf_n):
    u = 0.7222 + 0.1387*theta_n(cs_surf_n) + 0.029*theta_n(cs_surf_n)**0.5 -0.0172/theta_n(cs_surf_n)
    u = u + 0.0019/theta_n(cs_surf_n)**1.5 + 0.2808*np.exp(0.9-15*theta_n(cs_surf_n))
    u = u - 0.7984*np.exp(0.4465*theta_n(cs_surf_n)-0.4108)
    return u
def U_p(cs_surf_p):
    u1 = -4.656 + 88.669*theta_p(cs_surf_p)**2 -401.119*theta_p(cs_surf_p)**4
    u1 = u1 + 342.909*theta_p(cs_surf_p)**6 -462.471*theta_p(cs_surf_p)**8 + 433.434*theta_p(cs_surf_p)**10
    u2 = -1+18.933*theta_p(cs_surf_p)**2 -79.532*theta_p(cs_surf_p)**4 + 37.311*theta_p(cs_surf_p)**6
    u2 = u2 -73.083*theta_p(cs_surf_p)**8 + 95.96*theta_p(cs_surf_p)**10
    return u1/u2

p['R_l'] = 0.0704 # Lumped Contact Resistance [Ohm]
p['k_sei'] = 17.5e-5 # SEI Electric Conductivity [S.m^-1]
p['k_eff_n'] = 100 # Negative, Positive Electrode, Separator Electric Conductivity [S.m^-1]
p['k_eff_p'] = 10
p['k_eff_s'] = 0.1

# SEI Resistance
def R_sei(L_sei):
    return L_sei/(p['a_s_n'] * p['A'] * p['L_n'] * p['k_sei'])

# Electric Resistance
p['R_el'] = 1/(2*p['A']) * (p['L_n']/p['k_eff_n'] + 2*p['L_s']/p['k_eff_s'] + p['L_p']/p['k_eff_p'])

def f_V_cell(cs_surf_p, cs_surf_n, Tc, I, L_sei):
    return U_p(cs_surf_p) + eta_p(I,Tc,cs_surf_p) - U_n(cs_surf_n) - eta_n(I,Tc,cs_surf_n) - I*(p['R_l'] + p['R_el'] + R_sei(L_sei))
    
def f_V_oc(cs_surf_p, cs_surf_n):
    return U_p(cs_surf_p) - U_n(cs_surf_n)

# Side-Reaction Current
p['kf'] = 1.18e-22
p['phi_s_n'] = 2.3436
p['U_s'] = 0.5
p['beta'] = 1e-5
def i_s(cs_surf_n,c_surf_solv, L_sei, I, Tc):
    return -2*p['F']*p['kf']*cs_surf_n**2*c_surf_solv*np.exp(-p['beta']*p['F']/(p['Rg']*(Tc+273)) * (p['phi_s_n']-R_sei(L_sei)*I - p['U_s']))

def g_s_n(cs_surf_n, c_surf_solv, Tc, I, L_sei):
    return p['a_s_n']*p['L_n']*p['A']*i_s(cs_surf_n,c_surf_solv,L_sei,I,Tc)

#==============================================================================
# Cell Solid-Phase Diffusion
#==============================================================================
p['A_s_n'] = -2*np.eye(p['nrn'])
p['A_s_p'] = -2*np.eye(p['nrp'])
for i in range(p['nrn']-1):
    p['A_s_n'][i,i+1] = (i+2)/(i+1)
    p['A_s_n'][i+1,i] = (i+1)/(i+2)
    
for i in range(p['nrp']-1):
    p['A_s_p'][i,i+1] = (i+2)/(i+1)
    p['A_s_p'][i+1,i] = (i+1)/(i+2)
p['A_s_n'][-1,-1-1]=2
p['A_s_p'][-1,-1-1]=2


p['B_s_n'] = np.vstack((np.zeros((p['nrn']-1,1)), 2+2/(p['nrn']-1)))
p['B_s_p'] = np.vstack((np.zeros((p['nrp']-1,1)), 2+2/(p['nrp']-1)))

def f_cs_n(Tc, cs_n, I, c_surf_solv, L_sei):
    return alpha_s_n(Tc)*np.matmul(p['A_s_n'],cs_n.reshape((p['nrn'],1))) + p['beta_s_n']*p['B_s_n']*(I-g_s_n(cs_n[-1], c_surf_solv, Tc, I, L_sei))

def f_cs_p(Tc, cs_p, I):
    return alpha_s_p(Tc)*np.matmul(p['A_s_p'],cs_p.reshape((p['nrp'],1))) + p['beta_s_p']*p['B_s_p']*I

p['A_bulk_n'] = np.zeros((p['nrn']-1,p['nrn']))
p['A_bulk_p'] = np.zeros((p['nrp']-1,p['nrp']))
for i in range(p['nrn']-1):
    p['A_bulk_n'][i,i] = p['dr_n']**3/3 * ((i+1)**3 - i**3) - p['dr_n']**3 /4 * ((i+1)**4 - i**4) + p['dr_n']**3 * i/3 * ((i+1)**3-i**3)
    p['A_bulk_n'][i,i+1] = p['dr_n']**3/4 * ((i+1)**4 - i**4) - p['dr_n']**3 *i/3 * ((i+1)**3-i**3)
p['A_bulk_n'] = 3/p['R_s']**3 * np.matmul(np.ones((1,p['nrn']-1)), p['A_bulk_n'])

for i in range(p['nrp']-1):
    p['A_bulk_p'][i,i] = p['dr_p']**3/3 * ((i+1)**3 - i**3) - p['dr_p']**3 /4 * ((i+1)**4 - i**4) + p['dr_p']**3 * i/3 * ((i+1)**3-i**3)
    p['A_bulk_p'][i,i+1] = p['dr_p']**3/4 * ((i+1)**4 - i**4) - p['dr_p']**3 *i/3 * ((i+1)**3-i**3)
p['A_bulk_p'] = 3/p['R_s']**3 * np.matmul(np.ones((1,p['nrp']-1)), p['A_bulk_p'])

def f_SOC_n(cs_n):
    cs_n_bulk = np.matmul(p['A_bulk_n'],cs_n)
    return (cs_n_bulk/p['cs_max_n'] - p['theta_0_n'])/(p['theta_100_n']- p['theta_0_n'])

def f_SOC_p(cs_p):
    cs_p_bulk = np.matmul(p['A_bulk_p'],cs_p)
    return (cs_p_bulk/p['cs_max_p'] - p['theta_0_p'])/(p['theta_100_p']- p['theta_0_p'])

#==============================================================================
# Single-Cell Thermal Model
#==============================================================================

# Thermal dynamics
p['Cc'] = 360
p['Cs'] = 360
p['Rc'] = 0.284
p['Ru'] = 0.284
p['T_amb'] = 20

def f_Tc(I,V_cell,V_oc,Ts,Tc):
    return 1/p['Cc'] * (I*(V_oc-V_cell) + (Ts-Tc)/p['Rc'])
def f_Ts(T_amb,Ts,Tc):
    return 1/p['Cs'] * ((T_amb-Ts)/p['Ru'] - (Ts-Tc)/p['Rc'])


#==============================================================================
# SEI Layer Growth Model
#==============================================================================
p['M_sei'] = 0.162
p['rho_sei'] = 1690
p['beta_sei'] = -p['M_sei']/(2*p['F']*p['rho_sei']*p['a_s_n']*p['L_n']*p['A'])

def f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei):
    return p['beta_sei']*g_s_n(cs_surf_n,c_surf_solv, Tc, I, L_sei)

def f_Q(cs_surf_n,c_surf_solv,Tc,I,L_sei):
    return f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei)/p['beta_sei']

#==============================================================================
# Solvent Diffusion Model
#==============================================================================
p['D_solv_ref'] = 8.58e-19
def D_solv(Tc):
    return p['D_solv_ref']*np.exp(1/p['Rg'] * (1/298 - 1/(Tc+273)))
p['d_zeta'] = 1/(p['n_sei']-1)
def zeta(i):
    return i/(p['n_sei']-1)

def alpha_solv(Tc,L_sei):
    return D_solv(Tc)/(L_sei*p['d_zeta'])**2

def gamma_solv(i,L_sei,cs_surf_n,c_surf_solv,Tc,I):
    return (zeta(i)-1)/(2*L_sei*p['d_zeta']) * f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei)

def beta_solv(Tc,L_sei,cs_surf_n,c_surf_solv,I):
    return 2/(L_sei*p['d_zeta']) + 1/D_solv(Tc) * f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei)

p['A_c_solv'] = np.zeros((p['n_sei']-2,p['n_sei']))
p['B_c_solv'] = np.zeros((p['n_sei']-2,p['n_sei']))
for i in range(p['n_sei']-2):
    p['A_c_solv'][i,i:i+3] = np.array([1, -2, 1])
    p['B_c_solv'][i,i:i+3] = np.array([-1, 0, 1])
    
def f_c_solv(c_solv, Tc, L_sei, cs_surf_n, c_surf_solv, I):
    f1 = 2*alpha_solv(Tc,L_sei)*(c_solv[1]-c_solv[0])
    f1 = f1 + beta_solv(Tc, L_sei, cs_surf_n, c_surf_solv, I)*(i_s(cs_surf_n, c_surf_solv, L_sei, I, Tc)/p['F'] -f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei)*c_solv[0])

    f2 = alpha_solv(Tc, L_sei)*np.matmul(p['A_c_solv'],c_solv)
    f2 = f2+ np.diag(np.arange(1,p['n_sei']-1)/(p['n_sei']-1) -1)/(2*L_sei*p['d_zeta'])*f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei)*np.matmul(p['B_c_solv'],c_solv)
    
    f3 = 0
    return np.vstack((f1,f2,f3))

#==============================================================================
# Safety Constraint
#==============================================================================
p['volt_min'] = 2.7
p['volt_max'] = 4.2

#==============================================================================
# Sample Random Initial Condition
#==============================================================================
def sample_IC(p, SOC_min = 0, SOC_max = 1):
    SOC = 0.5*rnd.uniform(SOC_min,SOC_max)
    cs_n_bulk0 = (SOC*(p['theta_100_n']-p['theta_0_n']) + p['theta_0_n'])*p['cs_max_n']
    cs_p_bulk0 = (SOC*(p['theta_100_p']-p['theta_0_p']) + p['theta_0_p'])*p['cs_max_p']
    
    cs_n0 = np.zeros((p['nrn'],1))
    sum_n = cs_n_bulk0
    max_n = p['cs_max_n']
    for i in reversed(range(1,p['nrn'])):
        min_n = 1/(np.sum(p['A_bulk_n'][0,0:i+1]))*sum_n
        cs_n0[i] = rnd.uniform(min_n, min(sum_n/p['A_bulk_n'][0,i], max_n))
        max_n = cs_n0[i]
        sum_n = sum_n - p['A_bulk_n'][0,i]*cs_n0[i]
    cs_n0[0] = sum_n/p['A_bulk_n'][0,0]

    cs_p0 = np.zeros((p['nrn'],1))
    sum_p = cs_p_bulk0
    max_p = p['cs_max_p']
    for i in range(p['nrp']-1):
        min_p = 1/(np.sum(p['A_bulk_p'][0,i:])) *sum_p
        cs_p0[i] = rnd.uniform(min_p, min(sum_p/p['A_bulk_p'][0,i],max_p))
        max_p = cs_p0[i]
        sum_p = sum_p - p['A_bulk_p'][0,i]*cs_p0[i]
    cs_p0[-1] = sum_p/p['A_bulk_p'][0,-1]
    
    Ts0 = rnd.uniform(20,35)
    Tc0 = rnd.uniform(20,35)
    L_sei0 = rnd.uniform(5e-9,5e-8)
    Q0 = rnd.uniform(0,13)
    
    c_solv0 = np.zeros((p['n_sei'],1))
    c_solv0[0] = rnd.uniform(0,4800)
    for i in range(1,p['n_sei']):
        c_solv0[i] = rnd.uniform(0,c_solv0[i-1])
        
    return np.vstack((cs_n0,cs_p0,Ts0,Tc0,L_sei0,Q0,c_solv0))

def state_convert(x,p):
    cs_n = x[0:p['nrn']]
    cs_p = x[p['nrn']:p['nrp']+p['nrn']]
    Ts = x[p['nrn']+p['nrp']]
    Tc = x[p['nrn']+p['nrp']+1]
    L_sei = x[p['nrn']+p['nrp']+2]
    Q = x[p['nrn']+p['nrp']+3]
    c_solv = x[p['nrn']+p['nrp']+4:]
    c_surf_solv = c_solv[0]
    cs_surf_n = cs_n[-1]
    cs_surf_p = cs_p[-1]
    T_amb = p['T_amb']
    return cs_n, cs_p, Ts, Tc, L_sei, Q, c_solv, c_surf_solv, cs_surf_n, cs_surf_p, T_amb

def f_SPM(x,u,p):
    cs_n, cs_p, Ts, Tc, L_sei, Q, c_solv, c_surf_solv, cs_surf_n, cs_surf_p, T_amb = state_convert(x, p)
    I = u
    
    V_cell = f_V_cell(cs_surf_p, cs_surf_n, Tc, I, L_sei)
    V_oc = f_V_oc(cs_surf_p, cs_surf_n)
    f = np.vstack((f_cs_n(Tc, cs_n, I, c_surf_solv, L_sei),f_cs_p(Tc, cs_p, I)))
    f = np.vstack((f,f_Ts(T_amb, Ts, Tc),f_Tc(I, V_cell, V_oc, Ts, Tc)))
    f = np.vstack((f,f_L_sei(cs_surf_n, c_surf_solv, Tc, I, L_sei),f_Q(cs_surf_n, c_surf_solv, Tc, I, L_sei)))
    f = np.vstack((f,f_c_solv(c_solv, Tc, L_sei, cs_surf_n, c_surf_solv, I)))
    SOC_n = f_SOC_n(cs_n)
    SOC_p = f_SOC_p(cs_p)
    return f, V_cell, V_oc, SOC_n, SOC_p, i_s(cs_surf_n, c_surf_solv, L_sei, I, Tc)

def set_sample(i_s_max,SOC_min,SOC_max,d):
    f = 0
    #i_s_max: max absolute value of side-reaction current
    #d = {0: remain within interior of safe set, 1: border of safe set}
    while f == 0:
        SOC = rnd.uniform(SOC_min,SOC_max)
        cs_n_bulk0 = (SOC*(p['theta_100_n']-p['theta_0_n']) + p['theta_0_n'])*p['cs_max_n']
        cs_p_bulk0 = (SOC*(p['theta_100_p']-p['theta_0_p']) + p['theta_0_p'])*p['cs_max_p']
        
        cs_n0 = np.zeros((p['nrn'],1))
        sum_n = cs_n_bulk0
        max_n = p['cs_max_n']
        for i in reversed(range(1,p['nrn'])):
            min_n = 1/(np.sum(p['A_bulk_n'][0,0:i+1]))*sum_n
            cs_n0[i] = rnd.uniform(min_n, min(sum_n/p['A_bulk_n'][0,i], max_n))
            max_n = cs_n0[i]
            sum_n = sum_n - p['A_bulk_n'][0,i]*cs_n0[i]
        cs_n0[0] = sum_n/p['A_bulk_n'][0,0]

        cs_p0 = np.zeros((p['nrn'],1))
        sum_p = cs_p_bulk0
        max_p = p['cs_max_p']
        for i in range(p['nrp']-1):
            min_p = 1/(np.sum(p['A_bulk_p'][0,i:])) *sum_p
            cs_p0[i] = rnd.uniform(min_p, min(sum_p/p['A_bulk_p'][0,i],max_p))
            max_p = cs_p0[i]
            sum_p = sum_p - p['A_bulk_p'][0,i]*cs_p0[i]
        cs_p0[-1] = sum_p/p['A_bulk_p'][0,-1]
        
        Ts0 = rnd.uniform(20,35)
        Tc0 = rnd.uniform(20,35)
        L_sei0 = rnd.uniform(5e-9,5e-8)
        Q0 = rnd.uniform(0,13)
        
        c_solv_max = i_s_max/(2*p['F']*p['kf'])/(cs_n0[-1]**2)
        c_solv0 = np.zeros((p['n_sei'],1))
        if d == 0:
            c_solv0[0] = rnd.uniform(0,c_solv_max)
        else:
            c_solv0[0] = c_solv_max

        for i in range(1,p['n_sei']):
            c_solv0[i] = rnd.uniform(0,c_solv0[i-1])
        f=1
        # LHS = 1/(p['A']*p['L_n']*p['F']*p['a_s_n']*p['dr_n'])*(2+2/(p['nrn']-1))*(p['a_s_n']*p['L_n']*p['A']*2*p['F']*p['kf'])*cs_n0[-1]**2 * c_solv0[1]
        # RHS = 2*D_s_n(Tc0)/p['dr_n']**2 * (cs_n0[-2] - cs_n0[-1])
        # if  LHS>RHS:
        #     f=1
        
    return np.vstack((cs_n0,cs_p0,Ts0,Tc0,L_sei0,Q0,c_solv0))
    
    
def OneC(p):
    # negative electrode
    cs_n_bulk0 = p['theta_0_n']*p['cs_max_n']
    cs_n_bulk100 = p['theta_100_n']*p['cs_max_n']
    diff_mat_n = np.zeros((p['nrn']-1,p['nrn']))
    for i in range(p['nrn']-1):
        diff_mat_n[i,i:i+2] = np.array([1., -1.])

    c_n_100 = np.vstack((np.zeros((p['nrn']-1,1)), -1))
    c_n_0 = -c_n_100
    cs_n_100 = scipy.optimize.linprog(c_n_100, A_ub=diff_mat_n, b_ub = np.zeros((p['nrn']-1,1)), A_eq = p['A_bulk_n'], b_eq = cs_n_bulk100, bounds=(0., p['cs_max_n']))['x'][-1]
    cs_n_0 = scipy.optimize.linprog(c_n_0, A_ub=diff_mat_n, b_ub = np.zeros((p['nrn']-1,1)), A_eq = p['A_bulk_n'], b_eq = cs_n_bulk0, bounds=(0., p['cs_max_n']))['x'][-1]

    dcs_n = cs_n_100 - cs_n_0
    I_n = p['epsilon_s_n'] * p['L_n'] * p['A'] * dcs_n * p['F']/3600

    # positive electrode
    cs_p_bulk0 = p['theta_0_p']*p['cs_max_p']
    cs_p_bulk100 = p['theta_100_p']*p['cs_max_p']

    diff_mat_p = -diff_mat_n
    c_p_0 = np.vstack((np.zeros((p['nrp']-1,1)), -1))
    c_p_100 = -c_p_0

    cs_p_100 = scipy.optimize.linprog(c_p_100, A_ub=diff_mat_p, b_ub = np.zeros((p['nrp']-1,1)), A_eq = p['A_bulk_p'], b_eq = cs_p_bulk100, bounds=(11570., p['cs_max_p']))['x'][-1]
    cs_p_0 = scipy.optimize.linprog(c_p_0, A_ub=diff_mat_p, b_ub = np.zeros((p['nrp']-1,1)), A_eq = p['A_bulk_p'], b_eq = cs_p_bulk0, bounds=(11570., p['cs_max_p']))['x'][-1]

    dcs_p = cs_p_0 - cs_p_100
    I_p = p['epsilon_s_p'] * p['L_p'] * p['A'] * dcs_p * p['F']/3600

    I = min(I_n,I_p)
    return I
    
#==============================================================================
# CC-CV Parameters
#==============================================================================

p['V_max'] = 4.1
p['I_max'] = 3.0*OneC(p)

#==============================================================================
# Sampling Random Set of Parameters
#==============================================================================
# def param_sample(p):
#     return