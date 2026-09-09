"""SUPG/OSS stabilization terms for the DeC space-time solver"""

import numpy as np


def SUPG_stabilization(all_stabs,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Compute SUPG stabilization contributions for the standard formulation."""

    all_stabs["u"][:] = \
            al*dx_min*op["DxI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["DxDx2"]@(theta_m @ q_prev["u"]) \
        +   c*al*dx_min*op["DxDy"]@(theta_m @ q_prev["v"]) \
        +   al*dx_min*op["DxI"]@all_sources["p"]

    all_stabs["v"][:] = \
            al*dx_min*op["DyI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["DyDx"]@(theta_m @ q_prev["u"]) \
        +   c*al*dx_min*op["DyDy2"]@(theta_m @ q_prev["v"]) \
        +   al*dx_min*op["DyI"]@all_sources["p"]
    
    
    all_stabs["p"][:] = \
           al*dx_min*op["DxI"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt \
        +  al*dx_min*op["DyI"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        +  c*al*dx_min*op["DxDx2"]@(theta_m @ q_prev["p"]) \
        +  c*al*dx_min*op["DyDy2"]@(theta_m @ q_prev["p"]) \
        +  al*dx_min*op["DxI"]@all_sources["u"] \
        +  al*dx_min*op["DyI"]@all_sources["v"]
    
    return all_stabs

def SUPG_stabilization_MOR(all_stabs,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Compute SUPG stabilization contributions for the standard formulation."""

    all_stabs["u"][:] = \
            al*dx_min*op["u"]["p"]["DxI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["u"]["u"]["DxDx2"]@(theta_m @ q_prev["u"]) \
        +   c*al*dx_min*op["u"]["v"]["DxDy"]@(theta_m @ q_prev["v"]) \
        +   al*dx_min*op["u"]["p"]["DxI"]@all_sources["p"]

    all_stabs["v"][:] = \
            al*dx_min*op["v"]["p"]["DyI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["v"]["u"]["DyDx"]@(theta_m @ q_prev["u"]) \
        +   c*al*dx_min*op["v"]["v"]["DyDy2"]@(theta_m @ q_prev["v"]) \
        +   al*dx_min*op["v"]["p"]["DyI"]@all_sources["p"]
    
    all_stabs["p"][:] = \
           al*dx_min*op["p"]["u"]["DxI"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt \
        +  al*dx_min*op["p"]["v"]["DyI"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        +  c*al*dx_min*op["p"]["p"]["DxDx2"]@(theta_m @ q_prev["p"])\
        +  c*al*dx_min*op["p"]["p"]["DyDy2"]@(theta_m @ q_prev["p"])\
        +  al*dx_min*op["p"]["u"]["DxI"]@all_sources["u"]\
        +  al*dx_min*op["p"]["v"]["DyI"]@all_sources["v"]
    
    return all_stabs

def SUPG_stabilization_MOR_uv_p(all_stabs,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):          
    """Compute SUPG stabilization contributions for the standard formulation."""

    all_stabs["uv"][:] = \
            al*dx_min*op["uv"]["p"]["GradI"] @(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["uv"]["uv"]["GradDiv"]@(theta_m @ q_prev["uv"]) \
        +   al*dx_min*op["uv"]["p"]["GradI"]@all_sources["p"]

    all_stabs["p"][:] = \
           al*dx_min*op["p"]["uv"]["DivI"] @(q_prev["uv"][m,:] - q_prev["uv"][0,:])/dt \
        +  c*al*dx_min*op["p"]["p"]["DivGrad"]@(theta_m @ q_prev["p"]) \
        +  al*dx_min*op["p"]["uv"]["DivI"]@all_sources["uv"]
    
    return all_stabs

def SUPG_GF_stabilization(all_stabs,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Compute SUPG stabilization contributions for the GF formulation."""

    all_stabs["u"][:] = \
            al*dx_min*op["DxI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +   c*al*dx_min*op["DxDx2_tilde"]@(theta_m @ q_prev["u"]) \
        +   c*al*dx_min*op["DxDy_tilde"]@(theta_m @ q_prev["v"]) \
        +   al*dx_min*op["DxM_tilde"]@all_sources["p"]
    
    all_stabs["v"][:] = \
           al*dx_min*op["DyI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +  c*al*dx_min*op["DyDx_tilde"]@(theta_m@q_prev["u"]) \
        +  c*al*dx_min*op["DyDy2_tilde"]@(theta_m@q_prev["v"]) \
        +  al*dx_min*op["DyM_tilde"]@all_sources["p"]
    
    all_stabs["p"][:] = \
           al*dx_min*op["DxI"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt \
        +  al*dx_min*op["DyI"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        +  c*al*dx_min*op["DxDx2"]@(theta_m @ q_prev["p"]) \
        +  c*al*dx_min*op["DyDy2"]@(theta_m @ q_prev["p"]) \
        +  al*dx_min*op["DxI_tilde"]@all_sources["u"] \
        +  al*dx_min*op["DyI_tilde"]@all_sources["v"]
    
    return all_stabs

def SUPG_GF_stabilization_MOR(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute SUPG stabilization contributions for the GF formulation."""

    all_stabs["u"][:] = \
           al*dx_min*op["u"]["p"]["DxI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +  c*al*dx_min*op["u"]["u"]["DxDx2_tilde"]@(theta_m @ q_prev["u"]) \
        +  c*al*dx_min*op["u"]["v"]["DxDy_tilde"]@(theta_m @ q_prev["v"]) \
        +  al*dx_min*op["u"]["p"]["DxM_tilde"]@all_sources["p"]
    
    all_stabs["v"][:] = \
           al*dx_min*op["v"]["p"]["DyI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +  c*al*dx_min*op["v"]["u"]["DyDx_tilde"]@(theta_m@q_prev["u"]) \
        +  c*al*dx_min*op["v"]["v"]["DyDy2_tilde"]@(theta_m@q_prev["v"]) \
        +  al*dx_min*op["v"]["p"]["DyM_tilde"]@all_sources["p"]
    
    all_stabs["p"][:] = \
           al*dx_min*op["p"]["u"]["DxI"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt \
        +  al*dx_min*op["p"]["v"]["DyI"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        +  c*al*dx_min*op["p"]["p"]["DxDx2"]@(theta_m@q_prev["p"]) \
        +  c*al*dx_min*op["p"]["p"]["DyDy2"]@(theta_m@q_prev["p"]) \
        +  al*dx_min*op["p"]["u"]["DxI_tilde"]@all_sources["u"] \
        +  al*dx_min*op["p"]["v"]["DyI_tilde"]@all_sources["v"]
    
    return all_stabs

def SUPG_GF_stabilization_MOR_uv_p(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute SUPG stabilization contributions for the GF formulation."""

    all_stabs["uv"][:] = \
           al*dx_min*op["uv"]["p"]["GradI"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        +  c*al*dx_min*op["uv"]["uv"]["GradDiv_tilde"]@(theta_m @ q_prev["uv"]) \
        +  al*dx_min*op["uv"]["p"]["GradM_tilde"]@all_sources["p"]
    
    all_stabs["p"][:] = \
           al*dx_min*op["p"]["uv"]["DivI"]@(q_prev["uv"][m,:] - q_prev["uv"][0,:])/dt \
        +  c*al*dx_min*op["p"]["p"]["DivGrad"]@(theta_m@q_prev["p"]) \
        +  al*dx_min*op["p"]["uv"]["DivI_tilde"]@all_sources["uv"]
    
    return all_stabs

#GO: stabilization missing for MOR
def OSS_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute OSS stabilization for the standard formulation."""

    all_stabs["u"][:] = \
        c*al*dx_min*op["ZxMy"]@(theta_m@q_prev["u"])

    all_stabs["v"][:] = \
        c*al*dx_min*op["ZyMx"]@(theta_m@q_prev["v"])
    
    theta_p = theta_m@q_prev["p"]
    all_stabs["p"][:] = \
            c*al*dx_min*op["ZxMy"]@theta_p \
        +   c*al*dx_min*op["ZyMx"]@theta_p
    
    return all_stabs

#GO: stabilization missing for MOR
def OSS_GF_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute OSS stabilization for the GF formulation."""

    all_stabs["u"][:] = \
         c*al*dx_min*(op["ZxMy_tilde"]@(theta_m@q_prev["u"]) \
                     +op["DyZx_int"]@(theta_m@q_prev["v"]) \
                     +op["My_tilde_Zx_int"]@all_sources["p"] \
                    #  +1e-3*(op["DxDx2_tilde"]@(theta_m@q_prev["u"])\
                    #         +op["DxDy_tilde"]@(theta_m@q_prev["v"])\
                    #         -op["DxM_tilde"]@all_sources["p"]
                    #        )
                     )

    all_stabs["v"][:] = \
         c*al*dx_min*(op["ZyMx_tilde"]@(theta_m@q_prev["v"]) \
                     +op["DxZy_int"]@(theta_m@q_prev["u"]) \
                     +op["Mx_tilde_Zy_int"]@all_sources["p"]
                     )
    
    all_stabs["p"][:] = \
         c*al*dx_min*(op["ZxMy"]@(theta_m@q_prev["p"]) \
                     +op["ZyMx"]@(theta_m@q_prev["p"]) \
                     +op["Zx_int_My"]@all_sources["u"] \
                     +op["Zy_int_Mx"]@all_sources["v"]
                    )
    
    return all_stabs

#GO: stabilization missing for MOR
def OSS_curl_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute optional curl-targeted OSS stabilization (standard operators)."""

    beta_m = np.sum(theta_m)
    dtu = (q_prev["u"][m] - q_prev["u"][0])/dt/beta_m - all_sources["u"]
    dtv = (q_prev["v"][m] - q_prev["v"][0])/dt/beta_m - all_sources["v"]

    all_stabs["u"][:] = c*al*dx_min*op["ZyMx"]@dtu
    
    all_stabs["v"][:] = c*al*dx_min*op["ZxMy"]@dtv
    
    return all_stabs

def OSS_GF_curl_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute optional curl-targeted OSS stabilization (GF operators)."""

    beta_m = np.sum(theta_m)
    dtu = (q_prev["u"][m] - q_prev["u"][0])/dt/beta_m - all_sources["u"]
    dtv = (q_prev["v"][m] - q_prev["v"][0])/dt/beta_m - all_sources["v"]

    all_stabs["u"][:] = c*al*dx_min*(op["ZyMx_tilde"]@dtu-op["DxZy_int"]@dtv)
    
    all_stabs["v"][:] = c*al*dx_min*(op["ZxMy_tilde"]@dtv-op["DyZx_int"]@dtv)
    
    return all_stabs