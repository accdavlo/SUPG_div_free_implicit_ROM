"""Source-term and Galerkin-residual assembly for the DeC space-time solver"""

import numpy as np


def define_sources(all_sources, q_prev, sub_sources, theta_m, cor, coriolis_not_uni, fric):
    """Build momentum and pressure source terms in semi-discrete form.

    Signs follow the convention used in `DeC_one_step`, where the assembled
    source terms are moved to the left-hand side of the residual equations.
    """

    all_sources["u"][:] = -cor* (theta_m @ q_prev["v"]) \
            - theta_m @ (q_prev["v"]*coriolis_not_uni) \
            + fric* (theta_m @ q_prev["u"]) \
            - theta_m@sub_sources["u"]
    all_sources["v"][:] = cor* (theta_m @ q_prev["u"]) \
            + theta_m @ (q_prev["u"]*coriolis_not_uni) \
            + fric* (theta_m @ q_prev["v"]) \
            - theta_m@sub_sources["v"]
    all_sources["p"][:] = - theta_m@sub_sources["p"]
    
    return all_sources

#GO: Is coriolis_not_uni missing?
def define_sources_MOR(op, all_sources, q_prev, sub_sources, theta_m, cor, coriolis_not_uni, fric):
    """Build momentum and pressure source terms in semi-discrete form.

    Signs follow the convention used in `DeC_one_step`, where the assembled
    source terms are moved to the left-hand side of the residual equations.
    """

    all_sources["u"][:] = -cor* (theta_m @ q_prev["v"]) \
        + fric* (theta_m @ q_prev["u"]) \
        - theta_m@sub_sources["u"]
    all_sources["v"][:] = cor* (theta_m @ q_prev["u"])\
        + fric* (theta_m @ q_prev["v"])\
        - theta_m@sub_sources["v"]
    all_sources["p"][:] = - theta_m@sub_sources["p"]

    return all_sources

def define_sources_MOR_uv_p(op, all_sources, q_prev, sub_sources, theta_m, cor, coriolis_not_uni, fric):
    """Build momentum and pressure source terms in semi-discrete form.

    Signs follow the convention used in `DeC_one_step`, where the assembled
    source terms are moved to the left-hand side of the residual equations.
    """

    all_sources["uv"][:] = -cor* op["uv"]["uv"]["perp"]@(theta_m@ q_prev["uv"])\
        + fric* (theta_m @ q_prev["uv"])\
        - theta_m@sub_sources["uv"]
    all_sources["p"][:] = - theta_m@sub_sources["p"]

    return all_sources

def define_residuals(galer_residuals,q_prev,all_sources,m,op,c,dx_min ,al,theta_m,dt):
    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        + c * op["IDx"]@(theta_m @ q_prev["p"]) \
        +     op["mass"]@all_sources["u"]

    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        + c * op["IDy"]@(theta_m @ q_prev["p"]) \
        +     op["mass"]@all_sources["v"]
        
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        + c * op["IDx"]@(theta_m @ q_prev["u"]) \
        + c * op["IDy"]@(theta_m @ q_prev["v"]) \
        +     op["mass"]@all_sources["p"]

    return galer_residuals

def define_residuals_MOR(galer_residuals,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["u"][:] = op["u"]["u"]["mass"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt \
        + c * op["u"]["p"]["IDx"]@(theta_m @ q_prev["p"]) \
        +     op["u"]["u"]["mass"]@all_sources["u"]

    galer_residuals["v"][:] = op["v"]["v"]["mass"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt\
        + c * op["v"]["p"]["IDy"]@(theta_m @ q_prev["p"]) \
        +     op["v"]["v"]["mass"]@all_sources["v"]
        
    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        + c * op["p"]["u"]["IDx"]@(theta_m @ q_prev["u"]) \
        + c * op["p"]["v"]["IDy"]@(theta_m @ q_prev["v"]) \
        +     op["p"]["p"]["mass"]@all_sources["p"]

    return galer_residuals

def define_residuals_MOR_uv_p(galer_residuals, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["uv"][:] = op["uv"]["uv"]["mass"]@(q_prev["uv"][m,:] - q_prev["uv"][0,:])/dt \
        + c * op["uv"]["p"]["IDiv"]@(theta_m @ q_prev["p"]) \
        +     op["uv"]["uv"]["mass"]@all_sources["uv"]

    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        + c * op["p"]["uv"]["IDiv"]@(theta_m @ q_prev["uv"]) \
        +     op["p"]["p"]["mass"]@all_sources["p"]

    return galer_residuals

def define_GF_residuals(galer_residuals,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Assemble Galerkin residuals for the global-flux (GF) formulation."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt\
        + c * op["IDx"]@(theta_m @ q_prev["p"]) \
        +     op["mass_tilde_x"]@all_sources["u"]
    
    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        + c * op["IDy"]@(theta_m @ q_prev["p"]) \
        +     op["mass_tilde_y"]@all_sources["v"]
    
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        + c * op["IDx_tilde"]@(theta_m @ q_prev["u"]) \
        + c * op["IDy_tilde"]@(theta_m @ q_prev["v"]) \
        +     op["mass_tilde"]@all_sources["p"]

    return galer_residuals

def define_GF_residuals_MOR(galer_residuals,q_prev,all_sources,m,op,c,dx_min,al,theta_m,dt):
    """Assemble Galerkin residuals for the global-flux (GF) formulation."""

    galer_residuals["u"][:] = op["u"]["u"]["mass"]@(q_prev["u"][m,:] - q_prev["u"][0,:])/dt\
        + c * op["u"]["p"]["IDx"]@(theta_m @ q_prev["p"]) \
        +     op["u"]["u"]["mass_tilde_x"]@all_sources["u"]
    
    galer_residuals["v"][:] = op["v"]["v"]["mass"]@(q_prev["v"][m,:] - q_prev["v"][0,:])/dt \
        + c * op["v"]["p"]["IDy"]@(theta_m @ q_prev["p"]) \
        +     op["v"]["v"]["mass_tilde_y"]@all_sources["v"]
    
    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        + c * op["p"]["u"]["IDx_tilde"]@(theta_m @ q_prev["u"]) \
        + c * op["p"]["v"]["IDy_tilde"]@(theta_m @ q_prev["v"]) \
        +     op["p"]["p"]["mass_tilde"]@all_sources["p"]

    return galer_residuals

def define_GF_residuals_MOR_uv_p(galer_residuals, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Assemble Galerkin residuals for the global-flux (GF) formulation."""

    galer_residuals["uv"][:] = op["uv"]["uv"]["mass"]@(q_prev["uv"][m,:]-q_prev["uv"][0,:])/dt\
        + c * op["uv"]["p"]["IGrad"]@(theta_m @ q_prev["p"]) \
        +     op["uv"]["uv"]["mass_tilde_xy"]@all_sources["uv"]
    
    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:] - q_prev["p"][0,:])/dt \
        + c * op["p"]["uv"]["IDiv_tilde"]@(theta_m @ q_prev["uv"]) \
        +     op["p"]["p"]["mass_tilde"]@all_sources["p"]

    return galer_residuals