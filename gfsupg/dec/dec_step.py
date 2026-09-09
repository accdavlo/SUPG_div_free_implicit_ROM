"""One DeC correction sweep (single step) for standard and MOR solvers"""

import numpy as np

from .sources_residuals import define_sources, define_sources_MOR, define_sources_MOR_uv_p


def DeC_one_step(problem, DeC, FEM2D, dt, al, stab_curl_coeff, q_prev, L2, q_now,\
                 sub_sources, coriolis_not_uni, get_residual, get_stabilization, curl_stabilization,\
                 dirichlet_BC=None, curl_stab_flag=False):
    """Perform one DeC correction sweep over all sub-nodes.

    This routine assembles sources, residuals, and stabilization terms for each
    DeC sub-node and applies one explicit update using `inv_lump`.
    """

    # Compute L2 high order space time discretization of the residual

    c   = problem.c
    cor = problem.coriolis
    op  = FEM2D.operator
    fric = problem.friction

    all_sources   = dict()
    gal_residuals = dict()
    all_stabs     = dict()
    for var in problem.vars:
        all_sources[var]   = np.empty(q_prev[var][0,:].shape)
        gal_residuals[var] = np.empty(q_prev[var][0,:].shape)
        all_stabs[var]     = np.empty(q_prev[var][0,:].shape)

    for m in range(1,DeC.n_subNodes):
        # Carefull with the signs! source_u,_v,_p are meant on the LHS, while the other source was on the RHS
        define_sources(all_sources,q_prev,sub_sources,DeC.theta[m,:],cor,coriolis_not_uni,fric)
        get_residual(gal_residuals,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min,al,DeC.theta[m,:],dt)
        get_stabilization(all_stabs,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min,al,DeC.theta[m,:],dt)

        for var in problem.vars:
            L2[var][:] = gal_residuals[var] + all_stabs[var]

        if curl_stab_flag:
            curl_stabilization(all_stabs,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min,stab_curl_coeff,DeC.theta[m,:],dt)
            for var in ["u","v"]:
                L2[var][:] += all_stabs[var]

        for var in problem.vars:
            q_now[var][m,:] = q_prev[var][m,:] - dt*op["inv_lump"]@L2[var][:]

        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for var in dirichlet_BC[bc_item].vars:
                    q_now[var][m,dirichlet_BC[bc_item].indexes] = \
                        dirichlet_BC[bc_item].dirichlet_vector[var]

def DeC_one_step_MOR(ROM,problem, DeC, FEM2D, dt, al, stab_curl_coeff, q_prev, L2, q_now,\
                     sub_sources, coriolis_not_uni, get_residual, get_stabilization, curl_stabilization,\
                     dirichlet_BC = None, curl_stab_flag = False):
    """Perform one DeC correction sweep over all sub-nodes.

    This routine assembles sources, residuals, and stabilization terms for each
    DeC sub-node and applies one explicit update using `inv_lump`.
    """

    # Compute L2 high order space time discretization of the residual

    if ROM.variable_split == "u,v,p":
        define_sources_ = define_sources_MOR
    elif ROM.variable_split == "uv,p":
        define_sources_ = define_sources_MOR_uv_p

    c   = problem.c
    cor = problem.coriolis
    op  = FEM2D.operator_MOR
    fric = problem.friction

    all_sources   = dict()
    gal_residuals = dict()
    all_stabs     = dict()
    for var in ROM.vars:
        all_sources[var]   = np.empty(q_prev[var][0,:].shape)
        gal_residuals[var] = np.empty(q_prev[var][0,:].shape)
        all_stabs[var]     = np.empty(q_prev[var][0,:].shape)

    for m in range(1,DeC.n_subNodes):
        # Carefull with the signs! source_u,_v,_p are meant on the LHS, while the other source was on the RHS
        define_sources_(op,all_sources,q_prev,sub_sources,DeC.theta[m,:],cor,coriolis_not_uni,fric)
        get_residual(gal_residuals,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min,al,DeC.theta[m,:],dt)
        get_stabilization(all_stabs,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min,al,DeC.theta[m,:],dt)

        for var in ROM.vars:
            L2[var][:] = gal_residuals[var] + all_stabs[var]

        for var in ROM.vars:
            q_now[var][m,:] = q_prev[var][m,:] - dt*op[var][var]["inv_lump"]@L2[var][:]

        if dirichlet_BC is not None:
            raise NotImplementedError("Not implemented bcs in the MOR DeC solver")

