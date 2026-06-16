#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt


# In[2]:


#problem.T_fin = 1.
order=3

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")
# dec = DeC(4,5,"gaussLobatto")


# In[3]:


problem = SmoothVortexTestCase()#ConstantFlowTestCase(pert_coeff=1e-1, pert_type="an")


# In[4]:


# Ns = np.array([20,20], dtype=np.int32)

# geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)


# # In[5]:


# FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)


# # In[6]:


# solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)


# # In[7]:


# labels = []
# q_save = dict()
# tt_save = dict()
# comp_time= dict()
# err = dict()
# err_vertex = dict()
# i=-1
# for stab in ["SUPG", "OSS"]:
#     for trick in [True, False]:
#         for GF in [True, False]:
#             i+=1
#             labels.append( stab + "_GF"*GF + "_trick"*trick)

#             q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
#         solver.solve(with_error = True, with_error_vertex = True, GF=GF,\
#                     stab=stab, trick_second_der=trick)

# i+=1
# labels.append("Gal")
# q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
#         solver.solve(with_error = True, with_error_vertex = True, GF=False,\
#                     stab="SUPG", trick_second_der=False, stab_coeff=0.)


# # In[8]:


# q_tmp = dict()
# for var in problem.vars:
#     q_tmp[var] = solver.ic_vect[var]

# styles = ["-","--", "-", "--","-.",":", "-.", ":",":"]
# colors = ["k","k","b","b","g","g","r","r","m"]

# curl0 = FEM2D.compute_discrete_curl(q_tmp)
# err_curl = dict()

# for il, lab in enumerate(labels):
#     err_curl[lab] = np.zeros(len(tt_save[lab]))
#     for it, t in enumerate(tt_save[lab]):
#         q_tmp = dict()
#         for var in problem.vars:
#             q_tmp[var] = q_save[lab][var][it]
#         curl  =  FEM2D.compute_discrete_curl(q_tmp)
#         curl_diff = curl-curl0
#         err_curl[lab][it] =  curl_diff.T @ (FEM2D.operator["mass"] @ curl_diff)

#     plt.semilogy(tt_save[lab][1:-1], err_curl[lab][1:-1], label=lab, linestyle=styles[il], color=colors[il])
# plt.legend()
# plt.ylabel(r"$|| \nabla \times u^{n+1} - \nabla \times u^0 ||$")
# plt.xlabel("Time")
# plt.savefig(problem.folderName+"/curl_error_ord_%d_N_%04d.pdf"%(order,Ns[0]))


# # In[9]:


# labels = []
# q_save = dict()
# tt_save = dict()
# comp_time= dict()
# err = dict()
# err_vertex = dict()
# alphas = dict()
# i=-1
# stab="SUPG"
# for trick in [True, False]:
#     for GF in [True, False]:
#         i+=1
#         labels.append( stab + "_GF"*GF + "_trick"*trick)

#         q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
#                 solver.solve(with_error = True, with_error_vertex = True, GF=GF,\
#                 stab=stab, trick_second_der=trick)
#         alphas[labels[i]] = solver.stab_coeff

# i+=1
# labels.append("Gal")
# q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
#         solver.solve(with_error = True, with_error_vertex = True, GF=False,\
#                     stab="SUPG", trick_second_der=False, stab_coeff=0.)
# alphas[labels[i]] = solver.stab_coeff


# # In[10]:


# styles = ["-","--", "-", "--","-.",":", "-.", ":",":"]
# colors = ["k","k","b","b","g","g","r","r","m"]

# err_curl = dict()

# for il, lab in enumerate(labels):
#     q_tmp = dict()
#     for var in problem.vars:
#         q_tmp[var] = solver.ic_vect[var]
#     curl0 = FEM2D.compute_discrete_curl_involution(q_tmp, alphas[lab],FEM2D.geom.dx[0],FEM2D.geom.dx[1])
#     err_curl[lab] = np.zeros(len(tt_save[lab]))
#     for it, t in enumerate(tt_save[lab]):
#         q_tmp = dict()
#         for var in problem.vars:
#             q_tmp[var] = q_save[lab][var][it]
#         curl  =  FEM2D.compute_discrete_curl_involution(q_tmp, alphas[lab],FEM2D.geom.dx[0],FEM2D.geom.dx[1])
#         curl_diff = curl-curl0
#         err_curl[lab][it] =  curl_diff.T @ (FEM2D.operator["mass"] @ curl_diff)

#     plt.semilogy(tt_save[lab][1:-1], err_curl[lab][1:-1], label=lab, linestyle=styles[il], color=colors[il])
# plt.legend()
# plt.ylabel(r"$|| \nabla \times u^{n+1} - \nabla \times u^0 ||$")
# plt.xlabel("Time")
# plt.savefig(problem.folderName+"/curl_error_ord_%d_N_%04d.pdf"%(order,Ns[0]))


# # In[11]:


# from gfsupg.solver import SUPG_GF_stabilization,define_GF_residuals


# # In[12]:


# lab = "SUPG_GF"
# GF=True
# trick = False
# stab ="SUPG"
# q_save[lab], tt_save[lab], comp_time[lab], err[lab], err_vertex[lab]  = \
#                 solver.solve(with_error = True, with_error_vertex = True, GF=GF,\
#                 stab=stab, trick_second_der=trick)

# q_tmp = dict()
# gal_res = dict()
# all_sources = dict()
# res = dict()
# for var in problem.vars:
#     q_tmp[var]      = np.zeros((1,len(solver.ic_vect[var])))
#     res[var]        = np.zeros(len(solver.ic_vect[var]))
#     q_tmp[var][0,:] = solver.ic_vect[var]
#     gal_res[var]    = np.zeros((1,len(solver.ic_vect[var])))
#     all_sources[var]= np.zeros((len(solver.ic_vect[var])))
# SUPG_GF_stabilization(gal_res, q_tmp,all_sources, 0, FEM2D.operator, 1, FEM2D.geom.dx_min, solver.stab_coeff, np.array([1.]), 1. )

# for var in problem.vars:
#     res[var][:] = gal_res[var][0,:]

# plot_one_sol(problem, FEM2D, res)


# # In[13]:


# lab = "SUPG_GF"

# q_tmp = dict()
# for var in problem.vars:
#     q_tmp[var] = solver.ic_vect[var]

# curl0 = FEM2D.compute_discrete_curl_involution(q_tmp, alphas[lab],FEM2D.geom.dx[0],FEM2D.geom.dx[1])

# for it, t in enumerate(tt_save[lab][:3]):
#     plot_all_sols(problem, FEM2D,q_save[lab],it, t , 200)
#     q_tmp = dict()
#     for var in problem.vars:
#         q_tmp[var] = q_save[lab][var][it]
#     curl  =  FEM2D.compute_discrete_curl_involution(q_tmp, alphas[lab],FEM2D.geom.dx[0],FEM2D.geom.dx[1])
#     fig, axs = plt.subplots(1,2,figsize=(10,4))
#     plot_sol(FEM2D, curl, axs[0], fig, levels=200)
#     axs[0].set_title(r"$\nabla \times u$")
#     plot_sol(FEM2D, curl-curl0, axs[1], fig, levels=200)
#     axs[1].set_title(r"$\nabla \times u-\nabla \times u^0 $")



# # In[14]:


# for stab_coeff in [0.,1e-4,1e-3]:
#       q_save_OSS, tt_save_OSS, comp_time_OSS, err_OSS, err_vertex_OSS  =\
#       solver.solve(with_error = True, with_error_vertex = True, GF=True, \
#                    stab="OSS", stab_coeff = stab_coeff)
#       print("OSS coeff = ",stab_coeff)
#       print("error ", err_OSS)


# # In[15]:


# fig,axs = plt.subplots(1,3,figsize=(15,4))
# plot_sol(FEM2D, solver.ic_vect["u"], axs[0], fig)
# plot_sol(FEM2D, solver.ic_vect["v"], axs[1], fig)
# plot_sol(FEM2D, solver.ic_vect["p"], axs[2], fig)
# fig.suptitle("Initial condition")


# # In[16]:


# fig,axs = plt.subplots(1,1,figsize=(5,4))
# plot_sol(FEM2D, solver.ic_vect["u"]**2+solver.ic_vect["v"]**2, axs, fig, levels=100)

# fig.suptitle("Velocity squared initial condition")


# # In[17]:


# for it, t in enumerate(tt_save_OSS[1::4]):
#     plot_all_sols(problem, FEM2D,q_save_OSS,it, t , 200)


# # In[18]:


# for it, t in enumerate(tt_save["SUPG"][1::4]):
#     plot_all_sols(problem, FEM2D,q_save["SUPG"],it, t , 200)


# # In[19]:


# q_save, tt_save, comp_time, err, err_vertex  = solver.solve(with_error = True, with_error_vertex = True, GF=False)
# q_save_GF, tt_save_GF, comp_time_GF, err_GF, err_vertex_GF  \
#     = solver.solve(with_error = True, with_error_vertex = True, GF=True)


# # In[20]:


# for it, t in enumerate(tt_save[1::4]):
#     fig = plot_all_sols(problem, FEM2D,q_save,it, t , 200)
#     fig.suptitle("NGF Time %1.4f"%t)
#     plt.show()
#     fig = plot_all_sols(problem, FEM2D,q_save_GF,it, t , 200)
#     fig.suptitle("GF Time %1.4f"%t)
#     plt.show()


# # In[21]:


# u_ex = FEM2D.evaluate_function(lambda x,y: problem.exact["u"](x,y,0))
# v_ex = FEM2D.evaluate_function(lambda x,y: problem.exact["v"](x,y,0))

# fig,axs = plt.subplots(1,3,figsize=(15,4))
# plot_sol(FEM2D, v_ex, axs[0], fig, levels=100)
# plot_sol(FEM2D, q_save_GF["u"][-1,:]-u_ex, axs[1], fig, levels=100)
# plot_sol(FEM2D, q_save_GF["v"][-1,:]-v_ex, axs[2], fig, levels=100)


# # In[22]:


# fig,axs = plt.subplots(1,3,figsize=(15,4))
# plot_sol(FEM2D, solver.ic_vect["u"]**2+solver.ic_vect["v"]**2, axs[0], fig, levels=100)
# plot_sol(FEM2D, q_save_GF["u"][-1,:]**2+q_save_GF["v"][-1,:]**2, axs[1], fig, levels=100)
# plot_sol(FEM2D, q_save["u"][-1,:]**2+q_save["v"][-1,:]**2, axs[2], fig, levels=100)

# fig.suptitle("Velocity squared initial (left), final GF (center), final NGF (right)")


# ## Solving perturbation

# In[23]:


problem = SmoothVortexTestCase(pert_coeff=1e-10, is_smaller= True, pert_type = "opt") #LinearAcoustic2D("smaller_smooth_vortex_opt_perturbation",pert_coeff=1e-10)


# In[24]:


Ns = np.array([40,40], dtype=np.int32)

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)



# In[25]:


solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)
#solver.set_save_slabs(200)
solver.set_Nt_max(1e6)
q_save, tt_save, comp_time, err, err_vertex  = solver.solve(with_error = True, with_error_vertex = True, GF=False)

div_errors = FEM2D.compute_discrete_divergence_norms(q_save)
np.mean(np.abs(div_errors))




# In[ ]:


source = FEM2D.compute_sources(solver.ic_vect,problem)


# In[ ]:


solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
q_save_GF, tt_save_GF, comp_time_GF, err_GF, err_GF_vertex = solver.solve(with_error = True, with_error_vertex = True, GF=True)
print(err_GF)
print(err_GF_vertex)

div_errors_GF = FEM2D.compute_discrete_divergence_norms(q_save_GF)
np.mean(np.abs(div_errors_GF))


# In[ ]:


plt.semilogy(tt_save, div_errors,label="SUPG")
plt.semilogy(tt_save_GF, div_errors_GF,label="SUPG-GF")
plt.legend()


# In[ ]:


fig,axs = plt.subplots(1,3,figsize=(15,4))
plot_sol(FEM2D, solver.ic_vect["u"], axs[0], fig)
plot_sol(FEM2D, solver.ic_vect["v"], axs[1], fig)
plot_sol(FEM2D, solver.ic_vect["p"], axs[2], fig)
for iv, var in enumerate(problem.vars):
    axs[iv].set_title(var)


# In[ ]:


fig,axs = plt.subplots(1,3, figsize=(15,4))

zz = FEM2D.operator["IDx_tilde"]@solver.ic_vect["u"]+\
FEM2D.operator["IDy_tilde"]@solver.ic_vect["v"]
plot_sol(FEM2D, zz , axs[0], fig)
axs[0].set_title("Div")

ww = FEM2D.operator["DxDx2_tilde"]@solver.ic_vect["u"]+\
FEM2D.operator["DxDy_tilde"]@solver.ic_vect["v"]
plot_sol(FEM2D, ww , axs[1], fig)
axs[1].set_title("\partial_x div")


qq = FEM2D.operator["DyDx_tilde"]@solver.ic_vect["u"]+\
FEM2D.operator["DyDy2_tilde"]@solver.ic_vect["v"]
plot_sol(FEM2D, qq , axs[2], fig)
axs[2].set_title("\partial_y div")


# In[ ]:


plot_all_sols(problem, FEM2D, q_save, 1, tt_save[1], levels=None);
plot_all_sols(problem, FEM2D, q_save_GF, 1, tt_save_GF[1], levels=None);


# In[ ]:


plot_all_sols(problem, FEM2D, q_save, len(q_save)-1, tt_save[len(q_save)-1], levels=None);
plot_all_sols(problem, FEM2D, q_save_GF, len(q_save_GF)-1, tt_save_GF[len(q_save_GF)-1], levels=None);


# In[ ]:


diff   = dict()
diffGF = dict()
for var in problem.vars:
    diff[var] = q_save[var]-solver.ic_no_pert[var]
    diffGF[var]   = q_save_GF[var]-solver.ic_no_pert[var]
v2   = q_save["u"][-1,:]**2+q_save["v"][-1,:]**2
v2GF = q_save_GF["u"][-1,:]**2+q_save_GF["v"][-1,:]**2
v2IC = solver.ic_no_pert["u"]**2+ solver.ic_no_pert["v"]**2

diffv2   = v2  -v2IC
diffv2GF = v2GF-v2IC

plot_all_sols(problem, FEM2D, diff, len(q_save)-1, tt_save[len(q_save)-1], levels=None)
plot_all_sols(problem, FEM2D, diffGF, len(q_save_GF)-1, tt_save_GF[len(q_save_GF)-1], levels=None)

fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, np.abs(diffv2) , axs[0],fig, levels=21)
plot_sol(FEM2D, np.abs(diffv2GF),axs[1],fig, levels=21)
plot_sol(FEM2D, np.abs(v2IC),axs[2],fig, levels=21)
axs[0].set_title("Non Global Flux")
axs[1].set_title("Global Flux")
plt.show()

it = -1
fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, diff["u"][it,:]**2+diff["v"][it,:]**2 , axs[0],fig, levels=21)
plot_sol(FEM2D, diffGF["u"][it,:]**2+diffGF["v"][it,:]**2,axs[1],fig, levels=21)
plot_sol(FEM2D, np.abs(v2IC),axs[2],fig, levels=21)
axs[0].set_title("Non Global Flux")
axs[1].set_title("Global Flux")
plt.show()


# In[ ]:


FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)


# In[ ]:


solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
solver.set_save_slabs(200)
solver.set_Nt_max(1e6)
q_save, tt_save,  comp_time, err, err_vertex = solver.solve(stab_coeff=0.05,with_error = True, with_error_vertex = True)
print(err)


# #### Tentative FEM (with DeC+SUPG)
# * Weak formulation
# $$
# \begin{cases}
#     \sum_K \int_K\left( \varphi_i +\alpha h_K \partial_x \varphi_i \right)\left( \partial_t u +\partial_x p\right)=0\\
#     \sum_K \int_K\left( \varphi_i +\alpha h_K \partial_y \varphi_i \right)\left( \partial_t v +\partial_y p\right)=0\\
#     \sum_K \int_K\left( \varphi_i +\alpha h_K \partial_x \varphi_i+\alpha h_K\partial_y \varphi_i \right)\left( \partial_t p +\partial_x u + \partial_y v \right)=0
# \end{cases}
# $$
# * Let us define the residual quantities of each equation and its vector
# $$
# r(u,v,p):=
# \begin{cases}
#     r_u(u,v,p):= \partial_t u +\partial_x p\\
#     r_v(u,v,p):= \partial_t v +\partial_y p\\
#     r_v(u,v,p):= \partial_t p +\partial_x u +\partial_y v
# \end{cases}
# $$
# and the time descrete equivalent in each subtime node $m=1,\dots, M$ and high order time discretization
# $$
# r^m(u,v,p):=\begin{cases}
#     r^m_u(u,v,p):= \frac{u^m-u^0}{\Delta t} + \sum_r \theta^m_r \partial_x p^r\\
#     r^m_v(u,v,p):= \frac{v^m-v^0}{\Delta t} +\sum_r \theta^m_r\partial_y p^r\\
#     r^m_v(u,v,p):= \frac{p^m-p^0}{\Delta t} + \sum_r \theta^m_r (\partial_x u^r +\partial_y v^r)
# \end{cases}
# $$
# 
# * DeC on top of that
#     * $\mathcal L^1$
#     $$
#         \mathcal L^1(u,v,p) = \begin{cases}
#                 |C_i| \frac{u_i^{m}-u_i^{0}}{\Delta t} + \sum_K \int_K \varphi_i \partial_x p^{0}\\
#     |C_i| \frac{v_i^{m}-v_i^{0}}{\Delta t}+\sum_K \int_K \varphi_i  \partial_y p^{0}\\
#     |C_i| \frac{p_i^{m}-p_i^{0}}{\Delta t} +\sum_K \int_K  \varphi_i   (\partial_x u^0 + \partial_y v^0) 
#         \end{cases}
#     $$
#     * $\mathcal L^2$
#     $$
#         \mathcal L^{2,m}_i(u,v,p) = \int \left(\varphi_i + \alpha h \partial_x\varphi_i\, \text{sign}(J f^x) + \alpha h \partial_y \varphi_i\, \text{sign}(J f^y)   \right) r^m(u,v,p) dx 
#     $$
# 

# In Einstein notation:
# * Conservation Law $i = 1,\dots, N_{eq}$, $d=1,\dots,N_{\text{dim}}$
# $$
# \partial_t u^i + \partial_{x_d} f^{d,i}(u) =0.
# $$
# * SUPG option 1   $i,j,r, \ell = 1,\dots, N_{eq}$, $d,k=1,\dots,N_{\text{dim}}$
# $$
# \sum_K\int_K\left( \varphi^i + \alpha_{\ell j} \partial_{u^j} f^{k,r}(u) \partial_{x_k} \varphi^r \right) \left(\partial_t u^j + \partial_{x_d} f^{d,j}(u) \right)
# $$
# 
# * SUPG option 2   $i,j,r = 1,\dots, N_{eq}$, $d,k=1,\dots,N_{\text{dim}}$
# $$
# \sum_K\int_K\left( \varphi^i + \alpha_{ij} \partial_{u^r} f^{k,j}(u) \partial_{x_k} \varphi^r \right) \left(\partial_t u^j + \partial_{x_d} f^{d,j}(u) \right)
# $$

# In[ ]:


solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
solver.set_save_slabs(200)
solver.set_Nt_max(1e6)
q_save, tt_save, comp_time, err, err_vertex = solver.solve(with_error = True)
print(err)
div_errors = FEM2D.compute_discrete_divergence_norms(q_save)


# In[ ]:


solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=True)
solver.set_save_slabs(200)
solver.set_Nt_max(1e6)
q_save_GF, tt_save_GF, comp_time_GF, err_GF, err_GF_vertex = solver.solve(with_error = True)
print(err_GF)
div_errors_GF = FEM2D.compute_discrete_divergence_norms(q_save_GF)


# In[ ]:


plt.semilogy(tt_save, div_errors,label="SUPG")
plt.semilogy(tt_save_GF, div_errors_GF,label="SUPG-GF")
plt.legend()


# In[ ]:


print(compute_integral_error(problem, FEM2D, q_save, tt_save))

Nt_save = len(tt_save)
err = np.zeros((Nt_save,len(problem.vars)))
rel_err = np.zeros((Nt_save,len(problem.vars)))
for it in range(Nt_save):
    err[it,:], rel_err[it,:] = compute_errors(problem, FEM2D, q_save, it, tt_save[it])

plt.figure()
plt.semilogy(tt_save[:-1], err[:-1], label=[var + " NGF" for var in problem.vars])


err = np.zeros((Nt_save,len(problem.vars)))
rel_err = np.zeros((Nt_save,len(problem.vars)))
for it in range(Nt_save):
    err[it,:], rel_err[it,:] = compute_errors(problem, FEM2D, q_save_GF, it, tt_save_GF[it])

plt.semilogy(tt_save[:-1], err[:-1], "--", label=[var + " GF" for var in problem.vars])

plt.legend()
plt.title("Absolute errors")

# plt.figure()
# plt.plot(tt_save[:-1], rel_err[:-1], label=problem.vars)
# plt.legend()
# plt.title("Relative errors")
print(err)


# #### Plot solutions

# In[ ]:


levels = np.linspace(-1,1,14)

for it in range(0,135,20):
    plot_all_sols(problem, FEM2D, q_save, it, tt_save[it])#,levels)


# In[ ]:


levels = np.linspace(-1,1,14)


for it in range(0,135,20):
    plot_all_sols(problem, FEM2D, q_save_GF, it, tt_save_GF[it])#,levels)


# #### Plot exact

# In[ ]:


for it in range(0,Nt_save):
    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    t = tt_save[it]
    for k, var in enumerate(problem.vars):
        ex = FEM2D.evaluate_function(lambda x,y: problem.exact[var](x,y,t))
        axs[k].set_title(var)
        fig.suptitle("Time %1.4f"%t)
        plot_sol(FEM2D,ex, axs[k],fig, levels)


# #### Plot error

# In[ ]:


for it in range(0,Nt_save,3):
    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    t = tt_save[it]
    for k, var in enumerate(problem.vars):
        ex = FEM2D.evaluate_function(lambda x,y: problem.exact[var](x,y,t))
        axs[k].set_title(var)
        plot_sol(FEM2D,q_save[var][it,:]-ex, axs[k],fig)
    fig.suptitle("Time %1.4f"%t)
