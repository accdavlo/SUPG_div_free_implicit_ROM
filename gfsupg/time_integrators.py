import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from .quadr import lglnodes,equispaced, lgwt
import numba
from numba.experimental import jitclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


specs=[
        ('name', numba.types.string),
        ('distribution', numba.types.string),
        ('order', numba.int32),
        ('K_corr', numba.int32),
        ('M_sub', numba.int32),
        ('ARK',numba.float64[:,:]),
        ('bRK',numba.float64[:]),
        ('SDIRK',numba.boolean)]
@jitclass(specs)
class TimeIntegrator:
    """
    In this class we define the coefficients of the time integrator
    There some implicit classes (like IMEX_DeC, Lobatto,GaussLegendre, 
    SDRIK and DIRK and BDF (not working) methods ) 
    add the order in the name string, e.g. IMEX_DeC3 Lobatto6 DIRK3
    Other working schemes not listed here are CrankNicolson and CrankNicolson_Rosenbrock
    And explicit RK schemes (ERK)
    """
    def __init__(self, name):
        self.name = name
        self.order = search_order(self.name)
        if "IMEX_DeC" in self.name:
            self.K_corr = self.order
            self.M_sub = np.int32(np.ceil(self.K_corr/2))
            self.distribution = "gaussLobatto"
        elif "Lobatto" in self.name:
            self.M_sub = np.int32(np.ceil(self.order/2))
            self.distribution = "gaussLobatto"
        elif "GaussLegendre" in self.name:
            self.M_sub = np.int32(np.ceil((self.order-2)/2))
            self.distribution = "gaussLegendre"
        elif "SDIRK" in self.name:
            self.SDIRK = True
            if self.order ==1:
                self.ARK = np.array([[1.]])
                self.bRK = np.array([1.]) 
            elif self.order==2 and "Crouzeix" in self.name:
                x=0.5+np.sqrt(3.)/6.
                self.ARK = np.array([[x,0],[-np.sqrt(3.)/3.,x]])
                self.bRK = np.array([0.5,0.5])
            elif self.order==2:
                x=1./4.
                self.ARK = np.array([[x,0],[1-x,x]])
                self.bRK = np.array([1-x,x])
            elif self.order == 3:
                x = 0.4358665215
                y = -3*x**2/2.+4*x-0.25
                z =  3*x**2/2.-5*x+1.25
                self.ARK = np.array([[x,0,0],[(1.-x)/2.,x,0],\
                                     [y,z,x]])
                self.bRK = np.array([y,z,x])
            elif self.order == 4: # N\{o}rsett
                x = 1.06858
                y = 3.*(1-2.*x)**2
                self.ARK = np.array([[x,0,0],[1./2.-x,x,0],\
                                     [2*x,1-4*x,x]])
                self.bRK = np.array([1./2./y,(y-1.)/y,1./2./y])
        elif "DIRK" in self.name:
            self.SDIRK = False
            if self.order ==1:
                self.ARK = np.array([[1.]])
                self.bRK = np.array([1.]) 
            elif self.order == 2 and "CrankNicolson" in self.name:
                #"DIRK2CrankNicolson"
                self.ARK = np.array([[0.,0],[1./2.,1./2.]])
                self.bRK = np.array([0.5,0.5])
            elif self.order == 2:
                self.ARK = np.array([[0.5,0],[-1./2.,2.]])
                self.bRK = np.array([-0.5,1.5])
            elif self.order == 3:
                x = 0.4358665215
                y = -3*x**2/2.+4*x-0.25
                z =  3*x**2/2.-5*x+1.25
                self.ARK = np.array([[x,0,0],[(1.-x)/2.,x,0],\
                                     [y,z,x]])
                self.bRK = np.array([y,z,x])
            elif self.order == 4: 
                # N\{o}rsett
                x = 1.06858
                y = 3.*(1-2.*x)**2
                self.ARK = np.array([[x,0,0],[1./2.-x,x,0],\
                                     [2*x,1-4*x,x]])
                self.bRK = np.array([1./2./y,(y-1.)/y,1./2./y])
        elif "ERK" in self.name:
            if self.order ==1:
                self.ARK = np.array([[0.]])
                self.bRK = np.array([1.])
            elif self.order==2 and "midpoint" in self.name: 
                #"ERK2midpoint"
                self.ARK = np.array([[0.,0.],[0.5,0.]])
                self.bRK = np.array([0.,1.])
            elif self.order == 2:
                self.ARK = np.array([[0.,0.],[1.,0.]])
                self.bRK = np.array([0.5,0.5])
            elif self.order==3 and "SSP" in self.name: 
                #"ERK3SSP"
                self.ARK = np.array([[0.,0.,0.],[1.,0.,0.],[0.25,0.25,0.]])
                self.bRK = np.array([1./6.,1./6.,2./3.])
            elif self.order==3:
                self.ARK = np.array([[0.,0.,0.],[0.5,0.,0.],[-1.,2.,0.]])
                self.bRK = np.array([1./6.,2./3.,1./6.])
            elif self.order==4:
                self.ARK = np.array([[0.,0.,0.,0.],\
                                     [0.5,0.,0.,0.],\
                                     [0.,0.5,0.,0.],\
                                     [0.,0.,1.,0.]])
                self.bRK = np.array([1./6.,1./3.,1./3.,1./6.])



@numba.njit
def search_order(name):
    # Iterates exactly through the string representations of 1-9
    for i in range(1, 10):
        # Numba supports converting single integers to strings inside loops
        if str(i) in name:
            return i
            
    return 1

@numba.njit
def lagrange_basis(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
        y[ix]=1.
        for z in range(len(tmp)):
            y[ix]=y[ix]*tmp[z]
    return y

# def lagrange_basis(nodes,x,k):
#     y=np.zeros(x.size)
#     for ix, xi in enumerate(x):
#         tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
#         y[ix]=np.prod(tmp)
#     return y


@numba.njit
def get_nodes(order,nodes_type):
    if nodes_type=="equispaced":
        nodes,w = equispaced(order)
    elif nodes_type == "gaussLegendre":
        nodes,w = lgwt(order)
    elif nodes_type == "gaussLobatto":
        nodes, w = lglnodes(order-1,10**-15)
    nodes=nodes*0.5+0.5
    w = w*0.5
    return nodes, w


@numba.njit
def compute_theta_DeC(order, nodes_type):
    nodes, w = get_nodes(order,nodes_type)
    if nodes_type=="gaussLobatto":
        int_nodes, int_w = get_nodes(order,"gaussLobatto")
    else:
        int_nodes, int_w = get_nodes(order,"gaussLegendre")
    # generate theta coefficients
    theta = np.zeros((order,order))
    beta = np.zeros(order)
    recon = np.zeros(order)
    t_end = np.array([1.])
    for m in range(order):
        beta[m] = nodes[m]
        nodes_m = int_nodes*(nodes[m])
        w_m = int_w*(nodes[m])
        for r in range(order):
            ff=lagrange_basis(nodes,nodes_m,r)
            theta[r,m] = np.sum(ff*w_m)
        zz =lagrange_basis(nodes, t_end, m)
        recon[m] = zz[0]
    return theta, beta, recon


def compute_RK_from_DeC(M_sub,K_corr,nodes_type):
    order=M_sub+1;
    theta,beta, recon=compute_theta_DeC(order,nodes_type)
    bar_beta=beta[1:]  # M_sub
    bar_theta=theta[:,1:].transpose() # M_sub x (M_sub +1)
    theta0= bar_theta[:,0]  # M_sub x 1
    bar_theta= bar_theta[:,1:] #M_sub x M_sub
    A=np.zeros((M_sub*(K_corr-1)+1,M_sub*(K_corr-1)+1))  # (M_sub x K_corr +1)^2
    b=np.zeros(M_sub*(K_corr-1)+1)
    c=np.zeros(M_sub*(K_corr-1)+1)

    c[1:M_sub+1]=bar_beta
    A[1:M_sub+1,0]=bar_beta
    for k in range(1,K_corr-1):
        r0=1+M_sub*k
        r1=1+M_sub*(k+1)
        c0=1+M_sub*(k-1)
        c1=1+M_sub*(k)
        c[r0:r1]=bar_beta
        A[r0:r1,0]=theta0
        A[r0:r1,c0:c1]=bar_theta
    b[0]=theta0[-1]
    b[-M_sub:]=bar_theta[M_sub-1,:]
    return A,b,c


def dec(func, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))
    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                rhs[:,r]=func(u_p[:,r])
            for m in range(1,M_sub+1):
                u_a[:,m]= U[:,it-1]+delta_t*sum([Theta[r,m]*rhs[:,r] for r in range(M_sub+1)])
        U[:,it]=0.
        for m in range(M_sub+1):
            U[:,it]+=recon[m]*u_a[:,m]
    return tspan, U

# needs more debugging, converges only 1st order regardless of M_sub and K_corr
def decIMEX3(funcEX, funcIM_a0, funcIM_a1, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhsEX= np.zeros((dim,M_sub+1))
    rhsIM = np.zeros((dim,M_sub+1))
    a0 = np.zeros((dim,M_sub+1))
    a1 = np.zeros((dim,M_sub+1))
    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    # set U as initial
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
            a0[:,m] = funcIM_a0(u_p[:,m])
            a1[:,m] = funcIM_a1(u_p[:,m])
        for k in range(1,K_corr+1):
            #for m in range (M_sub+1):
              #a0[:,m] = funcIM_a0(u_p[:,m])
              #a1[:,m] = funcIM_a1(u_p[:,m])
            u_p=np.copy(u_a)
            #Explicit update
            for r in range(M_sub+1):
                rhsEX[:,r]=funcEX(u_p[:,r])
                rhsIM[:,r]=funcIM_a0(u_p[:,r]) + funcIM_a1(u_p[:,r])*u_p[:,r]
            for m in range(1,M_sub+1):
                L2=(u_p[:,m]-u_p[:,0])\
                -delta_t*sum([Theta[r,m]*(rhsEX[:,r]+rhsIM[:,r]) for r in range(M_sub+1)])
                u_a[:,m]= np.divide(u_p[:,m] - delta_t*beta[m]*a1[:,m]*u_p[:,m] \
                - L2 \
                ,(np.ones(dim)-delta_t*beta[m]*a1[:,m]))
        U[:,it]=0.
        for m in range(M_sub+1):
            U[:,it]+=recon[m]*u_a[:,m]
    return tspan, U


#IMEX2 WORKS with jacobian_im
def decIMEX2(funcEX, funcIM, jacIM, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    func_corr=np.zeros((dim,M_sub+1))
    func_corr_both=np.zeros((dim,M_sub+1))
    invJac = np.zeros((M_sub+1,dim,dim))
    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    # set U as initial
    # Python start with 0
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
        SS = jacIM(u_p[:,0])
        for m in range(1,M_sub+1):
            invJac[m,:,:] = np.linalg.inv(np.eye(dim) - delta_t*beta[m]*SS)
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                func_corr[:,r]=funcEX(u_p[:,r])
                func_corr_both[:,r]=func_corr[:,r] + funcIM(u_p[:,r])
            for m in range(1,M_sub+1):
                L2=(u_p[:,m]-u_p[:,0])\
                -delta_t*sum([Theta[r,m]*func_corr_both[:,r]\
                for r in range(M_sub+1)])
                u_a[:,m] = u_p[:,m] -np.matmul(invJac[m,:,:],L2)
        U[:,it]=0.
        for m in range(M_sub+1):
            U[:,it]+=recon[m]*u_a[:,m]
    return tspan, U

def EulerIMEX(funcEX, funcIM_a0, funcIM_a1, tspan, y_0):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    rhsEX= np.zeros((dim,1))
    a0 = np.zeros((dim,1))
    a1 = np.zeros((dim,1))

    # set U as initial
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        rhsEX = funcEX(U[:,it-1])
        U[:,it]=U[:,it-1] + delta_t*rhsEX
        a0 = funcIM_a0(U[:,it])
        a1 = funcIM_a1(U[:,it])
        U[:,it] = np.divide(U[:,it] + delta_t*a0,1. - delta_t*a1)

    return tspan, U

def decImplicit(func,jac_stiff, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))
    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    invJac=np.zeros((M_sub+1,dim,dim))
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
        SS=jac_stiff(u_p[:,0])
        for m in range(1,M_sub+1):
            invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*beta[m]*SS)
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                rhs[:,r]=func(u_p[:,r])
            for m in range(1,M_sub+1):
                u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                (-(u_p[:,m]-u_p[:,0])/delta_t\
                 +sum([Theta[r,m]*rhs[:,r] for r in range(M_sub+1)])))
        U[:,it]=0.
        for m in range(M_sub+1):
            U[:,it]+=recon[m]*u_a[:,m]
    return tspan, U


@numba.njit
def generic_time_scheme(lhs_res_ODE,lhs_res_ODE_BDF,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u, \
                time_integrator_class, Nsave, *func_args):

    if "IMEX_DeC" in time_integrator_class.name:
        tt,uu = decImplicit_variable_dt\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u, \
            time_integrator_class.M_sub,time_integrator_class.K_corr,time_integrator_class.distribution,\
            Nsave, *func_args)
    elif ("Lobatto" in time_integrator_class.name) or \
         ("GaussLegendre" in time_integrator_class.name):
        tt,uu = decImplicit_variable_dt_convergence\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u, \
            time_integrator_class.M_sub,time_integrator_class.distribution,\
            Nsave, *func_args)
    elif "BDF" in time_integrator_class.name:
        tt,uu = BDF\
        (lhs_res_ODE_BDF,rhs_res_ODE,jac_lhs,jac_rhs,\
            spectral_radius,CFL,tspan, u, \
            time_integrator_class.order,\
            Nsave, *func_args)
    elif time_integrator_class.name=="implicit_euler":
        tt,uu = implicit_Euler_variable_dt\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u,\
         Nsave, *func_args)
    elif time_integrator_class.name=="CrankNicolson":
        tt,uu = CrankNicolson\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u, \
            Nsave, *func_args)
    elif time_integrator_class.name=="CrankNicolson_Rosenbrock":
        tt,uu = CrankNicolson_Rosenbrock\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u,\
         Nsave, *func_args)
    elif "DIRK" in time_integrator_class.name:
        tt,uu = DIRK\
        (lhs_res_ODE,rhs_res_ODE,jac_lhs,jac_rhs,spectral_radius,CFL,tspan, u,\
            time_integrator_class.ARK, time_integrator_class.bRK, time_integrator_class.SDIRK ,\
            Nsave, *func_args)
    elif "ERK" in time_integrator_class.name:
        tt,uu = ERK\
        (rhs_res_ODE,jac_lhs,spectral_radius,CFL,tspan, u,\
            time_integrator_class.ARK, time_integrator_class.bRK, time_integrator_class.SDIRK ,\
            Nsave, *func_args)
    else:
        raise ValueError("Time integrator not implemented in numba")
    return tt, uu

@numba.njit
def decImplicit_variable_dt(lhs_res, rhs_res, lhs_jac, rhs_jac, spect_radius, CFL, tspan, y_0, M_sub, K_corr, distribution, Nsave=100,*func_args):
    """
    DeC implicit or IMEX for stiff problems. Time integration routine. Evolves u'=f(t,u), with the Jacobian of the stiff part of f being J
    Input: func(t,u,args) evolution function, jac_stiff(t,u,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    M_sub number of subtimestep intervals, K_corr number of iterations,
    distribution of the subtimesteps among "equispaced", "gaussLobatto", "gaussLegendre" 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    dim=len(y_0)
    M_sub1=M_sub+1

    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))

    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    invJac=np.zeros((M_sub+1,dim,dim))
    Jac=np.zeros((M_sub+1,dim,dim))

    un  = np.copy(y_0)
    un1 = np.copy(y_0)

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_sub = beta*delta_t+time

        time+=delta_t

        for m in range(M_sub+1):
            u_a[:,m]=un
            u_p[:,m]=un
        jac_ll = lhs_jac(t_sub[0], u_p[:,0], delta_t, *func_args)
        jac_rr = rhs_jac(t_sub[0], u_p[:,0], *func_args)
        for m in range(1,M_sub+1):
            #invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*beta[m]*SS)
            Jac[m,:,:]=jac_ll - beta[m]*jac_rr
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                rhs[:,r]=rhs_res(t_sub[r],u_p[:,r],*func_args)
            for m in range(1,M_sub+1):
                rhs_tot = np.zeros(dim) 
                for r in range(M_sub+1):
                    rhs_tot =rhs_tot +Theta[r,m]*rhs[:,r]
                # u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                # (-(u_p[:,m]-u_p[:,0])/delta_t+rhs_tot))
                lhs = lhs_res(t_sub[r],u_p[:,0],u_p[:,m], delta_t,*func_args)
                L2 = lhs-rhs_tot 
                u_a[:,m]= u_p[:,m]-np.linalg.solve(Jac[m,:,:],L2)
        un[:]=0.
        for m in range(M_sub+1):
            un+=recon[m]*u_a[:,m]
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll




@numba.njit
def decImplicit_variable_dt_convergence(lhs_res, rhs_res, lhs_jac, rhs_jac,\
    spect_radius, CFL, tspan, y_0, M_sub, distribution, Nsave=100,*func_args):
    """
    DeC implicit or IMEX for stiff problems. Time integration routine. Evolves u'=f(t,u), with the Jacobian of the stiff part of f being J
    Input: func(t,u,args) evolution function, jac_stiff(t,u,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    M_sub number of subtimestep intervals, K_corr number of iterations,
    distribution of the subtimesteps among "equispaced", "gaussLobatto", "gaussLegendre" 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    max_newton_iter = 50
    newton_tolerance = 1e-7

    dim=len(y_0)
    M_sub1=M_sub+1

    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))

    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    Jac=np.zeros((dim,dim))

    un  = np.copy(y_0)
    un1 = np.copy(y_0)

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_sub = beta*delta_t+time

        time+=delta_t
        newton_iter=0
        for m in range(M_sub+1):
            u_a[:,m]=un
            u_p[:,m]=un

        for k in range(1,max_newton_iter+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                rhs[:,r]=rhs_res(t_sub[r],u_p[:,r],*func_args)

            # Checks immediately last step
            m=M_sub
            rhs_tot = np.zeros(dim) 
            for r in range(M_sub+1):
                rhs_tot =rhs_tot +Theta[r,m]*rhs[:,r]
            jac_ll = lhs_jac(t_sub[m], u_a[:,m], delta_t, *func_args)
            jac_rr = rhs_jac(t_sub[m], u_a[:,m], *func_args)
            Jac=jac_ll - beta[m]*jac_rr
            # u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
            # (-(u_p[:,m]-u_p[:,0])/delta_t+rhs_tot))
            lhs = lhs_res(t_sub[r],un,u_p[:,m], delta_t,*func_args)
            L2 = lhs-rhs_tot 
            res_norm = np.linalg.norm(L2)/np.sqrt(dim)
            if res_norm<newton_tolerance:
                break

            u_a[:,m]= u_p[:,m]-np.linalg.solve(Jac,L2)

            #Updates other steps
            for m in range(M_sub):
                rhs_tot = np.zeros(dim) 
                for r in range(M_sub+1):
                    rhs_tot =rhs_tot +Theta[r,m]*rhs[:,r]
                jac_ll = lhs_jac(t_sub[m], u_a[:,m], delta_t, *func_args)
                jac_rr = rhs_jac(t_sub[m], u_a[:,m], *func_args)
                Jac = jac_ll - beta[m]*jac_rr
                # u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                # (-(u_p[:,m]-u_p[:,0])/delta_t+rhs_tot))
                lhs = lhs_res(t_sub[m],un,u_p[:,m], delta_t,*func_args)
                L2 = lhs-rhs_tot 
                u_a[:,m]= u_p[:,m]-np.linalg.solve(Jac,L2)

            if  k == max_newton_iter:
                with numba.objmode():
                    print("Newton not converged, time = %5.3f, residual = %1.3e"%(time,res_norm), end="\n")
                    print("", end="\n")
                if res_norm>1000:
                    return times, uAll
 
        un[:]=0.
        for m in range(M_sub+1):
            un+=recon[m]*u_a[:,m]
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll



@numba.njit
def CrankNicolson_Rosenbrock(lhs_res, rhs_res, lhs_jac, rhs_jac, spect_radius, CFL, tspan, y_0, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    rhs= np.zeros(dim)

    invJac=np.zeros((dim,dim))
    Jac=np.zeros((dim,dim))

    un  = np.copy(y_0)
    uns = np.copy(y_0)
    un1 = np.copy(y_0)


    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        time+=delta_t

        SS=  lhs_jac(t_old, un, delta_t, *func_args) - \
        rhs_jac(t_old,un, *func_args)
        
        rhs_n = rhs_res(t_old,un,*func_args)
        L2  = -rhs_n
        uns = un - np.linalg.solve(SS,L2)

        SS=lhs_jac(t_old, uns, delta_t, *func_args) - \
        rhs_jac(t_old,uns, *func_args)
        lhs = lhs_res(t_old,un,uns, delta_t,*func_args)
        rhs_s = rhs_res(t_old,uns,*func_args)
        L2  = lhs - (rhs_n+rhs_s)/2.
        un1 = uns - np.linalg.solve(SS,L2)


        un=np.copy(un1)
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll



@numba.njit
def CrankNicolson(lhs_res, rhs_res, lhs_jac, rhs_jac, spect_radius, CFL, tspan, y_0, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    max_newton_iter = 50
    newton_tolerance = 1e-7

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    L2= np.zeros(dim)

    SS=np.zeros((dim,dim))

    un  = np.copy(y_0)
    u_iter = np.copy(y_0)


    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        time+=delta_t

        rhs_n = rhs_res(t_old,un,*func_args)
        L2  = -rhs_n
        res_norm = np.linalg.norm(L2)/np.sqrt(dim)

        u_iter = np.copy(un)
        newton_iter=0
        while(res_norm>newton_tolerance and newton_iter<max_newton_iter):
            newton_iter+=1
            SS=lhs_jac(t_old, u_iter, delta_t, *func_args) - \
                0.5*rhs_jac(t_old,u_iter, *func_args)

            u_iter = u_iter - np.linalg.solve(SS,L2)
            lhs = lhs_res(t_old,un,u_iter, delta_t,*func_args)
            rhs_it = rhs_res(t_old,u_iter,*func_args)
            L2  = lhs - (rhs_n+rhs_it)/2.
            res_norm = np.linalg.norm(L2)/np.sqrt(dim)

        if newton_iter==max_newton_iter:
            with numba.objmode():
                print("Max iteration %d of newton reached with residual %g"%(newton_iter,res_norm), flush=False)
                print("")
            
            if res_norm>1000:
                break


        un=np.copy(u_iter)
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)

    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll


@numba.njit
def DIRK(lhs_res, rhs_res, lhs_jac, rhs_jac, spect_radius, CFL, tspan, y_0, ARK, bRK, SDIRK=False, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    cRK = np.sum(ARK,axis=1)
    sRK = np.shape(ARK)[0]
    fakeLastStep = np.all(ARK[-1,:]==bRK)

    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    max_newton_iter = 50
    newton_tolerance = 1e-7

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    L2= np.zeros(dim)

    SS=np.zeros((dim,dim))

    un  = np.copy(y_0)

    uRK = np.zeros((sRK,dim))
    rhsRK = np.zeros((sRK,dim))

    for i in range(sRK):
        uRK[i,:] = y_0

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args)

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        t_sub = t_old + delta_t*cRK

        time+=delta_t

        for s in range(sRK):
            u_iter = un
            rhs_old = np.zeros(dim)
            for z in range(s):
                rhs_old = rhs_old + ARK[s,z]*rhsRK[z,:]

            rhs_now = rhs_res(t_sub[s],u_iter,*func_args)
            L2  = -rhs_old-ARK[s,s]*rhs_now

            res_norm = np.linalg.norm(L2)/np.sqrt(dim)

            if SDIRK:
                SS=lhs_jac(t_old, un, delta_t, *func_args) - \
                    ARK[0,0]*rhs_jac(t_old,un, *func_args)
            newton_iter=0
            while(res_norm>newton_tolerance and newton_iter<max_newton_iter):
                newton_iter+=1
                if not SDIRK:
                    SS=lhs_jac(t_old, u_iter, delta_t, *func_args) - \
                        ARK[s,s]*rhs_jac(t_sub[s],u_iter, *func_args)

                u_iter = u_iter - np.linalg.solve(SS,L2)
                lhs = lhs_res(t_old,un,u_iter, delta_t,*func_args)
                rhs_now = rhs_res(t_sub[s],u_iter,*func_args)
                L2  = lhs -rhs_old-ARK[s,s]*rhs_now

                res_norm = np.linalg.norm(L2)/np.sqrt(dim)

            uRK[s,:] = u_iter
            rhsRK[s] = rhs_now

            if newton_iter==max_newton_iter:
                with numba.objmode():
                    print("Max iteration %d of newton reached with residual %g"%(newton_iter,res_norm), flush=False)
                    print("Stage %d , time %g"%(s,time), flush=False)
                    print("")
                
                if res_norm>1000:
                    return times, uAll

        if fakeLastStep:
            un = uRK[-1,:]
        else: 
            rhs_old = np.zeros(dim)
            for z in range(sRK):
                rhs_old = rhs_old + bRK[z]*rhsRK[z,:]
            SS=lhs_jac(t_old, un, delta_t, *func_args)
            un = un - np.linalg.solve(SS,-rhs_old)

        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll



@numba.njit
def ERK( rhs_res, lhs_jac, spect_radius, CFL, tspan, y_0, ARK, bRK, SDIRK=False, Nsave=100,*func_args):
    """
    Explicit Runge Kutta with implicit diffusion. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    cRK = np.sum(ARK,axis=1)
    sRK = np.shape(ARK)[0]
    fakeLastStep = np.all(ARK[-1,:]==bRK)

    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    L2= np.zeros(dim)

    SS=np.zeros((dim,dim))

    un  = np.copy(y_0)

    uRK = np.zeros((sRK,dim))
    rhsRK = np.zeros((sRK,dim))

    for i in range(sRK):
        uRK[i,:] = y_0

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args)

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        t_sub = t_old + delta_t*cRK

        time+=delta_t
        SS=lhs_jac(t_old, un, delta_t, *func_args)
        for s in range(sRK):
            u_iter = un
            rhss = np.zeros(dim)
            for z in range(s):
                rhss = rhss + ARK[s,z]*rhsRK[z,:]

            

            u_new = un - np.linalg.solve(SS,-rhss)
            rhs_new = rhs_res(t_sub[s],u_new,*func_args)

            uRK[s,:] = u_new
            rhsRK[s,:] = rhs_new

        rhss = np.zeros(dim)
        for z in range(sRK):
            rhss = rhss + bRK[z]*rhsRK[z,:]

        un = un - np.linalg.solve(SS,-rhss)

        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll




@numba.njit
def BDF(lhs_res, rhs_res, lhs_jac, rhs_jac, \
    spect_radius, CFL, tspan, y_0, \
    order, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    if order == 2:
        stencil_ut = np.array([-4./3.,1./3.]) # 1 excluded
        coeff_du = 2./3.
        stencil_length=2
    elif order ==3:
        stencil_ut = np.array([-18./11.,9./11.,-2./11.]) #1 excluded
        doeff_du = 6./11.
        stencil_length=3
    else:
        raise ValueError("Order not implemented")

    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    max_newton_iter = 14
    newton_tolerance = 1e-7

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    L2= np.zeros(dim)

    SS=np.zeros((dim,dim))

    un  = np.copy(y_0)
    u_iter = np.copy(y_0)
    uns = np.zeros((stencil_length-1,dim))
    for i in range(stencil_length-1):
        uns[i,:]=un


    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        time+=delta_t

        rhs_n = rhs_res(t_old,un,*func_args)
        L2  = -rhs_n
        res_norm = np.linalg.norm(L2)/np.sqrt(dim)

        if (it==1 and order==2) or (it<=2 and order==3):
            #Crank Nicolson
            u_iter = np.copy(un)
            newton_iter=0
            while(res_norm>newton_tolerance and newton_iter<max_newton_iter):
                newton_iter+=1
                SS=lhs_jac(t_old, u_iter, delta_t, *func_args) - \
                    rhs_jac(t_old,u_iter, *func_args)

                u_iter = u_iter - np.linalg.solve(SS,L2)
                lhs = lhs_res(t_old,uns,u_iter, delta_t,np.array([-1]),*func_args)
                rhs_it = rhs_res(t_old,u_iter,*func_args)
                L2  = lhs - (rhs_n+rhs_it)/2.
                res_norm = np.linalg.norm(L2)/np.sqrt(dim)

            if newton_iter==max_newton_iter:
                with numba.objmode():
                    print("Max iteration %d of newton reached with residual %g"%(newton_iter,res_norm), flush=False)
                    print("")
                
                if res_norm>1000:
                    break
        else:
            u_iter = np.copy(un)
            newton_iter=0
            while(res_norm>newton_tolerance and newton_iter<max_newton_iter):
                newton_iter+=1
                SS=lhs_jac(t_old, u_iter, coeff_du*delta_t, *func_args) - \
                    rhs_jac(t_old,u_iter, *func_args)

                u_iter = u_iter - np.linalg.solve(SS,L2)
                lhs = lhs_res(t_old,uns,u_iter, delta_t*coeff_du,stencil_ut,*func_args)
                rhs_it = rhs_res(t_old,u_iter,*func_args)
                L2  = lhs - rhs_it
                res_norm = np.linalg.norm(L2)/np.sqrt(dim)

            if newton_iter==max_newton_iter:
                with numba.objmode():
                    print("Max iteration %d of newton reached with residual %g"%(newton_iter,res_norm), flush=False)
                    print("")
                
                if res_norm>1000:
                    break

        un=np.copy(u_iter)

        uns[1:,:] = uns[:-1,:] 
        uns[0,:]  = un
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll



def scipy_decImplicit_variable_dt(lhs_res, rhs_res, lhs_jac, rhs_jac,\
        spect_radius, CFL, tspan, y_0,\
        M_sub, K_corr, distribution, scipy_operators, Nsave=100,\
        *func_args):
    """
    DeC implicit or IMEX for stiff problems. Time integration routine. Evolves u'=f(t,u), with the Jacobian of the stiff part of f being J
    Input: func(t,u,args) evolution function, jac_stiff(t,u,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    M_sub number of subtimestep intervals, K_corr number of iterations,
    distribution of the subtimesteps among "equispaced", "gaussLobatto", "gaussLegendre" 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    dim=len(y_0)
    M_sub1=M_sub+1

    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))

    Theta, beta, recon = compute_theta_DeC(M_sub+1,distribution)
    
    un  = np.copy(y_0)
    un1 = np.copy(y_0)

    un  = np.copy(y_0)
    un1 = np.copy(y_0)

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_sub = beta*delta_t+time

        time+=delta_t

        for m in range(M_sub+1):
            u_a[:,m]=un
            u_p[:,m]=un
        jac_ll = lhs_jac(t_sub[0], u_p[:,0], delta_t,scipy_operators, *func_args)
        jac_rr = rhs_jac(t_sub[0], u_p[:,0], scipy_operators,*func_args)
        Jac = [np.empty(0)]
        for m in range(1,M_sub+1):
            #invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*beta[m]*SS)
            Jac.append(jac_ll - beta[m]*jac_rr)
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                rhs[:,r]=rhs_res(t_sub[r],u_p[:,r],*func_args)
            for m in range(1,M_sub+1):
                rhs_tot = np.zeros(dim) 
                for r in range(M_sub+1):
                    rhs_tot =rhs_tot +Theta[r,m]*rhs[:,r]
                # u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                # (-(u_p[:,m]-u_p[:,0])/delta_t+rhs_tot))
                lhs = lhs_res(t_sub[r],u_p[:,0],u_p[:,m], delta_t,*func_args)
                L2 = lhs-rhs_tot 
                u_a[:,m]= u_p[:,m]-spsolve(Jac[m],L2)
        un[:]=0.
        for m in range(M_sub+1):
            un+=recon[m]*u_a[:,m]
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll


def scipy_CrankNicolson_Rosenbrock(lhs_res, rhs_res, lhs_jac, rhs_jac,\
    spect_radius, CFL, tspan, y_0, scipy_operators, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    rhs= np.zeros(dim)

    invJac=np.zeros((dim,dim))
    Jac=np.zeros((dim,dim))

    un  = np.copy(y_0)
    uns = np.copy(y_0)
    un1 = np.copy(y_0)


    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        time+=delta_t

        SS=  lhs_jac(t_old, un, delta_t,scipy_operators, *func_args) - \
        rhs_jac(t_old,un,scipy_operators, *func_args)
        
        rhs_n = rhs_res(t_old,un,*func_args)
        L2  = -rhs_n
        uns = un - spsolve(SS,L2)

        SS=lhs_jac(t_old, uns, delta_t, scipy_operators,*func_args) - \
        rhs_jac(t_old,uns, scipy_operators,*func_args)
        lhs = lhs_res(t_old,un,uns, delta_t,*func_args)
        rhs_s = rhs_res(t_old,uns,*func_args)
        L2  = lhs - (rhs_n+rhs_s)/2.
        un1 = uns - spsolve(SS,L2)


        un=np.copy(un1)
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll



def scipy_CrankNicolson(lhs_res, rhs_res, lhs_jac, rhs_jac, \
    spect_radius, CFL, tspan, y_0, scipy_operators, Nsave=100,*func_args):
    """
    CrankNicolson_Rosenbrock for stiff problems. Time integration routine. 
    Finds the solution to f(t,dt,u^n,u^n+1)=0 at each step, with the Jacobian of the stiff part of f being J
    Input: func(t,dt,u^n,u^n+1,args) evolution function, jac_stiff(dt,u^n,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0

    max_newton_iter = 15
    newton_tolerance = 1e-7

    dim=len(y_0)
    
    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    L2= np.zeros(dim)

    SS=np.zeros((dim,dim))

    un  = np.copy(y_0)
    u_iter = np.copy(y_0)


    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)
        t_old = time

        time+=delta_t

        rhs_n = rhs_res(t_old,un,*func_args)
        L2  = -rhs_n
        res_norm = np.linalg.norm(L2)/np.sqrt(dim)

        u_iter = np.copy(un)
        newton_iter=0
        while(res_norm>newton_tolerance and newton_iter<max_newton_iter):
            newton_iter+=1
            SS=lhs_jac(t_old, u_iter, delta_t, scipy_operators,*func_args) - \
                rhs_jac(t_old,u_iter,scipy_operators, *func_args)

            u_iter = u_iter - spsolve(SS,L2)
            lhs = lhs_res(t_old,un,u_iter, delta_t,*func_args)
            rhs_it = rhs_res(t_old,u_iter,*func_args)
            L2  = lhs - (rhs_n+rhs_it)/2.
            res_norm = np.linalg.norm(L2)/np.sqrt(dim)

        if newton_iter==max_newton_iter:
            with numba.objmode():
                print("Max iteration %d of newton reached with residual %g"%(newton_iter,res_norm), flush=False)
                print("")
            
            if res_norm>1000:
                break


        un=np.copy(u_iter)
        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll


@numba.njit
def implicit_Euler_variable_dt(lhs_res, rhs_res, lhs_jac, rhs_jac, spect_radius, CFL, tspan, y_0, Nsave=100,*func_args):
    """
    DeC implicit or IMEX for stiff problems. Time integration routine. Evolves u'=f(t,u), with the Jacobian of the stiff part of f being J
    Input: func(t,u,args) evolution function, jac_stiff(t,u,args) jacobian of the stiff part, 
    spect_radius is the function that gives a constraint on dt, CFL is the coefficient in front of the previous funciton, 
    t_span time domain interval, y_0 initial condition, 
    M_sub number of subtimestep intervals, K_corr number of iterations,
    distribution of the subtimesteps among "equispaced", "gaussLobatto", "gaussLegendre" 
    """
    #NtMax=numba.int64(100000)
    time=tspan[0]
    T_fin = tspan[1]
    dtSave=T_fin/(Nsave-1.00001)
    nextSaveTime = 0.
    itSave = 0
    max_newt_iter = 100
    newt_tol = 1e-14
    dim=len(y_0)

    uAll=np.zeros((Nsave, dim))
    times=np.zeros(Nsave)

    u_p=np.zeros(dim)
    u_a=np.zeros(dim)
    rhs= np.zeros(dim)

    invJac=np.zeros((dim,dim))
    Jac=np.zeros((dim,dim))

    un  = np.copy(y_0)

    it=0
    uAll[itSave,:]=un
    nextSaveTime += dtSave 
    times[itSave]=time
    while (time<T_fin and it<1000000):
        it=it+1
        maxRho=spect_radius(time,un,*func_args);

        delta_t=min(CFL* maxRho,T_fin-time)


        k=0
        res = newt_tol+100
        while (k<max_newt_iter and res > newt_tol):
            k=k+1
            u_p=np.copy(u_a)
            SS=lhs_jac(time, u_p, delta_t, *func_args) - \
                rhs_jac(time,u_p, *func_args)
            Jac=SS

            lhs = lhs_res(time,un,u_p, delta_t,*func_args)
            rhs = rhs_res(time,u_p,*func_args)
            L2 =lhs - rhs
            res = np.linalg.norm(L2)
            u_a= u_p-delta_t*np.linalg.solve(Jac[:,:],L2)

        un=np.copy(u_a)

        time+=delta_t

        if time > nextSaveTime:
            itSave += 1
            uAll[itSave,:]=un
            times[itSave]=time
            nextSaveTime += dtSave 
            with numba.objmode():
                print("Time %5.3f out of %5.3f"%(time,T_fin), end="\r", flush=True)
 
    itSave += 1
    uAll[itSave,:]=un
    times[itSave]=time
    with numba.objmode():
        print("")

    return times, uAll







def decMPatankar(prod_dest, rhs, tspan, y_0, M_sub, K_corr, distribution):
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    prod_p = np.zeros((dim,dim,M_sub+1))
    dest_p = np.zeros((dim,dim,M_sub+1))
    rhs_p= np.zeros((dim,M_sub+1))
    Theta, beta,recon = compute_theta_DeC(M_sub+1,distribution)
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
            u_p[:,m]=U[:,it-1]
        for k in range(1,K_corr+1):
            u_p=np.copy(u_a)
            for r in range(M_sub+1):
                prod_p[:,:,r], dest_p[:,:,r]=prod_dest(u_p[:,r])
                rhs_p[:,r]=rhs(u_p[:,r])
            for m in range(1,M_sub+1):
                u_a[:,m]= patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,M_sub,Theta,u_p,dim)
        U[:,it]=0.
        for m in range(M_sub+1):
            U[:,it]+=recon[m]*u_a[:,m]
    return tspan, U


def patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,M_sub,Theta,u_p,dim):
    mass= np.eye(dim)
    RHS= u_p[:,0]
    for i in range(dim):
        for r in range(M_sub+1):
            RHS[i]=RHS[i]+delta_t*Theta[r,m]*rhs_p[i,r]
            if Theta[r,m]>0:
                for j in range(dim):
                    mass[i,j]=mass[i,j]-delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[j,m])
                    mass[i,i]=mass[i,i]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[i,m])
            elif Theta[r,m]<0:
                for j in range(dim):
                    mass[i,i]=mass[i,i]- delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[i,m])
                    mass[i,j]=mass[i,j]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[j,m])
    return np.linalg.solve(mass,RHS)
