import numpy as np
from numpy import sin, cos, pi, sqrt
import os

from .problem import ConservationLaw


class Maxwell_GLM_2D(ConservationLaw):
    def __init__(self, name, pert_coeff = None):
        self.name  = name
        self.dim   = 2
        self.n_eq  = 8
        self.vars  = ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "phi", "psi")
        self.equations = "Maxwell_GLM"

        self.c = 1. # Maxwell speed
        self.ch = 1. # Acoustics speed
        self.define_parameters()
        self.define_geom()
        
        self.pert_coeff = pert_coeff
        self.IC()
        self.dirichlet_BC()

        try:
            self.generate_exact()
        except:
            pass


        self.folderName=self.equations+str(self.dim)+"D_"+self.name
        os.makedirs(self.folderName, exist_ok=True)
        

    def max_dt(self, q_all, dx):
        """ to be improved! this is slow!!"""
        dt_max = np.min(dx) / abs(self.c) 
        return dt_max

    def define_parameters(self):
        if self.name in ["test1"]:
            self.T_fin = 1.
        elif self.name in ["test1_long"]:
            self.T_fin = 100.
        elif self.name in ["test1_num_perturbation", \
                            "test1_opt_perturbation", \
                            "test1_int_perturbation"]:
            self.T_fin = 1.
        elif self.name in ["vortex","smooth_vortex","smaller_smooth_vortex","source_vortex","source_vortex_dirichlet"]:
            self.T_fin = 1.
        elif self.name in ["vortex_long","smooth_vortex_long","smaller_smooth_vortex_long",\
                           "source_vortex_long","source_vortex_long_dirichlet"]:
            self.T_fin = 100.
        elif self.name in ["vortex_num_perturbation", \
                           "vortex_an_perturbation",\
                           "vortex_opt_perturbation",\
                           "vortex_int_perturbation"] :
            self.T_fin = 0.35
            

    def define_geom(self):
        if "vortex" in self.name or self.name == "RP4" and "smaller" not in self.name:# and "coriolis" not in self.name:
            self.geometry_name = "square"
            self.xL = np.array([0.,0.], dtype=np.float64)
            self.xR = np.array([1.,1.], dtype=np.float64)
            self.BC = np.array([1,1,1,1], dtype=np.int32) # non periodic
            if "source_vortex" in self.name:
                self.BC = np.array([1,1,1,2], dtype=np.int32) # not all dirichlet
        elif "test1" in self.name:
            self.geometry_name = "periodic_large_square"
            self.xL = np.array([-3.,-3.], dtype=np.float64)
            self.xR = np.array([3.,3.], dtype=np.float64)
            self.BC = np.array([0,0,0,0], dtype=np.int32) # periodic

        
        self.geometry_folder = "Geometry_"+self.geometry_name+"/"
        os.system("mkdir %s"%self.geometry_folder)

    def dirichlet_BC(self):

        if self.name in ["source_vortex_dirichlet","source_vortex_long_dirichlet", "source_vortex_opt_perturbation" \
                         "SG","SG1","SG_long","SG_num_perturbation","SG_opt_perturbation","smooth_vortex_opt_perturbation","coriolis_vortex_long","moving_source"]:
                        #,\
                        # "constant_flow_an_perturbation", "constant_flow"]:
            # At all boundaries I'm imposing dirichlet BC for all vars
            self.dirichlet = dict()
            self.dirichlet["all"] = self.vars
        else:
            self.dirichlet = None
            
    def IC(self):
        """
        Returns a dictionary of lambda functions of the ICs
        """
        if self.name in ["test1","test1_long"]:
            zero_func = lambda x,y: np.zeros_like(x)
            Ex0     = lambda x,y: x*np.exp(-x**2-y**2)
            Ey0     = lambda x,y: y*np.exp(-x**2-y**2)
            self.ics   = dict()
            for var in self.vars:
                self.ics[var] = zero_func
            self.ics["Ex"] = Ex0
            self.ics["Ey"] = Ey0
            return self.ics

        elif self.name in ["test1_num_perturbation", \
                           "test1_opt_perturbation", \
                           "test1_int_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "test1_long"
            self.steady_state_test = "test1_long"

            x0=0.4;y0=0.43

            zero_func = lambda x,y: np.zeros_like(x)
            delta_B = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)  # 0.

            self.perturbation      = dict()
            for var in self.vars:
                self.perturbation[var] = zero_func
            self.perturbation["Bz"] = delta_B
            return self.perturbation


    def generate_exact(self):
        if self.name in ["test1", "test1_long"]:

            zero_func = lambda x,y: np.zeros_like(x)       
            Ex0     = lambda x,y: x*np.exp(-x**2-y**2)
            Ey0     = lambda x,y: y*np.exp(-x**2-y**2)
            self.exact   = dict()
            for var in self.vars:
                self.exact[var] = zero_func 
            self.exact["Ex"] = Ex0
            self.exact["Ey"] = Ey0
            return self.exact
        
        
def pressure_perturbation_function(x, y, x0, y0):
    r0 = 0.1; 
    r = sqrt((x-x0)**2+(y-y0)**2)
    rho = np.minimum((r/r0)**2,1.-1e-16)
    u = (r < r0)*np.exp(-1/(2*(1-rho)**2))/np.exp(-1/2)
    return u