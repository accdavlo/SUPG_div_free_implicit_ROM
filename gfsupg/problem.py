import numpy as np
from numpy import sin, cos, pi, sqrt
import os

class ConservationLaw:
    """To be defined"""
    def __init__(self, name):
        self.name = name

class LinearAcoustic2D(ConservationLaw):
    def __init__(self, name, pert_coeff = None):
        self.name  = name
        self.dim   = 2
        self.n_eq  = 3
        self.vars  = ("u", "v", "p")
        self.c     = 1.
        self.equations = "acoustics"
        self.coriolis             = 0.
        self.coriolis_non_uniform = None
        self.source               = None
        self.p_source             = None
        self.friction             = 0.

        self.define_parameters()
        self.define_geom()
        
        self.pert_coeff = pert_coeff
        self.IC()
        self.dirichlet_BC()

        try:
            self.generate_exact()
        except:
            pass


        self.folderName="LinAc2D_"+self.name
        os.system('mkdir "'+self.folderName+'"')      

    def max_dt(self, q_all, dx):
        """ to be improved! this is slow!!"""
        dt_max = np.min(dx) / np.abs(self.c) 
        return dt_max

    def define_parameters(self):
        if self.name in ["vortex","smooth_vortex","smaller_smooth_vortex","source_vortex","source_vortex_dirichlet"]:
            self.T_fin = 1.
        elif self.name in ["vortex_long","smooth_vortex_long","smaller_smooth_vortex_long",\
                           "source_vortex_long","source_vortex_long_dirichlet"]:
            self.T_fin = 100.
        elif self.name in ["vortex_num_perturbation", \
                           "vortex_an_perturbation",\
                           "vortex_opt_perturbation",\
                           "vortex_int_perturbation",\
                           "smooth_vortex_num_perturbation", \
                           "smooth_vortex_an_perturbation", \
                           "smooth_vortex_opt_perturbation", \
                           "smooth_vortex_int_perturbation",\
                           "smaller_smooth_vortex_num_perturbation", \
                           "smaller_smooth_vortex_an_perturbation", \
                           "smaller_smooth_vortex_opt_perturbation", \
                           "smaller_smooth_vortex_int_perturbation",\
                           "source_vortex_num_perturbation", \
                           "source_vortex_an_perturbation", \
                           "source_vortex_opt_perturbation", \
                           "source_vortex_int_perturbation"] :
            self.T_fin = 0.35
            
        elif "moving_source" in self.name:
            self.T_fin = 0.1
            self.background_speed = np.array([-0.1,0.1])
        elif "oblique" in self.name:
            self.T_fin = 1.
        elif self.name == "coriolis_vortex":
            self.T_fin = 1.
            self.coriolis = 0.2
        elif self.name == "coriolis_vortex_long":
            self.T_fin = 100.
            self.coriolis = 0.2
        elif self.name in ["coriolis_vortex_opt_perturbation",\
                           "coriolis_vortex_int_perturbation",\
                           "coriolis_vortex_an_perturbation",\
                           "coriolis_vortex_num_perturbation"]:
            self.T_fin = 0.35
            self.coriolis = 0.2
        elif self.name == "RP4":
            self.T_fin = 0.4
            self.coriolis = 0.
        elif self.name in ["SG","SG_long", "SG1", "SG_num_perturbation","SG_opt_perturbation"]:
            if self.name in ["SG","SG_long", "SG_num_perturbation","SG_opt_perturbation"]:
                self.b_SG     = 1.0  # end of y domain
                self.lambda_SG= 1.0  # end of x domain
                self.T_fin    = 1.0
                self.coriolis = 0.
                self.cor_f_0      = 0.01
                self.cor_phi_0    = 0.01
                self.friction = 0.01
                self.wind_F   = 0.1
                self.D_SG     = 1.0
                if self.name=="SG_long":
                    self.T_fin = 100.0
                elif self.name in ["SG_num_perturbation","SG_opt_perturbation"]:
                    self.T_fin = 0.35 
            elif self.name == "SG1":
                self.b_SG     = 10.0e9  # end of y domain
                self.lambda_SG= 2.*np.pi*10**8  # end of x domain
                self.D_SG     = 20000.
                self.T_fin    = 4.e4
                self.coriolis = 0.
                self.cor_f_0  = 1e-13
                self.cor_phi_0= 0.0
                self.friction = 0.02
                self.wind_F   = 1.0
                raise ValueError("SG1 not fully implemented! ")


            self.coriolis_non_uniform = lambda x,y, t=0: self.cor_f_0*y+self.cor_phi_0 
            self.source     = dict()
            self.source["u"]= lambda x,y, t=0: - self.wind_F * np.cos(np.pi*y/self.b_SG)
            self.source["v"]= lambda x,y, t=0: 0.
            self.source["p"]= lambda x,y, t=0: 0.          
            

            self.alpha_SG = self.D_SG*self.cor_f_0/self.friction
            self.gamma_SG = self.wind_F*np.pi/self.friction/self.b_SG

            discr = np.sqrt(self.alpha_SG**2/4.+(np.pi/self.b_SG)**2)

            self.A_SG     = -self.alpha_SG/2. + discr
            self.B_SG     = -self.alpha_SG/2. - discr
            self.k_SG     = (1.-np.exp(self.B_SG*self.lambda_SG))/\
                  (np.exp(self.A_SG*self.lambda_SG)-np.exp(self.B_SG*self.lambda_SG))
            self.q_SG     = 1.-self.k_SG

        elif self.name in ["constant_flow_an_perturbation", "constant_flow"]:
            self.T_fin = 0.35
        if "source" in self.name:
            self.x0=0.5
            self.y0=0.5
            self.p_source = True


            self.coefficient_source = 0.01
            self.scale_source = 100.
            self.x0_source = 0.65
            self.y0_source = 0.39

            g      = lambda x,y, t=0: gaussian(x-self.x0_source, y-self.y0_source, a=self.scale_source)

            self.source = dict()
            self.source["u"] = lambda x,y, t=0: 0.
            self.source["v"] = lambda x,y, t=0: 0.
            self.source["p"] = lambda x,y, t=0: (((x-self.x0_source)**2+(y-self.y0_source)**2)*\
                self.scale_source -1)*4*self.scale_source*self.coefficient_source*g(x,y) 

        if self.name == "moving_source":
            self.coefficient_source = 1e-3
            self.x_moving = lambda x,t : x-self.x0_source -self.background_speed[0]*t
            self.y_moving = lambda y,t : y-self.y0_source -self.background_speed[1]*t 
            self.gaussian_func  = lambda x,y, t: gaussian(self.x_moving(x,t), self.y_moving(y,t), a=self.scale_source)
            self.grad_g = lambda x,y,t: grad_gaussian(self.x_moving(x,t), self.y_moving(y,t), a=self.scale_source)
            self.hess_g = lambda x,y,t: hess_gaussian(self.x_moving(x,t), self.y_moving(y,t), a=self.scale_source)
            self.laplace_g = lambda x,y,t: laplace_gaussian(self.x_moving(x,t), self.y_moving(y,t), a=self.scale_source)
            
            self.source["p"] = lambda x,y,t : \
                self.coefficient_source*(-self.background_speed.T@self.hess_g(x,y,t) @self.background_speed \
                                     + self.laplace_g(x,y,t) )


    def define_geom(self):
        if "vortex" in self.name or self.name == "RP4" and "smaller" not in self.name:# and "coriolis" not in self.name:
            self.geometry_name = "square"
            self.xL = np.array([0.,0.], dtype=np.float64)
            self.xR = np.array([1.,1.], dtype=np.float64)
            self.BC = np.array([1,1,1,1], dtype=np.int32) # non periodic
            if "source_vortex" in self.name:
                self.BC = np.array([1,1,1,2], dtype=np.int32) # not all dirichlet
        elif self.name=="moving_source":
            self.geometry_name = "square"
            self.xL = np.array([0.,0.], dtype=np.float64)
            self.xR = np.array([1.,1.], dtype=np.float64)
            self.BC = np.array([1,1,1,1], dtype=np.int32) # non periodic


        elif "oblique" in self.name \
            or "constant_flow" in self.name\
            or "smaller" in self.name:# or "coriolis_vortex" in self.name:
            self.geometry_name = "periodic_square"
            self.xL = np.array([0.,0.], dtype=np.float64)
            self.xR = np.array([1.,1.], dtype=np.float64)
            self.BC = np.array([0,0,0,0], dtype=np.int32) # periodic
        elif self.name in ["SG","SG_long","SG1","SG_num_perturbation","SG_opt_perturbation"]:
            if self.b_SG == 1.0 and self.lambda_SG == 1.0:
                self.geometry_name = "square"
            else:
                self.geometry_name = "square_SG"
            self.xL = np.array([0.,0.], dtype=np.float64)
            self.xR = np.array([self.lambda_SG, self.b_SG], dtype=np.float64)
            self.BC = np.array([1,1,1,1], dtype=np.int32) # non periodic
        
        
        self.geometry_folder = "Geometry_"+self.geometry_name+"/"
        os.system("mkdir %s"%self.geometry_folder)

    def dirichlet_BC(self):

        if self.name in ["source_vortex_dirichlet","source_vortex_long_dirichlet", "source_vortex_opt_perturbation", \
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
        if self.name in ["vortex", "vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: analytic_travelling_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: analytic_travelling_vortex_function(x, y, x0, y0)*( x-x0)
            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["coriolis_vortex","coriolis_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.0 - self.coriolis*\
                                 gaussian_vortex_pressure_function(x, y, x0, y0)
            u0     = lambda x,y: gaussian_vortex_velocity_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: gaussian_vortex_velocity_function(x, y, x0, y0)*( x-x0)
            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["smooth_vortex", "smooth_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: smooth_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: smooth_vortex_function(x, y, x0, y0)*( x-x0)
            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["smaller_smooth_vortex", "smaller_smooth_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*(-y+y0)
            v0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*( x-x0)
            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["source_vortex","source_vortex_dirichlet",\
                           "source_vortex_long","source_vortex_long_dirichlet"]:
            
            g      = lambda x,y: gaussian(x-self.x0_source, y-self.y0_source, a=self.scale_source)

            p0     = lambda x,y: 1.
           
            u0     = lambda x,y: smooth_vortex_function(x, y, self.x0, self.y0)*(-y+self.y0)\
                - 2.*self.scale_source*self.coefficient_source*g(x,y)*(x-self.x0_source)
            
            v0     = lambda x,y: smooth_vortex_function(x, y, self.x0, self.y0)*( x-self.x0)\
                - 2.*self.scale_source*self.coefficient_source*g(x,y)*(y-self.y0_source)


            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["moving_source"]:
            
            p0     = lambda x,y : 1. + self.coefficient_source* self.background_speed.T@ self.grad_g(x,y,0.)
           
            u0     = lambda x,y: self.coefficient_source*self.grad_g(x,y,0.)[0]
            #smooth_vortex_function(x, y, self.x0, self.y0)*(-y+self.y0)\
            v0     = lambda x,y: self.coefficient_source*self.grad_g(x,y,0.)[1] 
            #lambda x,y: smooth_vortex_function(x, y, self.x0, self.y0)*( x-self.x0)\
             

            self.ics   = dict()
            self.ics["u"] = u0
            self.ics["v"] = v0
            self.ics["p"] = p0
            return self.ics
        elif self.name in ["vortex_num_perturbation", \
                           "vortex_opt_perturbation", \
                           "vortex_int_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "vortex_long"
            self.steady_state_test = "vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)  # 0.
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation
        elif self.name in ["vortex_an_perturbation"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: analytic_travelling_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: analytic_travelling_vortex_function(x, y, x0, y0)*( x-x0)

            self.with_perturbation = True
            self.base_test = "vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: 0.
            delta_u = lambda x,y: vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: u0(x,y)+delta_u(x,y)
            self.ics["v"] = lambda x,y: v0(x,y)+delta_v(x,y)
            self.ics["p"] = lambda x,y: p0(x,y)+delta_p(x,y)
            return self.ics
            

        elif self.name in ["smooth_vortex_num_perturbation",\
                           "smooth_vortex_opt_perturbation",\
                           "smooth_vortex_int_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "smooth_vortex_long"
            self.steady_state_test = "smooth_vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)#0.
            delta_u = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation
        elif self.name in ["smooth_vortex_an_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: smooth_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: smooth_vortex_function(x, y, x0, y0)*( x-x0)

            self.with_perturbation = True
            self.base_test = "smooth_vortex_long"

            x0_pert=0.4;y0_pert=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0_pert, y0_pert)
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: u0(x,y)+delta_u(x,y)
            self.ics["v"] = lambda x,y: v0(x,y)+delta_v(x,y)
            self.ics["p"] = lambda x,y: p0(x,y)+delta_p(x,y)
            return self.ics
        elif self.name in ["smaller_smooth_vortex_num_perturbation",\
                           "smaller_smooth_vortex_opt_perturbation",\
                           "smaller_smooth_vortex_int_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "smaller_smooth_vortex_long"
            self.steady_state_test = "smaller_smooth_vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: 0.#self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)#0.
            delta_u = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)
            delta_v = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation
        elif self.name in ["smaller_smooth_vortex_an_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.
            u0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*(-y+y0)
            v0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*( x-x0)
            
            self.with_perturbation = True
            self.base_test = "smaller_smooth_vortex_long"

            x0_pert=0.4;y0_pert=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0_pert, y0_pert)
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: u0(x,y)+delta_u(x,y)
            self.ics["v"] = lambda x,y: v0(x,y)+delta_v(x,y)
            self.ics["p"] = lambda x,y: p0(x,y)+delta_p(x,y)
            return self.ics
        elif self.name in ["coriolis_vortex_opt_perturbation", \
                           "coriolis_vortex_int_perturbation"\
                           "coriolis_vortex_num_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "coriolis_vortex_long"
            self.steady_state_test = "coriolis_vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)  # 0.
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation      
        elif self.name in ["coriolis_vortex_an_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            x0=0.5;y0=0.5

            p0     = lambda x,y: 1.0 - self.coriolis*\
                                 gaussian_vortex_pressure_function(x, y, x0, y0)
            u0     = lambda x,y: gaussian_vortex_velocity_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y: gaussian_vortex_velocity_function(x, y, x0, y0)*( x-x0)

            self.with_perturbation = True
            self.base_test = "coriolis_vortex_long"

            x0_pert=0.4;y0_pert=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0_pert, y0_pert)
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: u0(x,y)+delta_u(x,y)
            self.ics["v"] = lambda x,y: v0(x,y)+delta_v(x,y)
            self.ics["p"] = lambda x,y: p0(x,y)+delta_p(x,y)
            return self.ics

        elif self.name in ["source_vortex_num_perturbation",\
                           "source_vortex_opt_perturbation",\
                           "source_vortex_int_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3

            self.with_perturbation = True
            self.base_test         = "source_vortex_long"
            self.steady_state_test = "source_vortex_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)#0.
            delta_u = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation
        elif self.name in ["source_vortex_an_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3

            g      = lambda x,y: gaussian(x-self.x0_source, y-self.y0_source, a=self.scale_source)

            p0     = lambda x,y: 1.
           
            u0     = lambda x,y: smooth_vortex_function(x, y, self.x0, self.y0)*(-y+self.y0)\
                - 2.*self.scale_source*self.coefficient_source*g(x,y)*(x-self.x0_source)
            
            v0     = lambda x,y: smooth_vortex_function(x, y, self.x0, self.y0)*( x-self.x0)\
                - 2.*self.scale_source*self.coefficient_source*g(x,y)*(y-self.y0_source)



            self.with_perturbation = True
            self.base_test = "source_vortex_long"

            x0_pert=0.4;y0_pert=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0_pert, y0_pert)
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: u0(x,y)+delta_u(x,y)
            self.ics["v"] = lambda x,y: v0(x,y)+delta_v(x,y)
            self.ics["p"] = lambda x,y: p0(x,y)+delta_p(x,y)
            return self.ics



        elif "oblique" in self.name:
            self.theta = pi/4.
            self.lamb  = 1./4.
            xi = lambda x,y : x*cos(self.theta)+y*sin(self.theta)
            coef = 2.*pi/self.lamb /cos(self.theta)
            self.ics   = dict()
            self.ics["u"] = lambda x,y: 0.
            self.ics["v"] = lambda x,y: 0.
            self.ics["p"] = lambda x,y: cos(coef*xi(x,y) )
            return self.ics
        elif self.name == "RP4":
            self.ics   = dict()
            self.ics["u"] = lambda x,y: 1.0*(x>0.5)*(y>0.5)
            self.ics["v"] = lambda x,y: 0.0
            self.ics["p"] = lambda x,y: 0.0
            return self.ics
        elif self.name in ["SG","SG_long", "SG1"]:
            self.ics   = dict()
            self.ics["u"] = lambda x,y: SG_u(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG)
            self.ics["v"] = lambda x,y: SG_v(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG)
            self.ics["p"] = lambda x,y: SG_p(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG, self.wind_F, self.cor_phi_0, self.cor_f_0)
            return self.ics
        elif self.name in ["SG_num_perturbation","SG_opt_perturbation"]:
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "SG_long"
            self.steady_state_test = "SG_long"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)#0.
            delta_u = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0.# vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p
            return self.perturbation
        elif self.name == "constant_flow":
            self.ics   = dict()
            self.ics["u"] = lambda x,y: 0.
            self.ics["v"] = lambda x,y: 0.
            self.ics["p"] = lambda x,y: 1.0
            return self.ics

        elif self.name == "constant_flow_an_perturbation":
            if self.pert_coeff is None:
                self.pert_coeff = 1e-3
            self.with_perturbation = True
            self.base_test         = "constant_flow"

            x0=0.4;y0=0.43

            delta_p = lambda x,y: self.pert_coeff*pressure_perturbation_function(x, y, x0, y0)#(abs(x-x0)<0.05)*(abs(y-y0)<0.05)  # 0.
            delta_u = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*(-y+y0)
            delta_v = lambda x,y: 0. #vortex_perturbation_function(x, y, x0, y0)*( x-x0)
            self.perturbation      = dict()
            self.perturbation["u"] = delta_u
            self.perturbation["v"] = delta_v
            self.perturbation["p"] = delta_p

            self.ics   = dict()
            self.ics["u"] = lambda x,y: 0.0
            self.ics["v"] = lambda x,y: 0.0
            self.ics["p"] = lambda x,y: 1.0+delta_p(x,y)
            return self.ics


    def generate_exact(self):
        if self.name in ["vortex", "vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y,t: 1.
            u0     = lambda x,y,t: analytic_travelling_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y,t: analytic_travelling_vortex_function(x, y, x0, y0)*( x-x0)
            self.exact   = dict()
            self.exact["u"] = u0
            self.exact["v"] = v0
            self.exact["p"] = p0
            return self.exact
        elif self.name in ["coriolis_vortex","coriolis_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y,t: 1.0 - self.coriolis*\
                                 gaussian_vortex_pressure_function(x, y, x0, y0)
            u0     = lambda x,y,t: gaussian_vortex_velocity_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y,t: gaussian_vortex_velocity_function(x, y, x0, y0)*( x-x0)
            self.exact   = dict()
            self.exact["u"] = u0
            self.exact["v"] = v0
            self.exact["p"] = p0
            return self.exact
        elif self.name in ["smooth_vortex", "smooth_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y,t: 1.
            u0     = lambda x,y,t: smooth_vortex_function(x, y, x0, y0)*(-y+y0)
            v0     = lambda x,y,t: smooth_vortex_function(x, y, x0, y0)*( x-x0)
            self.exact   = dict()
            self.exact["u"] = u0
            self.exact["v"] = v0
            self.exact["p"] = p0
            return self.exact
        
        elif self.name in ["smaller_smooth_vortex", "smaller_smooth_vortex_long"]:
            x0=0.5;y0=0.5

            p0     = lambda x,y,t: 1.
            u0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*(-y+y0)
            v0     = lambda x,y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.)*2.*( x-x0)
            self.exact   = dict()
            self.exact["u"] = u0
            self.exact["v"] = v0
            self.exact["p"] = p0
            return self.exact
        
        elif self.name in ["source_vortex","source_vortex_dirichlet",\
                           "source_vortex_long","source_vortex_long_dirichlet"]:
            
            self.exact   = dict()
            self.exact["u"] = lambda x,y,t: self.ics["u"](x,y)
            self.exact["v"] = lambda x,y,t: self.ics["v"](x,y)
            self.exact["p"] = lambda x,y,t: self.ics["p"](x,y)
            return self.exact

        elif self.name in ["moving_source"]:
            self.exact   = dict()
            self.exact["u"] = lambda x,y,t: self.ics["u"](x-self.background_speed[0]*t,y-self.background_speed[1]*t)
            self.exact["v"] = lambda x,y,t: self.ics["v"](x-self.background_speed[0]*t,y-self.background_speed[1]*t)
            self.exact["p"] = lambda x,y,t: self.ics["p"](x-self.background_speed[0]*t,y-self.background_speed[1]*t)
            return self.exact


        elif "oblique" in self.name:
            self.theta = pi/4.
            self.lamb  = 1./4.
            xi = lambda x,y : x*cos(self.theta)+y*sin(self.theta)
            xip = lambda x,y,t : xi(x,y) + self.c *t
            xim = lambda x,y,t : xi(x,y) - self.c *t
            coef = 2.*pi/self.lamb /cos(self.theta)
            self.exact   = dict()
            self.exact["u"] = lambda x,y,t: -0.5/self.c*( cos(coef*xip(x,y,t) )-cos(coef*xim(x,y,t) )  )*cos(self.theta)
            self.exact["v"] = lambda x,y,t: -0.5/self.c*( cos(coef*xip(x,y,t) )-cos(coef*xim(x,y,t) )  )*sin(self.theta)
            self.exact["p"] = lambda x,y,t: 0.5*( cos(coef*xip(x,y,t) )+cos(coef*xim(x,y,t) )  )
            return self.exact
        elif self.name in ["SG","SG_long", "SG1"]:
            self.exact   = dict()
            self.exact["u"] = lambda x,y,t: SG_u(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG)
            self.exact["v"] = lambda x,y,t: SG_v(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG)
            self.exact["p"] = lambda x,y,t: SG_p(x,y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG, self.wind_F, self.cor_phi_0, self.cor_f_0)
            return self.exact
        else:
            self.exact = None
            return self.exact 



def lambda_vortex(r):
    lam = (20.*cos(r))/3. + (27.*cos(r)**2.)/16. + (4.*cos(r)**3)/9.+ cos(r)**4/16. + (20.*r*sin(r))/3. \
            + (35.*r**2)/16. + (27.*r*cos(r)*sin(r))/8. + (4.*r*cos(r)**2*sin(r))/3. + (r*cos(r)**3*sin(r))/4.
    return lam
def analytic_travelling_vortex_function(x, y, x0, y0):
    r0 = 0.45; deltah = 0.1; w = pi/r0; g = 9.81
    Gam=(12.*pi*np.sqrt(deltah*g))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    r = sqrt((x-x0)**2+(y-y0)**2)
    u = (r < r0)*Gam*(1. + cos(w*r))**2
    return u

def vortex_perturbation_function(x, y, x0, y0):
    r0 = 0.05; deltah = 0.001; w = pi/r0; g = 9.81
    Gam=(12.*pi*np.sqrt(deltah*g))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    r = sqrt((x-x0)**2+(y-y0)**2)
    u = (r < r0)*Gam*(1. + cos(w*r))**2
    return u



def smooth_vortex_function(x, y, x0, y0):
    r0 = 0.45
    deltah = 0.1 
    g = 9.81
    Gam = 0.2                               #(12.*pi*np.sqrt(deltah*g))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    rho = np.minimum((r/r0)**2,1.-1e-16)
    u = (r < r0)*2*Gam*np.exp(-1/(2*(1-rho)**2))*np.sqrt(g/(r0*(1-rho)**3))
    return u

def gaussian(x,y,a):
    # g = e^{-a(x^2+y^2)}
    r2 = x**2+y**2
    return np.exp(-a*r2)

def grad_gaussian(x,y,a):
    # grad_g = -2a (x,y)^T*e^{-a(x^2+y^2)}
    r2 = x**2+y**2
    grad  = -2*a*np.array([x,y]) *np.exp(-a*r2)
    return grad

def hess_gaussian(x,y,a):
    r2 = x**2+y**2
    exp_f = np.exp(-a*r2)
    hess = np.array([[-1+2*a*x**2 , 2*a*x*y ],[2*a*x*y, -1+2*a*y**2]]) *2*a*exp_f
    return hess 

def laplace_gaussian(x,y,a):
    r2 = x**2+y**2
    exp_f = np.exp(-a*r2)
    laplace = (-1+2*a*x**2  -1+2*a*y**2) *2*a*exp_f
    return laplace

def gaussian_vortex_pressure_function(x,y,x0,y0):
    r2 = (x-x0)**2+(y-y0)**2
    return 0.1*np.exp(-100.*r2)

def gaussian_vortex_velocity_function(x,y,x0,y0):
    r2 = (x-x0)**2+(y-y0)**2
    return 20.*np.exp(-100.*r2)

def pressure_perturbation_function(x, y, x0, y0):
    r0 = 0.1; 
    r = sqrt((x-x0)**2+(y-y0)**2)
    rho = np.minimum((r/r0)**2,1.-1e-16)
    u = (r < r0)*np.exp(-1/(2*(1-rho)**2))/np.exp(-1/2)
    return u


def exact_radial_RP4(r,t):
    v = 1/2/np.pi*L_func(r/t)
    return v



def L_func(s):
    if s>=1:
        return 0.
    elif s > 1e-6:
        return np.log( (1+np.sqrt(1-s**2))/s)
    else:
        return - np.log(s/2) - s**2/4 


def SG_wind(x,y, F, b):
    return np.array([-F*np.cos(np.pi*y/b),0.])


def SG_u(x, y, b, gamma, k, q, A, B):
    return gamma * (b / np.pi) * np.cos(np.pi * y / b) * (k * np.exp(A * x) + q * np.exp(B * x) - 1)

def SG_v(x, y, b, gamma, k, q, A, B):
    return -gamma * (b / np.pi)**2 * np.sin(np.pi * y / b) * (k * A * np.exp(A * x) + q * B * np.exp(B * x))

def SG_p(x, y, b, gamma, k, q, A, B, F, phi_0, f_0):
    phi = lambda y : f_0 * y + phi_0
    term1 = -F * (k / A * np.exp(A * x) + q / B * np.exp(B * x))
    term2 = -F * (b / np.pi)**2 * (k * A * np.exp(A * x) + q * B * np.exp(B * x)) * (np.cos(np.pi * y / b) - 1)
    term3 = phi(y) * gamma * (b / np.pi)**2 * np.sin(np.pi * y / b)
    term4 = gamma * f_0 * (b / np.pi)**3 * (np.cos(np.pi * y / b) - 1)
    
    return term1 + term2 - (term3 + term4) * (k * np.exp(A * x) + q * np.exp(B * x) - 1)


def SG_exact_sol(x,y,  b, gamma, k, q, A, B, F, phi_0, f_0):
    sol = dict()
    sol["u"] = SG_u(x,y,  b, gamma, k, q, A, B)
    sol["v"] = SG_v(x,y,  b, gamma, k, q, A, B)
    sol["p"] = SG_p(x,y,  b, gamma, k, q, A, B, F, phi_0, f_0)
    return sol


def SG_exact_fun():
    sol = dict()
    sol["u"] = lambda x,y,  b, gamma, k, q, A, B, F, phi_0, f_0: SG_u(x,y,  b, gamma, k, q, A, B)
    sol["v"] = lambda x,y,  b, gamma, k, q, A, B, F, phi_0, f_0: SG_v(x,y,  b, gamma, k, q, A, B)
    sol["p"] = lambda x,y,  b, gamma, k, q, A, B, F, phi_0, f_0: SG_p(x,y,  b, gamma, k, q, A, B, F, phi_0, f_0)
    return sol