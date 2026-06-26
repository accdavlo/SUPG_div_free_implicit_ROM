import numpy as np
from numpy import sin, cos, pi, sqrt
import os

# --- Base Classes ---

class ConservationLaw:
    """To be defined"""
    def __init__(self, name):
        self.name = name


class LinearAcoustic2D(ConservationLaw):
    """
    Base structural class for 2D Linear Acoustic equations.
    Handles shared boilerplate attributes and scheduling actions.
    """
    def __init__(self, name, pert_coeff=None, T_fin=1.0):
        super().__init__(name)
        self.dim = 2
        self.n_eq = 3
        self.vars = ("u", "v", "p")
        self.c = 1.0
        self.equations = "acoustics"
        self.coriolis               = 0.0
        self.coriolis_non_uniform   = None
        self.source                 = None
        self.p_source               = None
        self.friction               = 0.0
        
        self.pert_coeff = pert_coeff
        self.T_fin = T_fin

        # Configuration pipeline executed dynamically by child test states
        self.define_parameters()
        self.define_geom()
        self.IC()
        self.dirichlet_BC()

        try:
            self.generate_exact()
        except Exception:
            pass

        self.folderName = f"LinAc2D_{self.name}"
        os.makedirs(self.folderName, exist_ok=True)      

    def max_dt(self, q_all, dx):
        """Calculates maximum time-step constraint."""
        return np.min(dx) / abs(self.c)

    def define_parameters(self):
        pass

    def define_geom(self):
        pass

    def dirichlet_BC(self):
        self.dirichlet = None
            
    def IC(self):
        self.ics = {}
        return self.ics

    def generate_exact(self):
        self.exact = None
        return self.exact

    def _setup_geometry_folder(self):
        self.geometry_folder = f"Geometry_{self.geometry_name}/"
        os.makedirs(self.geometry_folder, exist_ok=True)


# --- Concrete Child Test Cases ---

class VortexTestCase(LinearAcoustic2D):
    def __init__(self, basis_name="vortex", pert_coeff=None, is_long=False, pert_type=None):
        self.is_long = is_long
        self.pert_type = pert_type  # Accepts: 'num', 'an', 'opt', 'int'
        self.basis_name = basis_name
        name = f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")

        # Determine simulation duration based on parameter logic
        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)

    def define_geom(self):
        self.geometry_name = "square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self._setup_geometry_folder()

    def IC(self):
        x0, y0 = 0.5, 0.5
        p0 = lambda x, y: 1.0
        u0 = lambda x, y: analytic_travelling_vortex_function(x, y, x0, y0) * (-y + y0)
        v0 = lambda x, y: analytic_travelling_vortex_function(x, y, x0, y0) * (x - x0)
        
        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = "vortex_long"

            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p) if self.pert_type != "an" else 0.0
            delta_u = lambda x, y: vortex_perturbation_function(x, y, x0_p, y0_p) * (-y + y0_p) if self.pert_type == "an" else 0.0
            delta_v = lambda x, y: vortex_perturbation_function(x, y, x0_p, y0_p) * (x - x0_p) if self.pert_type == "an" else 0.0
            
            self.perturbation = {"u": delta_u, "v": delta_v, "p": delta_p}
            
            if self.pert_type == "an":
                self.ics["u"] = lambda x, y: u0(x, y) + delta_u(x, y)
                self.ics["v"] = lambda x, y: v0(x, y) + delta_v(x, y)
                self.ics["p"] = lambda x, y: p0(x, y) + delta_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        x0, y0 = 0.5, 0.5
        self.exact = {
            "u": lambda x, y, t: analytic_travelling_vortex_function(x, y, x0, y0) * (-y + y0),
            "v": lambda x, y, t: analytic_travelling_vortex_function(x, y, x0, y0) * (x - x0),
            "p": lambda x, y, t: 1.0
        }
        return self.exact

class CoriolisVortexTestCase(LinearAcoustic2D):
    def __init__(self, basis_name="coriolis_vortex", pert_coeff=None, is_long=False, pert_type=None):
        self.is_long = is_long
        self.pert_type = pert_type
        self.basis_name = basis_name
        name = f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")

        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)
        self.coriolis = 0.2

    def define_geom(self):
        self.geometry_name = "square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        # Specific boundary overriding rules
        if self.is_long:
            self.dirichlet = {"all": self.vars}
        else:
            self.dirichlet = None

    def IC(self):
        x0, y0 = 0.5, 0.5
        p0 = lambda x, y: 1.0 - self.coriolis * gaussian_vortex_pressure_function(x, y, x0, y0)
        u0 = lambda x, y: gaussian_vortex_velocity_function(x, y, x0, y0) * (-y + y0)
        v0 = lambda x, y: gaussian_vortex_velocity_function(x, y, x0, y0) * (x - x0)
        
        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = "coriolis_vortex_long"
            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
            self.perturbation = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": delta_p}
            
            if self.pert_type == "an":
                self.ics["p"] = lambda x, y: p0(x, y) + delta_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        x0, y0 = 0.5, 0.5
        self.exact = {
            "u": lambda x, y, t: gaussian_vortex_velocity_function(x, y, x0, y0) * (-y + y0),
            "v": lambda x, y, t: gaussian_vortex_velocity_function(x, y, x0, y0) * (x - x0),
            "p": lambda x, y, t: 1.0 - self.coriolis * gaussian_vortex_pressure_function(x, y, x0, y0)
        }
        return self.exact


class SmoothVortexTestCase(LinearAcoustic2D):
    def __init__(self, basis_name="smooth_vortex", pert_coeff=None, is_long=False, pert_type=None, is_smaller=False):
        self.is_long = is_long
        self.pert_type = pert_type
        self.is_smaller = is_smaller
        self.basis_name = basis_name
        name = ("smaller_" if self.is_smaller else "") + f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")


        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)

    def define_geom(self):
        if self.is_smaller:
            self.geometry_name = "periodic_square"
            self.BC = np.array([0, 0, 0, 0], dtype=np.int32)
        else:
            self.geometry_name = "square"
            self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        if self.pert_type == "opt":
            self.dirichlet = {"all": self.vars}
        else:
            self.dirichlet = None

    def IC(self):
        x0, y0 = 0.5, 0.5
        p0 = lambda x, y: 1.0
        
        if self.is_smaller:
            u0 = lambda x, y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.) * 2. * (-y + y0)
            v0 = lambda x, y: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.) * 2. * (x - x0)
            base_str = "smaller_smooth_vortex_long"
        else:
            u0 = lambda x, y: smooth_vortex_function(x, y, x0, y0) * (-y + y0)
            v0 = lambda x, y: smooth_vortex_function(x, y, x0, y0) * (x - x0)
            base_str = "smooth_vortex_long"
            
        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = base_str

            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            if self.is_smaller and self.pert_type != "an":
                delta_u = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                delta_p = lambda x, y: 0.0
            else:
                delta_u = lambda x, y: 0.0
                delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                
            delta_v = lambda x, y: 0.0
            self.perturbation = {"u": delta_u, "v": delta_v, "p": delta_p}
            
            if self.pert_type == "an":
                # Re-evaluate an perturbations
                d_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                self.ics["u"] = lambda x, y: u0(x, y)
                self.ics["v"] = lambda x, y: v0(x, y)
                self.ics["p"] = lambda x, y: p0(x, y) + d_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        x0, y0 = 0.5, 0.5
        if self.is_smaller:
            u0 = lambda x, y, t: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.) * 2. * (-y + y0)
            v0 = lambda x, y, t: smooth_vortex_function(2.*(x-x0), 2.*(y-y0), 0., 0.) * 2. * (x - x0)
        else:
            u0 = lambda x, y, t: smooth_vortex_function(x, y, x0, y0) * (-y + y0)
            v0 = lambda x, y, t: smooth_vortex_function(x, y, x0, y0) * (x - x0)
            
        self.exact = {"u": u0, "v": v0, "p": lambda x, y, t: 1.0}
        return self.exact


class SmoothVortexTestCaseParam(LinearAcoustic2D):
    def __init__(self, basis_name="smooth_vortex_param", pert_coeff=None, is_long=False, pert_type=None, is_smaller=False, \
                 g=9.81, r0=0.45, coeff_exp=1):
        self.is_long = is_long
        self.pert_type = pert_type
        self.is_smaller = is_smaller
        self.basis_name = basis_name
        self.g = g
        self.r0 = r0
        self.coeff_exp = coeff_exp
        name = ("smaller_" if self.is_smaller else "") + f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")

        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)

    def define_geom(self):
        if self.is_smaller:
            self.geometry_name = "periodic_square"
            self.BC = np.array([0, 0, 0, 0], dtype=np.int32)
        else:
            self.geometry_name = "square"
            self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        if self.pert_type == "opt":
            self.dirichlet = {"all": self.vars}
        else:
            self.dirichlet = None

    def IC(self):
        x0, y0 = 0.5, 0.5
        p0 = lambda x, y: 1.0

        if self.is_smaller:
            u0 = lambda x, y: smooth_vortex_function_param(2.*(x-x0), 2.*(y-y0), 0., 0., self.g, self.r0, self.coeff_exp) * 2. * (-y + y0)
            v0 = lambda x, y: smooth_vortex_function_param(2.*(x-x0), 2.*(y-y0), 0., 0., self.g, self.r0, self.coeff_exp) * 2. * (x - x0)
            base_str = "smaller_smooth_vortex_long"
        else:
            u0 = lambda x, y: smooth_vortex_function_param(x, y, x0, y0, self.g, self.r0, self.coeff_exp) * (-y + y0)
            v0 = lambda x, y: smooth_vortex_function_param(x, y, x0, y0, self.g, self.r0, self.coeff_exp) * (x - x0)
            base_str = "smooth_vortex_long"

        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = base_str

            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            if self.is_smaller and self.pert_type != "an":
                delta_u = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                delta_p = lambda x, y: 0.0
            else:
                delta_u = lambda x, y: 0.0
                delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)

            delta_v = lambda x, y: 0.0
            self.perturbation = {"u": delta_u, "v": delta_v, "p": delta_p}

            if self.pert_type == "an":
                # Re-evaluate an perturbations
                d_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                self.ics["u"] = lambda x, y: u0(x, y)
                self.ics["v"] = lambda x, y: v0(x, y)
                self.ics["p"] = lambda x, y: p0(x, y) + d_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        x0, y0 = 0.5, 0.5
        if self.is_smaller:
            u0 = lambda x, y, t: smooth_vortex_function_param(2.*(x-x0), 2.*(y-y0), 0., 0., self.g, self.r0, self.coeff_exp) * 2. * (-y + y0)
            v0 = lambda x, y, t: smooth_vortex_function_param(2.*(x-x0), 2.*(y-y0), 0., 0., self.g, self.r0, self.coeff_exp) * 2. * (x - x0)
        else:
            u0 = lambda x, y, t: smooth_vortex_function_param(x, y, x0, y0, self.g, self.r0, self.coeff_exp) * (-y + y0)
            v0 = lambda x, y, t: smooth_vortex_function_param(x, y, x0, y0, self.g, self.r0, self.coeff_exp) * (x - x0)

        self.exact = {"u": u0, "v": v0, "p": lambda x, y, t: 1.0}
        return self.exact

    def set_parameters(self, mu):
        self.g = mu[0]
        self.r0 = mu[1]
        self.coeff_exp = mu[2]

class ShuVortexTestCaseParam(LinearAcoustic2D):
    def __init__(self, basis_name="shu_vortex_param", pert_coeff=None, is_long=False, pert_type=None, is_smaller=False, x0=0.5, y0=0.5, r0_param=0.1):
        self.is_long = is_long
        self.pert_type = pert_type
        self.is_smaller = is_smaller
        self.basis_name = basis_name
        
        self.x0 = x0
        self.y0 = y0
        self.r0_param = r0_param
 
        name = ("smaller_" if self.is_smaller else "") + f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")

        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)

    def define_geom(self):
        if self.is_smaller:
            self.geometry_name = "periodic_square"
            self.BC = np.array([0, 0, 0, 0], dtype=np.int32)
        else:
            self.geometry_name = "square"
            self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        if self.pert_type == "opt":
            self.dirichlet = {"all": self.vars}
        else:
            self.dirichlet = None

    def IC(self):
        p0 = lambda x, y: 1.0
        u0 = lambda x, y: shu_vortex_function_param(x, y, self.x0, self.y0, self.r0_param) * (-(y - self.y0))
        v0 = lambda x, y: shu_vortex_function_param(x, y, self.x0, self.y0, self.r0_param) * (x - self.x0)

        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True

            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            if self.is_smaller and self.pert_type != "an":
                delta_u = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                delta_p = lambda x, y: 0.0
            else:
                delta_u = lambda x, y: 0.0
                delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)

            delta_v = lambda x, y: 0.0
            self.perturbation = {"u": delta_u, "v": delta_v, "p": delta_p}

            if self.pert_type == "an":
                # Re-evaluate an perturbations
                d_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
                self.ics["u"] = lambda x, y: u0(x, y)
                self.ics["v"] = lambda x, y: v0(x, y)
                self.ics["p"] = lambda x, y: p0(x, y) + d_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        u0 = lambda x, y, t: shu_vortex_function_param(x, y, self.x0, self.y0, self.r0_param) * (-(y - self.y0))
        v0 = lambda x, y, t: shu_vortex_function_param(x, y, self.x0, self.y0, self.r0_param) * (x - self.x0)

        self.exact = {"u": u0, "v": v0, "p": lambda x, y, t: 1.0}
        return self.exact

    def set_parameters(self, mu):
        self.x0 = mu[0]
        self.y0 = mu[1]
        self.r0_param = mu[2]   


class SourceVortexTestCase(LinearAcoustic2D):
    def __init__(self, basis_name="source_vortex", pert_coeff=None, is_long=False, pert_type=None, is_dirichlet=False):
        self.is_long = is_long
        self.pert_type = pert_type
        self.is_dirichlet = is_dirichlet
        self.basis_name = basis_name
        name = f"{basis_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "") + (f"_dirichlet" if is_dirichlet else "")

        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
        super().__init__(name, pert_coeff, T_fin)

    def define_parameters(self):
        self.x0, self.y0 = 0.5, 0.5
        self.p_source = True
        self.coefficient_source = 0.01
        self.scale_source = 100.0
        self.x0_source, self.y0_source = 0.65, 0.39

        g = lambda x, y, t=0: gaussian(x - self.x0_source, y - self.y0_source, a=self.scale_source)
        self.source = {
            "u": lambda x, y, t=0: 0.0,
            "v": lambda x, y, t=0: 0.0,
            "p": lambda x, y, t=0: (((x - self.x0_source)**2 + (y - self.y0_source)**2) * self.scale_source - 1) * 4 * self.scale_source * self.coefficient_source * g(x, y)
        }

    def define_geom(self):
        self.geometry_name = "square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1 if (self.is_dirichlet or self.pert_type == "opt") else 2], dtype=np.int32)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        if self.is_dirichlet or self.pert_type in ["opt", "long_dirichlet"]:
            self.dirichlet = {"all": self.vars}
        else:
            self.dirichlet = None

    def IC(self):
        g = lambda x, y: gaussian(x - self.x0_source, y - self.y0_source, a=self.scale_source)
        p0 = lambda x, y: 1.0
        u0 = lambda x, y: smooth_vortex_function(x, y, self.x0, self.y0) * (-y + self.y0) - 2. * self.scale_source * self.coefficient_source * g(x, y) * (x - self.x0_source)
        v0 = lambda x, y: smooth_vortex_function(x, y, self.x0, self.y0) * (x - self.x0) - 2. * self.scale_source * self.coefficient_source * g(x, y) * (y - self.y0_source)
        
        self.ics = {"u": u0, "v": v0, "p": p0}

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = "source_vortex_long"
            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
            self.perturbation = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": delta_p}
            
            if self.pert_type == "an":
                self.ics["p"] = lambda x, y: p0(x, y) + delta_p(x, y)
                return self.ics
            return self.perturbation
        return self.ics

    def generate_exact(self):
        self.exact = {
            "u": lambda x, y, t: self.ics["u"](x, y),
            "v": lambda x, y, t: self.ics["v"](x, y),
            "p": lambda x, y, t: self.ics["p"](x, y)
        }
        return self.exact


class MovingSourceTestCase(LinearAcoustic2D):
    def __init__(self, name="moving_source", pert_coeff=None):
        super().__init__(name, pert_coeff, T_fin=0.1)

    def define_parameters(self):
        self.background_speed = np.array([-0.1, 0.1])
        self.x0_source, self.y0_source = 0.65, 0.39
        self.scale_source = 100.0
        self.coefficient_source = 1e-3
        
        self.x_moving = lambda x, t: x - self.x0_source - self.background_speed[0] * t
        self.y_moving = lambda y, t: y - self.y0_source - self.background_speed[1] * t 
        
        self.grad_g = lambda x, y, t: grad_gaussian(self.x_moving(x, t), self.y_moving(y, t), a=self.scale_source)
        self.hess_g = lambda x, y, t: hess_gaussian(self.x_moving(x, t), self.y_moving(y, t), a=self.scale_source)
        self.laplace_g = lambda x, y, t: laplace_gaussian(self.x_moving(x, t), self.y_moving(y, t), a=self.scale_source)
        
        self.source = {
            "u": lambda x, y, t=0: 0.0,
            "v": lambda x, y, t=0: 0.0,
            "p": lambda x, y, t: self.coefficient_source * (-self.background_speed.T @ self.hess_g(x, y, t) @ self.background_speed + self.laplace_g(x, y, t))
        }

    def define_geom(self):
        self.geometry_name = "square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        self.dirichlet = {"all": self.vars}

    def IC(self):
        p0 = lambda x, y: 1.0 + self.coefficient_source * self.background_speed.T @ self.grad_g(x, y, 0.0)
        u0 = lambda x, y: self.coefficient_source * self.grad_g(x, y, 0.0)[0]
        v0 = lambda x, y: self.coefficient_source * self.grad_g(x, y, 0.0)[1]
        self.ics = {"u": u0, "v": v0, "p": p0}
        return self.ics

    def generate_exact(self):
        self.exact = {
            "u": lambda x, y, t: self.ics["u"](x - self.background_speed[0] * t, y - self.background_speed[1] * t),
            "v": lambda x, y, t: self.ics["v"](x - self.background_speed[0] * t, y - self.background_speed[1] * t),
            "p": lambda x, y, t: self.ics["p"](x - self.background_speed[0] * t, y - self.background_speed[1] * t)
        }
        return self.exact


class ObliqueTestCase(LinearAcoustic2D):
    def __init__(self, name="oblique", T_fin=1.0, pert_coeff=None):
        super().__init__(name, pert_coeff, T_fin)

    def define_geom(self):
        self.geometry_name = "periodic_square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([0, 0, 0, 0], dtype=np.int32)
        self._setup_geometry_folder()

    def IC(self):
        self.theta = pi / 4.0
        self.lamb = 1.0 / 4.0
        xi = lambda x, y: x * cos(self.theta) + y * sin(self.theta)
        coef = 2.0 * pi / self.lamb / cos(self.theta)
        
        self.ics = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": lambda x, y: cos(coef * xi(x, y))}
        return self.ics

    def generate_exact(self):
        self.theta = pi / 4.0
        self.lamb = 1.0 / 4.0
        xi = lambda x, y: x * cos(self.theta) + y * sin(self.theta)
        xip = lambda x, y, t: xi(x, y) + self.c * t
        xim = lambda x, y, t: xi(x, y) - self.c * t
        coef = 2.0 * pi / self.lamb / cos(self.theta)
        
        self.exact = {
            "u": lambda x, y, t: -0.5 / self.c * (cos(coef * xip(x, y, t)) - cos(coef * xim(x, y, t))) * cos(self.theta),
            "v": lambda x, y, t: -0.5 / self.c * (cos(coef * xip(x, y, t)) - cos(coef * xim(x, y, t))) * sin(self.theta),
            "p": lambda x, y, t: 0.5 * (cos(coef * xip(x, y, t)) + cos(coef * xim(x, y, t)))
        }
        return self.exact

    def set_parameters(self, mu):
        pass

    def set_final_time(self, Tf):
        self.T_fin = Tf


class RP4TestCase(LinearAcoustic2D):
    def __init__(self, name="RP4", pert_coeff=None):
        super().__init__(name, pert_coeff, T_fin=0.4)

    def define_geom(self):
        self.geometry_name = "square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self._setup_geometry_folder()

    def IC(self):
        self.ics = {
            "u": lambda x, y: 1.0 * (x > 0.5) * (y > 0.5),
            "v": lambda x, y: 0.0,
            "p": lambda x, y: 0.0
        }
        return self.ics


class StommelGyreTestCase(LinearAcoustic2D):
    def __init__(self, base_name="SG", pert_coeff=None, is_long=False, pert_type=None):
        self.is_long = is_long
        self.pert_type = pert_type

        name = f"{base_name}_{'long' if is_long else 'short'}" + (f"_{pert_type}_perturbation" if pert_type else "")
        
        T_fin = 100.0 if is_long else (0.35 if pert_type else 1.0)
            
        super().__init__(name, pert_coeff, T_fin)

    def define_parameters(self):
        self.b_SG = 1.0
        self.lambda_SG = 1.0
        self.cor_f_0 = 0.01
        self.cor_phi_0 = 0.01
        self.friction = 0.01
        self.wind_F = 0.1
        self.D_SG = 1.0

        self.coriolis_non_uniform = lambda x, y, t=0: self.cor_f_0 * y + self.cor_phi_0 
        self.source = {
            "u": lambda x, y, t=0: -self.wind_F * np.cos(np.pi * y / self.b_SG),
            "v": lambda x, y, t=0: 0.0,
            "p": lambda x, y, t=0: 0.0
        }

        self.alpha_SG = self.D_SG * self.cor_f_0 / self.friction
        self.gamma_SG = self.wind_F * np.pi / self.friction / self.b_SG
        discr = np.sqrt(self.alpha_SG**2 / 4.0 + (np.pi / self.b_SG)**2)

        self.A_SG = -self.alpha_SG / 2.0 + discr
        self.B_SG = -self.alpha_SG / 2.0 - discr
        self.k_SG = (1.0 - np.exp(self.B_SG * self.lambda_SG)) / (np.exp(self.A_SG * self.lambda_SG) - np.exp(self.B_SG * self.lambda_SG))
        self.q_SG = 1.0 - self.k_SG

    def define_geom(self):
        self.geometry_name = "square" if (self.b_SG == 1.0 and self.lambda_SG == 1.0) else "square_SG"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([self.lambda_SG, self.b_SG], dtype=np.float64)
        self.BC = np.array([1, 1, 1, 1], dtype=np.int32)
        self._setup_geometry_folder()

    def dirichlet_BC(self):
        self.dirichlet = {"all": self.vars}

    def IC(self):
        self.ics = {
            "u": lambda x, y: SG_u(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG),
            "v": lambda x, y: SG_v(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG),
            "p": lambda x, y: SG_p(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG, self.wind_F, self.cor_phi_0, self.cor_f_0)
        }

        if self.pert_type:
            self.with_perturbation = True
            self.base_test = "SG_long"

            if self.pert_type != "an":
                self.steady_state_test = type(self)(is_long=True)

            x0_p, y0_p = 0.4, 0.43
            delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
            self.perturbation = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": delta_p}
            return self.perturbation
        return self.ics

    def generate_exact(self):
        self.exact = {
            "u": lambda x, y, t: SG_u(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG),
            "v": lambda x, y, t: SG_v(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG),
            "p": lambda x, y, t: SG_p(x, y, self.b_SG, self.gamma_SG, self.k_SG, self.q_SG, self.A_SG, self.B_SG, self.wind_F, self.cor_phi_0, self.cor_f_0)
        }
        return self.exact


class ConstantFlowTestCase(LinearAcoustic2D):
    def __init__(self, name="constant_flow", pert_coeff=None, pert_type=None):
        self.pert_type = pert_type
        super().__init__(name, pert_coeff, T_fin=0.35)

    def define_geom(self):
        self.geometry_name = "periodic_square"
        self.xL = np.array([0., 0.], dtype=np.float64)
        self.xR = np.array([1., 1.], dtype=np.float64)
        self.BC = np.array([0, 0, 0, 0], dtype=np.int32)
        self._setup_geometry_folder()

    def IC(self):
        self.ics = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": lambda x, y: 1.0}

        if self.pert_type == "an":
            self.with_perturbation = True
            self.base_test = "constant_flow"

            x0_p, y0_p = 0.4, 0.43
            delta_p = lambda x, y: self.pert_coeff * pressure_perturbation_function(x, y, x0_p, y0_p)
            self.perturbation = {"u": lambda x, y: 0.0, "v": lambda x, y: 0.0, "p": delta_p}
            self.ics["p"] = lambda x, y: 1.0 + delta_p(x, y)
            return self.ics
        return self.ics
    



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
    g = 9.81
    Gam = 0.2                               #(12.*pi*np.sqrt(deltah*g))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    rho = np.minimum((r/r0)**2,1.-1e-16)
    u = (r < r0)*2*Gam*np.exp(-1/(2*(1-rho)**2))*np.sqrt(g/(r0*(1-rho)**3))
    return u

def smooth_vortex_function_param(x, y, x0, y0, g, r0, coeff_exp):
    Gam = 0.2                               #(12.*pi*np.sqrt(deltah*g))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    rho = np.minimum((r/r0)**2,1.-1e-16)
    u = (r < r0)*2*Gam*np.exp(-coeff_exp/(2*(1-rho)**2))*np.sqrt(g/(r0*(1-rho)**3))
    return u

def shu_vortex_function_param(x, y, x0, y0, r0_param):                             # vortex intensity parameter
    Gam = 0.2
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    omega=Gam*np.exp(-(r/r0_param)**2)
    return omega

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