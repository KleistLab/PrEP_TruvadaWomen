import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from .ode_solver import euler
from .utils import Drug, DrugClass


class AbstractPharmacokinetics(ABC):
    """
    Base class for pharmacokinetics.

    Parameters:
    regimen: 'Regimen' object
        a Regimen object containing all PK relevant parameters
    sample_file: str
        name of xlsx file contains the PK parameters. If None, a default sample will be assigned.
    ratio: int
            ratio to slice the PK profile
            slice the concentration profile in the way that every 'ratio' data points will be stored instead of all
            (useful when the time step of PK smaller than time step of PGS)
    Attributes:
    regimen: Regimen object
    _sample: ndarray (n_samples, n_parameters)
        matrix contains the PK parameters of each sample
    _coefficient_matrix: ndarray (, n_samples, n_compartments, n_compartments)
        coefficient matrix to solve PK ODE
    _concentration_profile_whole: ndarray (n_steps, n_samples, [n_regimen, ], n_compartment)
        matrix of concentration profile of each regimen patterns in regimen for all samples, containing the
        profiles for all compartments in PK model
    _concentration_profile_target: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        matrix of concentration of the target compartment
    _sample_file: str, file name
    _ode_solver: callable
        function to solve ODE, take 5 parameters:
            fun: callable
                Right-hand side of the system, the calling signature is fun(t, y)
            t_start, t_end: double
                Time interval of integration.
            y0: double, array-like
                Initial values of the ivp
            _time_step: double
                step size for solving ODE, positive
    _time_step: float
        time step [hr] for solving ODE
    """

    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        self.regimen = regimen
        self._samples = None
        self._ratio = ratio
        self._coefficient_matrix = None
        self._concentration_profile_whole = None
        self._concentration_profile_target = None
        self._sample_file = sample_file
        self._time_span = regimen.get_timespan()
        self._time_step = time_step
        self._ode_solver = ode_solver

    def read_sample_file(self, file_name):
        """
        Read another file instead of using the default PK parameters. This function can be used if
        no sample file is given during initialization.
        :param
        file_name: str
            file contains the PK parameters
        :return:
        """
        if self._sample_file:
            raise SystemExit('Sample file is already given. Generate a new instance for a new sample file\n')
        self._sample_file = file_name
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()

    @abstractmethod
    def _generate_sample_parameters(self):
        """
        Generate a matrix containing the PK parameters of every sample

        :return:
        parameter: double ndarray (n_samples, n_parameters)
            matrix containing the PK parameters
        """
        pass

    @abstractmethod
    def _generate_pk_coefficient_matrix(self):
        """
        Generate a matrix containing coefficients in PK ODE system

        :return:
        terms: double ndarray (n_samples, [n_regimen, ], n_compartments_z, n_compartments_z)
            matrix containing the constants in PK ODE
        """
        pass

    def _pk_model(self, t, z):
        """
        update the concentration and return the right hand side of the PK ODE

        :parameter:
        t: double
            Current time point during ODE solving
        z: double array (n_samples, [n_regimen, ], n_compartments_z, 1)
            Current state of concentrations in each compartment
        :return:
        model: double ndarray (n_samples, [n_regimen, ], n_compartments_z, 1)
            The right hand side of PK model
        """
        pass

    def _compute_concentration(self):
        """
        Compute the PK profiles using self._ode_solver for self.regimen
        """
        raise NotImplementedError

    def get_concentration(self):
        return self._concentration_profile_target

    def set_concentration(self, proportion):
        self._concentration_profile_target = self._concentration_profile_target * proportion

    def get_concentration_whole(self):
        return self._concentration_profile_whole

    def _slice_concentration_array(self):
        steps_of_input_span = int((self._time_span[1] - self._time_span[0]) / self._time_step)
        if self._concentration_profile_target.shape[0] - 1 > steps_of_input_span:
            diff = self._concentration_profile_target.shape[0] - steps_of_input_span - 1
            self._concentration_profile_target = self._concentration_profile_target[diff:]

    @staticmethod
    def get_pk_class(regimen, file=None, ratio=1, ode_solver=euler, time_step=0.02):
        """
        map a drug (of a regimen) to its corresponding pk class. Must be expanded if pk class of new drug is added.
        :parameter:
         regimen: Regimen object
         file: str
            name of file contains the PK parameters
        :return:
        Pharmacokinetics_ : class object
            corresponding pk class for the drug
        """
        if regimen.get_drug() is Drug.DTG:
            return PharmacokineticsDTG(regimen, file, ratio, ode_solver, time_step)

        if regimen.get_drug() is Drug.EFV:
            return PharmacokineticsEFV(regimen, file, ratio, ode_solver, time_step)

        if regimen.get_drug() is Drug.ISL:
            return PharmacokineticsISL(regimen, file, ratio, ode_solver, time_step)

        # comment this condition out to use the old model of NRTI
        if regimen.get_drug() is Drug.TDF:              # new PK model of TDF (only for women), PMID: 25581815
            return PharmacokineticsTDF(regimen, file, ratio, ode_solver, time_step)

        # comment this condition out to use the old model of NRTI
        if regimen.get_drug() is Drug.FTC:              # new PK model of FTC (only for women), PMID: 30150483
            return PharmacokineticsFTC(regimen, file, ratio, ode_solver, time_step)

        if regimen.get_drug_class() is DrugClass.NRTI:      # old model for TDF and FTC, PMID: 27439573
            return PharmacokineticsNRTI(regimen, file, ratio, ode_solver, time_step)

    def set_ode_solver(self, ode_solver):
        """
        set the ODE solver
        """
        self._ode_solver = ode_solver

    def set_time_step(self, t_step):
        """
        set the time step for solving ODE
        """
        self._time_step = t_step


class PharmacokineticsDTG(AbstractPharmacokinetics):
    """
    unit [nM]
    """
    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        super().__init__(regimen, sample_file, ratio, ode_solver, time_step)
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()
        self._concentration_profile_target = self._concentration_profile_whole[..., [1]]
        self._slice_concentration_array()

    def _generate_sample_parameters(self):
        """
        :return:
        param: double ndarray (n_samples, 5)
        """
        if self._sample_file:
            import warnings
            warnings.simplefilter("ignore")
            df = pd.read_excel(self._sample_file)
            param = torch.tensor(df.iloc[:, [5, 2, 1, 3, 4]].values, dtype=torch.double)
        else:
            param = torch.tensor([[2.24, 17.7, 0.85, 0.0082, 0.73]], dtype=torch.double)
        return param

    def _generate_pk_coefficient_matrix(self):
        """
        :return:
        term: double ndarray (n_samples, [n_regimen, ], 3, 3)
        """
        term = torch.zeros([self._samples.shape[0], 3, 3], dtype=torch.double)
        term[:, 0, 0] = -self._samples[:, 0]
        term[:, 1, 0] = self._samples[:, 0] / self._samples[:, 1]
        term[:, 1, 1] = -(self._samples[:, 2] + self._samples[:, 3]) / self._samples[:, 1]
        term[:, 1, 2] = self._samples[:, 3] / self._samples[:, 4]
        term[:, 2, 1] = self._samples[:, 3] / self._samples[:, 1]
        term[:, 2, 2] = -term[:, 1, 2]
        regimen_matrix = self.regimen.get_regimen_matrix()
        for dim in range(len(regimen_matrix.shape) - 1):
            term = term.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        term = term.repeat([1] + list(regimen_matrix.shape)[:-1] + [1, 1])
        return term

    def _pk_model(self, t, z, *args):
        """
        :param **kwargs:
        :parameter:
        t: double
        z: double array (n_samples, [n_regimen, ], 3, 1) (Z1, Z2, Z3)
        :return:
        model: double ndarray (n_samples, [n_regimen, ], 3, 1)
        """
        return torch.matmul(self._coefficient_matrix, z)

    def _compute_concentration(self):
        """
        :return:
        concentration: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        """

        tmp_concentration = list()
        period = self.regimen.get_period()
        regimen_matrix = self.regimen.get_regimen_matrix()
        molecular_weight = self.regimen.get_molecular_weight()
        z0 = torch.zeros([self._samples.shape[0]] + list(regimen_matrix.shape)[:-1] + [3, 1],
                         dtype=torch.double)
        z_array = None  # initiate variable for later use.
        for i in range(regimen_matrix.shape[-1]):
            z0[..., 0, 0] = z0[..., 0, 0] + regimen_matrix[..., i] * 1e6 / molecular_weight
            z_array = self._ode_solver(self._pk_model, i * period, (i + 1) * period, z0, self._time_step)
            z0 = z_array[-1].clone()
            tmp_concentration.append(z_array[:-1:self._ratio])      # ignore the last point to avoid duplicate
        tmp_concentration.append(z_array[[-1]])       # add the last point of the last iteration
        concentration = torch.cat(tmp_concentration).squeeze(dim=-1)
        return concentration


class PharmacokineticsNRTI(AbstractPharmacokinetics):
    """
    old PK model for NRTI (of male), from paper doi:10.1002/psp4.12095
    unit [uM]
    """
    param_dict = {'3TC': [],
                  'FTC': [0.93, 0.542, 43.823, 0.409, 0.113, 0.082, 0.6191, 0.9464, 0.0176],
                  'TDF': [0.32, 1, 110.31, 0.1234, 0.2926, 0.1537, 0.0032, 0.1020, 0.006]
                  }
    """
    param_dict: dict
        dictionary for PK parameters. [Fbio, ka, V1, ke, k12, k21, Vmax, Km, kout]
    """

    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        super().__init__(regimen, sample_file, ratio, ode_solver, time_step)
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()
        self._concentration_profile_target = self._concentration_profile_whole[..., [2]]
        self._slice_concentration_array()

    def _generate_sample_parameters(self):
        """
        :return:
        param: double ndarray (n_samples, 5)
        """
        if self._sample_file:
            df = pd.read_excel(self._sample_file)
            param = torch.tensor(df.iloc[:, 0:9].values, dtype=torch.double)
        else:
            drug = self.regimen.get_drug_name()
            param = torch.tensor([PharmacokineticsNRTI.param_dict[drug]], dtype=torch.double)
        return param

    def _generate_pk_coefficient_matrix(self):
        """
        :return:
        term: double ndarray (n_samples, [n_regimen, ], 4, 4)
        """
        term = torch.zeros([self._samples.shape[0], 4, 4], dtype=torch.double)
        term[:, 0, 0] = -self._samples[:, 3] - self._samples[:, 4]
        term[:, 0, 1] = self._samples[:, 5]
        term[:, 0, 3] = self._samples[:, 1] * self._samples[:, 0] / self._samples[:, 2]
        term[:, 1, 0] = self._samples[:, 4]
        term[:, 1, 1] = -self._samples[:, 5]
        term[:, 2, 0] = self._samples[:, 6]
        term[:, 2, 2] = -self._samples[:, 8]
        term[:, 3, 3] = -self._samples[:, 1]
        regimen_matrix = self.regimen.get_regimen_matrix()
        for dim in range(len(regimen_matrix.shape) - 1):
            term = term.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        term = term.repeat([1] + list(regimen_matrix.shape)[:-1] + [1, 1])
        return term

    def _pk_model(self, t, c):
        """
        :param **kwargs:
        :parameter:
        t: double
        c: double array (n_samples, [n_regimen, ], 4, 1)  (C1, C2, C3, D)
        :return:
        model: double ndarray (n_samples, [n_regimen, ], 4, 1)
        """
        coefficient_matrix = self._coefficient_matrix.clone()
        coefficient_matrix[..., 2, 0] = torch.div(coefficient_matrix[..., 2, 0], (c[..., 0, 0] + self._samples[:, 7]))
        return torch.matmul(coefficient_matrix, c)

    def _compute_concentration(self):
        """
        :return:
        concentration: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        """
        tmp_concentration = list()
        period = self.regimen.get_period()
        regimen_matrix = self.regimen.get_regimen_matrix()
        molecular_weight = self.regimen.get_molecular_weight()
        d0 = torch.zeros([self._samples.shape[0]] + list(regimen_matrix.shape)[:-1] + [4, 1], dtype=torch.double)
        d_array = None  # initiate variable for later use.
        for i in range(regimen_matrix.shape[-1]):
            d0[..., 3, 0] = d0[..., 3, 0] + regimen_matrix[..., i] * 1e3 / molecular_weight
            d_array = self._ode_solver(self._pk_model, i * period, (i + 1) * period, d0, self._time_step)
            d0 = d_array[-1].clone()
            tmp_concentration.append(d_array[:-1:self._ratio])  # ignore the last point to avoid duplicate
        tmp_concentration.append(d_array[[-1]])  # add the last point of the last iteration
        concentration = torch.cat(tmp_concentration).squeeze(dim=-1)
        return concentration


class PharmacokineticsTDF(AbstractPharmacokinetics):
    """
    PK model for TDF (of female), from paper Burns et al. PMID: 25581815
    unit [uM]
    IMPORTANT: if the PK parameters in file burnssimparam.csv is used, the time step and ODE solver should be
    0.002 and RK4, otherwise the solved PK values can overflow.
    """

    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        super().__init__(regimen, sample_file, ratio, ode_solver, time_step)
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()
        self._concentration_profile_target = self._concentration_profile_whole[..., [2]]
        self._slice_concentration_array()

    def _generate_sample_parameters(self):
        """
        :return:
        param: double ndarray (n_samples, 7), [Ka, K23, K32, K24, K40, K20, Vc]
        """
        if self._sample_file:
            df = pd.read_csv(self._sample_file, index_col='ID', dtype=float)
            param = torch.tensor(df.iloc[:, 0:].values, dtype=torch.double)
        else:       # median of the PK parameters of 1000 individuals
            param = torch.tensor([[9.24, 0.631, 0.396, 0.016, 1.3e-2, 0.1313, 360.94]], dtype=torch.double)
        return param

    def _generate_pk_coefficient_matrix(self):
        """
        :return:
        term: double ndarray (n_samples, [n_regimen, ], 4, 4)
        """
        term = torch.zeros([self._samples.shape[0], 4, 4], dtype=torch.double)
        term[:, 0, 0] = -self._samples[:, 1] - self._samples[:, 3] - self._samples[:, 5]
        term[:, 0, 1] = self._samples[:, 2]
        term[:, 0, 3] = self._samples[:, 0]
        term[:, 1, 0] = self._samples[:, 1]
        term[:, 1, 1] = -self._samples[:, 2]
        term[:, 2, 0] = self._samples[:, 3]
        term[:, 2, 2] = -self._samples[:, 4]
        term[:, 3, 3] = -self._samples[:, 0]
        regimen_matrix = self.regimen.get_regimen_matrix()
        for dim in range(len(regimen_matrix.shape) - 1):
            term = term.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        term = term.repeat([1] + list(regimen_matrix.shape)[:-1] + [1, 1])
        return term

    def _pk_model(self, t, c):
        """
        :param **kwargs:
        :parameter:
        t: double
        c: double array (n_samples, [n_regimen, ], 4, 1) (C1, C2, C3, D)
        :return:
        model: double ndarray (n_samples, [n_regimen, ], 4, 1)
        """
        return torch.matmul(self._coefficient_matrix, c)

    def _compute_concentration(self):
        """
        :return:
        concentration: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        """
        tmp_concentration = list()
        period = self.regimen.get_period()
        regimen_matrix = self.regimen.get_regimen_matrix()
        molecular_weight = self.regimen.get_molecular_weight()
        d0 = torch.zeros([self._samples.shape[0]] + list(regimen_matrix.shape)[:-1] + [4, 1], dtype=torch.double)
        d_array = None  # initiate variable for later use.
        samples = self._samples.clone()                         # generate a copy of pk parameters of samples
        for dim in range(len(regimen_matrix.shape) - 1):        # for C1: central compartment divided by central volume
            samples = samples.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        samples = samples.repeat([1] + list(regimen_matrix.shape)[:-1] + [1])
        for i in range(regimen_matrix.shape[-1]):
            d0[..., 3, 0] = d0[..., 3, 0] + regimen_matrix[..., i] * 1e3 / molecular_weight
            d_array = self._ode_solver(self._pk_model, i * period, (i + 1) * period, d0, self._time_step)
            d0 = d_array[-1].clone()
            d_array[..., 0, 0] = torch.div(d_array[..., 0, 0], samples[..., 6])  # C1 / Vc
            tmp_concentration.append(d_array[:-1:self._ratio])  # ignore the last point to avoid duplicate
        tmp_concentration.append(d_array[[-1]])  # add the last point of the last iteration
        # Cellular compartment divided by cellular volume 1000
        concentration = torch.cat(tmp_concentration).squeeze(dim=-1)
        concentration[..., 2] = concentration[..., 2] / 1000
        return concentration


class PharmacokineticsFTC(AbstractPharmacokinetics):
    """
    PK model for FTC (of female), from paper Garrett et al. PMID: 30150483
    unit [uM]
    """

    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        super().__init__(regimen, sample_file, ratio, ode_solver, time_step)
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()
        self._concentration_profile_target = self._concentration_profile_whole[..., [2]]
        self._slice_concentration_array()

    def _generate_sample_parameters(self):
        """
        :return:
        param: double ndarray (n_samples, 9) [Ka, CL, Vmax, Km, CLp, CLd, V2, V3, V4 ]
        """
        if self._sample_file:
            df = pd.read_csv(self._sample_file, index_col='ID', dtype=float)
            param = torch.tensor(df.iloc[:, 0:].values, dtype=torch.double)
        else:       # median of the PK parameters of 1000 individuals
            param = torch.tensor([[0.596, 21.87, 6.93,	11.4, 0.74,	7.482,	60.40, 0.989, 118.515]], dtype=torch.double)
        return param

    def _generate_pk_coefficient_matrix(self):
        """
        :return:
        term: double ndarray (n_samples, [n_regimen, ], 4, 4)
        """
        term = torch.zeros([self._samples.shape[0], 4, 4], dtype=torch.double)
        term[:, 0, 0] = -(self._samples[:, 1] + self._samples[:, 5]) / self._samples[:, 6]
        term[:, 0, 1] = self._samples[:, 5] / self._samples[:, 8]
        term[:, 0, 2] = self._samples[:, 4] / self._samples[:, 7]
        term[:, 0, 3] = self._samples[:, 0]
        term[:, 1, 0] = self._samples[:, 5] / self._samples[:, 6]
        term[:, 1, 1] = -self._samples[:, 5] / self._samples[:, 8]
        term[:, 2, 0] = self._samples[:, 2]
        term[:, 2, 2] = -self._samples[:, 4] / self._samples[:, 7]
        term[:, 3, 3] = -self._samples[:, 0]
        regimen_matrix = self.regimen.get_regimen_matrix()
        for dim in range(len(regimen_matrix.shape) - 1):
            term = term.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        term = term.repeat([1] + list(regimen_matrix.shape)[:-1] + [1, 1])
        return term

    def _pk_model(self, t, c):
        """
        :param **kwargs:
        :parameter:
        t: double
        c: double array (n_samples, [n_regimen, ], 4, 1) (C1, C2, C3, D)
        :return:
        model: double ndarray (n_samples, [n_regimen, ], 4, 1)
        """
        coefficient_matrix = self._coefficient_matrix.clone()
        coefficient_matrix[..., 2, 0] = torch.div(coefficient_matrix[..., 2, 0], (c[..., 0, 0] + self._samples[..., [3]]))
        coefficient_matrix[..., 0, 0] = coefficient_matrix[..., 0, 0] \
            - torch.div(self._samples[..., [2]], (c[..., 0, 0] + self._samples[..., [3]]))
        return torch.matmul(coefficient_matrix, c)

    def _compute_concentration(self):
        """
        :return:
        concentration: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        """
        tmp_concentration = list()
        period = self.regimen.get_period()
        regimen_matrix = self.regimen.get_regimen_matrix()
        molecular_weight = self.regimen.get_molecular_weight()
        d0 = torch.zeros([self._samples.shape[0]] + list(regimen_matrix.shape)[:-1] + [4, 1], dtype=torch.double)
        d_array = None  # initiate variable for later use.
        samples = self._samples.clone()                    # generate a copy of pk parameters of samples
        for dim in range(len(regimen_matrix.shape) - 1):        # add dimension to self._samples for later use:
            samples = samples.unsqueeze(dim + 1)      # C1 and C3 have to be divided by corresponding volume
        samples = samples.repeat([1] + list(regimen_matrix.shape)[:-1] + [1])
        for i in range(regimen_matrix.shape[-1]):
            d0[..., 3, 0] = d0[..., 3, 0] + regimen_matrix[..., i] * 1e3 / molecular_weight
            d_array = self._ode_solver(self._pk_model, i * period, (i + 1) * period, d0, self._time_step)
            d0 = d_array[-1].clone()
            tmp_concentration.append(d_array[:-1:self._ratio])  # ignore the last point to avoid duplicate
        tmp_concentration.append(d_array[[-1]])  # add the last point of the last iteration
        # Cellular compartment divided by cellular volume 1000
        concentration = torch.cat(tmp_concentration).squeeze(dim=-1)
        concentration[..., 0] = torch.div(concentration[..., 0],
                                          samples[..., 6])  # C1: central compartment divided by central volume
        concentration[..., 2] = torch.div(concentration[..., 2],
                                          samples[..., 7])  # C3: cellular compartment divided by cellular volume
        return concentration


class PharmacokineticsEFV(AbstractPharmacokinetics):
    """
    PK model for EFV
    unit [nM]
    """

    def __init__(self, regimen, sample_file=None, ratio=1, ode_solver=euler, time_step=0.02):
        super().__init__(regimen, sample_file, ratio, ode_solver, time_step)
        self._samples = self._generate_sample_parameters()
        self._coefficient_matrix = self._generate_pk_coefficient_matrix()
        self._concentration_profile_whole = self._compute_concentration()
        self._concentration_profile_target = self._concentration_profile_whole[..., [1]]
        self._slice_concentration_array()

    def _generate_sample_parameters(self):
        """
        :return:
        param: double ndarray (n_samples, 3) [Ka, Vd ,CLss]
        """
        if self._sample_file:
            df = pd.read_csv(self._sample_file, index_col='ID', dtype=float)
            param = torch.tensor(df.iloc[:, 0:].values, dtype=torch.double)
        else:       # median of the PK parameters of 1000 individuals
            param = torch.tensor([[0.6, 266.185, 8.9579]], dtype=torch.double)
        return param

    def _generate_pk_coefficient_matrix(self):
        """
        :return:
        term: double ndarray (n_samples, [n_regimen, ], 2, 2)
        """
        term = torch.zeros([self._samples.shape[0], 2, 2], dtype=torch.double)
        term[:, 0, 0] = -self._samples[:, 0]
        term[:, 1, 0] = self._samples[:, 0] / self._samples[:, 1]
        term[:, 1, 1] = -self._samples[:, 2] / self._samples[:, 1]

        regimen_matrix = self.regimen.get_regimen_matrix()
        for dim in range(len(regimen_matrix.shape) - 1):
            term = term.unsqueeze(dim + 1)                  # add dimension according to regimen_matrix
        term = term.repeat([1] + list(regimen_matrix.shape)[:-1] + [1, 1])
        return term

    def _pk_model(self, t, c):
        """
        :param **kwargs:
        :parameter:
        t: double
        c: double array (n_samples, [n_regimen, ], 2, 1) (C_dose, C_central)
        :return:
        model: double ndarray (n_samples, [n_regimen, ], 2, 1)
        """
        _alpha = 1.7124     # parameter for the autoinduction of EFV
        _t_50 = 245         # parameter for the autoinduction of EFV
        coefficient_matrix = self._coefficient_matrix.clone()
        coefficient_matrix[..., 1, 1] = coefficient_matrix[..., 1, 1] * \
                                        (1 / _alpha + (1 - 1 / _alpha) * (t - self._t0) / (t - self._t0 + 245))
        return torch.matmul(coefficient_matrix, c)

    def _compute_concentration(self):
        """
        :return:
        concentration: ndarray (n_steps, n_samples, [n_regimen, ], 1)
        """
        tmp_concentration = list()
        period = self.regimen.get_period()
        regimen_matrix = self.regimen.get_regimen_matrix()
        molecular_weight = self.regimen.get_molecular_weight()
        d0 = torch.zeros([self._samples.shape[0]] + list(regimen_matrix.shape)[:-1] + [2, 1], dtype=torch.double)
        d_array = None  # initiate variable for later use.
        t0_idx = 0
        while not regimen_matrix[0, t0_idx]:
            t0_idx += 1
        self._t0 = t0_idx * period      # to update CL(t) for autoinduction. Important: only be applied on 1D regimen!!
        for i in range(regimen_matrix.shape[-1]):
            d0[..., 0, 0] = d0[..., 0, 0] + regimen_matrix[..., i] * 1e6 / molecular_weight
            d_array = self._ode_solver(self._pk_model, i * period, (i + 1) * period, d0, self._time_step)
            d0 = d_array[-1].clone()
            tmp_concentration.append(d_array[:-1:self._ratio])  # ignore the last point to avoid duplicate
        tmp_concentration.append(d_array[[-1]])  # add the last point of the last iteration
        concentration = torch.cat(tmp_concentration).squeeze(dim=-1)
        return concentration



