from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch

from .ode_solver import euler
from .utils import ViralDynamicParameter

# TODO: solving ODE using RK4 is not totally correct, because f(t + h/2) = f(t)


class AbstractPgSystem(ABC):
    """
    Base class for PGS model

    Parameters:
    -----------
    strain_manager: StrainManager object
        object, contain the total propensities computed from all strains
    time_span: tuple
        overall time span (all regimens should be rescaled so that all pk objects
        have the same length as time_span/time_step)
    """
    _time_step = 0.02
    _ode_solver = euler
    """
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

    def __init__(self, pd_interface, time_span):
        self._pd_interface = pd_interface
        self._time_span = time_span
        self._steps, self._shape = self._get_shape()

    def _get_shape(self):
        """
        extract the dimension of the drug-dependent propensity.
        :return:
        step: int
            count of steps (by PK ODE solving)
        shape: array-like
            list of dimension (n_samples, [, n_period_of_regimen])
        """
        propensity = self._pd_interface.get_propensity()
        for value in propensity.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0] - 1, list(value.shape)[1:]
        else:
            raise SystemExit('Program will break, possible reason: no available regimen. ')

    @abstractmethod
    def _pgs_model(self, t, p, coeff_mat, const_mat, varying_propensities):
        """
        Right hand side of PGS model in vectorized form,
        by default pgs3 (with V, T1 and T2 three compartments), can be overwritten.
        Parameters:
        t: double
            Current time point
        p: double ndarray (self._shape, n_vd_compartments, 1)
            Matrix of extinction probability for current time point
        coeff_mat: double 2darray (n_vd_compartments, n_vd_compartments)
            Square matrix of coefficients, dimension depends on the viral dynamic model
        const_mat: double 2darray (n_vd_compartments, 1)
            column matrix of constants, dimension depends on the viral dynamic model
        varying_propensities: array-like
            contains the propensities whose values are time-dependent, each element is a tuple consist of
            the index (1-6) and array of the propensity
        :return:
        model: double ndarray (self._shape, n_vd_compartments, n_vd_compartments)
            The right hand side of PGS model
        """
        pass

    @classmethod
    def set_ode_solver(cls, ode_solver):
        cls._ode_solver = ode_solver

    @classmethod
    def set_time_step(cls, t_step):
        cls._time_step = t_step


class PgSystemExtinction(AbstractPgSystem):
    """
    PGS class compute the extinction probability

    Attributes:
    _reservoir: boolean, if reservoir is considered
    pos_dict_coeff: dictionary contains the position and sign of each propensity (a1-a6) in 
        the coefficient matrix of PGS ODE. (i, j) denotes the position, +/-1 denotes the sign. 
    """
    def __init__(self, pd_interface, time_span, reservoir=False):
        super().__init__(pd_interface, time_span)
        self._reservoir = reservoir
        self.pos_dict_coeff = {1: [[(0, 0), +1]],
                               2: [[(1, 1), +1]],
                               3: [[(2, 2), +1]],
                               4: [[(0, 0), +1], [(0, 1), -1 * (1 - self._reservoir * ViralDynamicParameter.P_Ma4)]],
                               5: [[(1, 1), +1], [(1, 2), -1 * (1 - self._reservoir * ViralDynamicParameter.P_La5)]],
                               6: [[(2, 0), -1], [(2, 2), +1]]}

    def _pgs_model(self, t, p, coeff_mat, const_mat, varying_propensities):
        coeff_mat = coeff_mat.repeat(self._shape + [1, 1])
        const_mat = const_mat.repeat(self._shape + [1, 1])
        step = round((t - self._time_span[0]) / self._time_step)
        propensity = self._pd_interface.get_propensity()
        for idx, propensity_info in varying_propensities:   # for each time-dependent propensity
            value = propensity[idx][step]
            for pos, sign in propensity_info:
                coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + value * sign
            if idx < 4:     # update constant matrix
                const_mat[..., idx - 1, 0] = value
        # update coeff_matrix[2,0] by multiplying P3
        # Caution: P3 can only be multiplied once!! ()
        coeff_mat[..., 2, 0] = torch.mul(coeff_mat[..., 2, 0], p[..., 2, 0])
        return torch.matmul(coeff_mat, p) - const_mat

    def compute_pe(self):
        """
        Compute the extinction probability. dP/dt = coeff_mat * P - const_mat
        :return:
        result: ndarray (n_points, n_sample, [n_regimen, ], 3*n_strain, 1), extinction probability profile
        """
        coeff_mat = torch.zeros((3, 3), dtype=torch.double)
        const_mat = torch.zeros((3, 1), dtype=torch.double)
        time_dependent_propensity = []      # store the info of time-dependent propensities
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 7):         # iterate over a1-a6
            if isinstance(propensity_dict[idx], float):
                for pos, sign in self.pos_dict_coeff[idx]:      # iterate over each position in coefficient matrix
                    coeff_mat[pos[0], pos[1]] = coeff_mat[pos[0], pos[1]] + propensity_dict[idx] * sign
                if idx < 4:
                    const_mat[idx - 1, 0] = propensity_dict[idx]
            else:       # store the index of propensity (1-6), position and sign info
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat, 'varying_propensities': time_dependent_propensity}
        helpfunc = partial(self._pgs_model, **kwarg)
        p0 = torch.zeros(self._shape + [3, 1], dtype=torch.double)
        return type(self)._ode_solver(helpfunc, self._time_span[1], self._time_span[0], p0, self._time_step)

    def _pgs_model_cumulative(self, t, p, coeff_mat, const_mat):
        """
        Right hand side of basic PGS model in vectorized form (V, T1, T2), used to compute the cumulative probability
        :parameter:
        t: float
        p: ndarray (self._shape, 3*n_strain, 1)
        coeff_mat: ndarray (3*n_strain, 3*n_strain)
        const_mat: ndarray (3*n_strain, 1)
        :return:
        model: ndarray (self._shape, 3*n_strain, 1)
            The right hand side of the PGS ODE
        """
        step = round(t / self._time_step) - 1
        coeff_term = coeff_mat[step].clone()
        const_term = const_mat[step]
        coeff_term[..., 2, 0] = torch.mul(coeff_term[..., 2, 0], p[..., 2, 0])
        return torch.matmul(coeff_term, p) - const_term

    def compute_pe_cumulative(self):
        """
        Compute the cumulative extinction probability. The result is cumulative extinction probability profile
        with 1 point each hour. For example, if the time span is 0hr - 100hr, there will be 100 points on the
        cumulative probability profile. Since the PK and corresponding propensity have more data points
        (100 / self._time_step where self._time_Step < 1), these points will be separated into 100 groups, each has
        n = 1 / self._time_step points. The propensity array will be reformed into dimension (n x 100).
        Solving PGS will be processed with inside each group, with n steps in each iteration, 100 iterations in total.
        After each iteration, the first group can be removed from the coefficient matrix and P array, because the
        iteration is already over and the last-step value of this group denotes the cumulative probability of
        the first point.
        :return:
        result: ndarray (n_points, n_sample, [n_regimen, ], 3*n_strain, 1)
            cumulative probability profile
        """
        n_points = round(self._time_span[1] - self._time_span[0])
        step_each_iter = self._steps // n_points
        coeff_mat = torch.zeros([step_each_iter, n_points] + self._shape + [3, 3], dtype=torch.double)
        const_mat = torch.zeros([step_each_iter, n_points] + self._shape + [3, 1], dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 7):
            if isinstance(propensity_dict[idx], float):       # add the constant terms into coeff_mat
                for pos, sign in self.pos_dict_coeff[idx]:
                    coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + propensity_dict[idx] * sign
                if idx < 4:
                    const_mat[..., idx - 1, 0] = propensity_dict[idx]     # add the constant terms into const_mat
            else:       # store the info of time-varying propensity
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        for idx, propensity_info in time_dependent_propensity:
            # get rid of additional value at the beginning
            propensity_array = propensity_dict[idx][1:]
            for j in range(step_each_iter):
                for pos, sign in propensity_info:                    # update coeff_mat with time-varying propensity
                    coeff_mat[j, ..., pos[0], pos[1]] = \
                        coeff_mat[j, ..., pos[0], pos[1]] + propensity_array[j::step_each_iter] * sign
                if idx < 4:                                         # update const_mat with time-varying propensity
                    const_mat[j, ..., idx - 1, 0] = propensity_array[j::step_each_iter]
        p0 = torch.zeros([n_points] + self._shape + [3, 1], dtype=torch.double)
        result = torch.zeros([n_points] + self._shape + [3, 1], dtype=torch.double)
        for i in range(n_points):
            kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat}
            helpfunc = partial(self._pgs_model_cumulative, **kwarg)
            p0 = type(self)._ode_solver(helpfunc,
                                        step_each_iter * self._time_step,
                                        0, p0, self._time_step)
            result[i] = p0[0, 0]                        # update result with the first value of the first point
            p0 = p0[0, 1:]                              # remove all values of the first point
            coeff_mat = coeff_mat[:, :-1]               # remove the last group of coeff and const matrix
            const_mat = const_mat[:, :-1]               # since they are already outdated
        return result


class PgSystemInfection(AbstractPgSystem):
    """
    PGS class to compute infection probability
    """
    def __init__(self, pd_interface, time_span):
        super().__init__(pd_interface, time_span)
        self.pos_dict_coeff = {1: [[(0, 0), +1]],
                               2: [[(1, 1), +1]],
                               3: [[(2, 2), +1]],
                               4: [[(0, 0), +(1-ViralDynamicParameter.P_Ma4)],
                                   [(0, 1), -(1-ViralDynamicParameter.P_Ma4)]],
                               5: [[(1, 1), +(1-ViralDynamicParameter.P_La5)],
                                   [(1, 2), -(1-ViralDynamicParameter.P_La5)]],
                               6: [[(2, 0), -1]]}
        self.counter = 0

    def _pgs_model(self, t, p, coeff_mat, const_mat, varying_propensities):
        coeff_mat = coeff_mat.repeat(self._shape + [1, 1])
        const_mat = const_mat.repeat(self._shape + [1, 1])
        step = round((t - self._time_span[0]) / self._time_step)
        for idx, propensity_info in varying_propensities:
            propensity = self._pd_interface.get_propensity()
            value = propensity[idx][step]
            for pos, sign in propensity_info:
                coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + value * sign
            if idx == 4:
                const_mat[..., 0, 0] = value * ViralDynamicParameter.P_Ma4
            if idx == 5:
                const_mat[..., 1, 0] = value * ViralDynamicParameter.P_La5
        coeff_mat[..., 2, 0] = coeff_mat[..., 2, 0] * (1 - p[..., 2, 0])
        return torch.matmul(coeff_mat, p) - const_mat

    def compute_pi(self):
        coeff_mat = torch.zeros((3, 3), dtype=torch.double)
        const_mat = torch.zeros((3, 1), dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 7):         # iterate over a1-a6
            if isinstance(propensity_dict[idx], float):
                for pos, sign in self.pos_dict_coeff[idx]:      # iterate over each position in coefficient matrix
                    coeff_mat[pos[0], pos[1]] = coeff_mat[pos[0], pos[1]] + propensity_dict[idx] * sign
                if idx == 4:
                    const_mat[0, 0] = propensity_dict[idx] * ViralDynamicParameter.P_Ma4
                if idx == 5:
                    const_mat[1, 0] = propensity_dict[idx] * ViralDynamicParameter.P_La5
            else:
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat, 'varying_propensities': time_dependent_propensity}
        helpfunc = partial(self._pgs_model, **kwarg)
        p0 = torch.zeros(self._shape + [3, 1], dtype=torch.double)
        return type(self)._ode_solver(helpfunc, self._time_span[1], self._time_span[0], p0, self._time_step)

    def _pgs_model_cumulative(self, t, p, coeff_mat, const_mat):
        """
        Right hand side of basic PGS model in vectorized form (V, T1, T2), used to compute the cumulative probability
        :parameter:
        t: float
        p: ndarray (self._shape, 3*n_strain, 1)
        coeff_mat: ndarray (3*n_strain, 3*n_strain)
        const_mat: ndarray (3*n_strain, 1)
        :return:
        model: ndarray (self._shape, 3*n_strain, 1)
            The right hand side of the PGS ODE
        """
        step = round(t / self._time_step) - 1
        coeff_term = coeff_mat[step].clone()
        const_term = const_mat[step]
        coeff_term[..., 2, 0] = torch.mul(coeff_term[..., 2, 0], (1 - p[..., 2, 0]))
        return torch.matmul(coeff_term, p) - const_term

    def compute_pi_cumulative(self):
        """
        Compute the cumulative infection probability, same way as compute_pe_cumulative
        :return:
        result: ndarray (n_points, n_Sample, [n_regimen, ], 3*n_strain, 1)
        """
        n_points = round(self._time_span[1] - self._time_span[0])
        step_each_iter = self._steps // n_points
        coeff_mat = torch.zeros([step_each_iter, n_points] + self._shape + [3, 3], dtype=torch.double)
        const_mat = torch.zeros([step_each_iter, n_points] + self._shape + [3, 1], dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 7):
            if isinstance(propensity_dict[idx], float):  # add the constant terms into coeff_mat
                for pos, sign in self.pos_dict_coeff[idx]:
                    coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + propensity_dict[idx] * sign
                if idx == 4:  # add the constant terms into const_mat
                    const_mat[..., 0, 0] = propensity_dict[idx] * ViralDynamicParameter.P_Ma4
                if idx == 5:
                    const_mat[..., 1, 0] = propensity_dict[idx] * ViralDynamicParameter.P_La5
            else:  # store the info of time-varying propensity
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        for idx, propensity_info in time_dependent_propensity:  # add the time-varying propensities
            propensity_array = propensity_dict[idx][1:]  # get rid of the additional value at beginning
            for j in range(step_each_iter):
                for pos, sign in propensity_info:  # update coeff_mat with time-varying propensity
                    coeff_mat[j, ..., pos[0], pos[1]] = \
                        coeff_mat[j, ..., pos[0], pos[1]] + propensity_array[j::step_each_iter] * sign
                if idx == 4:  # update const_mat with time-varying propensity
                    const_mat[j, ..., 0, 0] = propensity_array[j::step_each_iter] * ViralDynamicParameter.P_Ma4
                if idx == 5:
                    const_mat[j, ..., 1, 0] = propensity_array[j::step_each_iter] * ViralDynamicParameter.P_La5
        p0 = torch.zeros([n_points] + self._shape + [3, 1], dtype=torch.double)
        result = torch.zeros([n_points] + self._shape + [3, 1], dtype=torch.double)
        for i in range(n_points):
            kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat}
            helpfunc = partial(self._pgs_model_cumulative, **kwarg)
            p0 = type(self)._ode_solver(helpfunc,
                                        step_each_iter * self._time_step,
                                        0, p0, self._time_step)
            result[i] = p0[0, 0]  # update result with the first value of the first point
            p0 = p0[0, 1:]  # remove all values of the first point
            coeff_mat = coeff_mat[:, :-1]  # remove the last group of coeff and const matrix
            const_mat = const_mat[:, :-1]  # since they are already outdated
        return result


class PgSystemMacrophageIncluded(AbstractPgSystem):
    """
    PGS class to compute infection probability for extended viral dynamics (macrophage considered)
    """
    def __init__(self, pd_interface, time_span):
        super().__init__(pd_interface, time_span)
        self.pos_dict_coeff = {1: [[(0, 0), +1]],
                               2: [[(1, 1), +1]],
                               3: [[(2, 2), +1]],
                               4: [[(0, 0), +1],
                                   [(0, 1), -1]],
                               5: [[(1, 1), +1],
                                   [(1, 2), -1]],
                               6: [[(2, 0), -1]],
                               7: [[(1, 1), +1]],
                               8: [[(0, 0), +1]],
                               9: [[(3, 3), +1]],
                               10: [[(4, 4), +1]],
                               11: [[(3, 3), +1],
                                    [(3, 4), -1]],
                               12: [[(4, 0), -1]]}

    def _pgs_model(self, t, p, coeff_mat, const_mat, varying_propensities):
        coeff_mat = coeff_mat.repeat(self._shape + [1, 1])
        const_mat = const_mat.repeat(self._shape + [1, 1])
        step = round((t - self._time_span[0]) / self._time_step)
        for idx, propensity_info in varying_propensities:
            propensity = self._pd_interface.get_propensity()
            value = propensity[idx][step]
            if idx == 7:  # update the const matrix with value of a7
                const_mat[..., 1, 0] = value
            for pos, sign in propensity_info:
                coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + value * sign
        coeff_mat[..., 2, 0] = torch.mul(coeff_mat[..., 2, 0], (1 - p[..., 2, 0]))
        coeff_mat[..., 4, 0] = torch.mul(coeff_mat[..., 4, 0], (1 - p[..., 4, 0]))
        return torch.matmul(coeff_mat, p) - const_mat

    def compute_pi(self):
        coeff_mat = torch.zeros((5, 5), dtype=torch.double)
        const_mat = torch.zeros((5, 1), dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 13):         # iterate over a1-a12
            if isinstance(propensity_dict[idx], float):
                if idx == 7:        # update the const matrix with value of a7
                    const_mat[1, 0] = propensity_dict[idx]
                for pos, sign in self.pos_dict_coeff[idx]:      # iterate over each position in coefficient matrix
                    coeff_mat[pos[0], pos[1]] = coeff_mat[pos[0], pos[1]] + propensity_dict[idx] * sign
            else:
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat, 'varying_propensities': time_dependent_propensity}
        helpfunc = partial(self._pgs_model, **kwarg)
        p0 = torch.zeros(self._shape + [5, 1], dtype=torch.double)
        return type(self)._ode_solver(helpfunc, self._time_span[1], self._time_span[0], p0, self._time_step)

    def _pgs_model_cumulative(self, t, p, coeff_mat, const_mat):
        """
        Right hand side of basic PGS model in vectorized form (V, T1, T2), used to compute the cumulative probability
        :parameter:
        t: float
        p: ndarray (self._shape, 3*n_strain, 1)
        coeff_mat: ndarray (3*n_strain, 3*n_strain)
        const_mat: ndarray (3*n_strain, 1)
        :return:
        model: ndarray (self._shape, 5*n_strain, 1)
            The right hand side of the PGS ODE
        """
        step = round(t / self._time_step) - 1
        coeff_term = coeff_mat[step].clone()
        const_term = const_mat[step]
        coeff_term[..., 2, 0] = torch.mul(coeff_term[..., 2, 0], (1 - p[..., 2, 0]))
        coeff_term[..., 4, 0] = torch.mul(coeff_term[..., 4, 0], (1 - p[..., 4, 0]))
        return torch.matmul(coeff_term, p) - const_term

    def compute_pi_cumulative(self):
        """
        Compute the cumulative infection probability, same way as compute_pe_cumulative
        :return:
        result: ndarray (n_points, n_Sample, [n_regimen, ], 3*n_strain, 1)
        """
        n_points = round(self._time_span[1] - self._time_span[0])
        step_each_iter = self._steps // n_points
        coeff_mat = torch.zeros([step_each_iter, n_points] + self._shape + [5, 5], dtype=torch.double)
        const_mat = torch.zeros([step_each_iter, n_points] + self._shape + [5, 1], dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 13):
            if isinstance(propensity_dict[idx], float):  # add the constant terms into coeff_mat
                if idx == 7:  # update the const matrix with value of a7
                    const_mat[..., 1, 0] = propensity_dict[idx]
                for pos, sign in self.pos_dict_coeff[idx]:
                    coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + propensity_dict[idx] * sign
            else:  # store the info of time-varying propensity
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        for idx, propensity_info in time_dependent_propensity:  # add the time-varying propensities
            propensity_array = propensity_dict[idx][1:]  # get rid of the additional value at beginning
            for j in range(step_each_iter):
                if idx == 7:  # update const_mat with time-varying propensity
                    const_mat[j, ..., 1, 0] = propensity_array[j::step_each_iter]
                for pos, sign in propensity_info:  # update coeff_mat with time-varying propensity
                    coeff_mat[j, ..., pos[0], pos[1]] = \
                        coeff_mat[j, ..., pos[0], pos[1]] + propensity_array[j::step_each_iter] * sign
        p0 = torch.zeros([n_points] + self._shape + [5, 1], dtype=torch.double)
        result = torch.zeros([n_points] + self._shape + [5, 1], dtype=torch.double)
        for i in range(n_points):
            kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat}
            helpfunc = partial(self._pgs_model_cumulative, **kwarg)
            p0 = type(self)._ode_solver(helpfunc,
                                        step_each_iter * self._time_step,
                                        0, p0, self._time_step)
            result[i] = p0[0, 0]  # update result with the first value of the first point
            p0 = p0[0, 1:]  # remove all values of the first point
            coeff_mat = coeff_mat[:, :-1]  # remove the last group of coeff and const matrix
            const_mat = const_mat[:, :-1]  # since they are already outdated
        return result


class PgSystemReservoirNewApproach(AbstractPgSystem):
    """
    PGS class to compute probability of reservoir establishment
    """
    def __init__(self, pd_interface, time_span):
        super().__init__(pd_interface, time_span)
        # position of propensity used to compute cumulative reservoir probability (coefficient matrix)
        self.pos_dict_coeff = {1: [[(0, 0), +1]],
                               2: [[(1, 1), +1]],
                               3: [[(2, 2), +1]],
                               4: [[(0, 0), +1],
                                   [(0, 1), -1]],
                               5: [[(1, 1), +1],
                                   [(1, 2), -1]],
                               6: [[(2, 0), -1]],
                               7: [[(1, 1), +1]]}
        # position of propensity used to compute cumulative probability distribution of reservoir (coefficient matrix)
        self.pos_dict_dist_coeff = {1: [[(0, 0), +1]],
                                    2: [[(1, 1), +1]],
                                    3: [[(2, 2), +1]],
                                    4: [[(0, 0), +1],
                                        [(0, 1), -1]],
                                    5: [[(1, 1), +1],
                                        [(1, 2), -1]],
                                    6: [[(2, 2), +1]],
                                    7: [[(1, 1), +1]]}
        # position of propensity used to compute cumulative probability distribution of reservoir (constant matrix)
        self.pos_dict_dist_const = {1: [[(0, 0), +1]],
                                    2: [[(1, 0), +1]],
                                    3: [[(2, 0), +1]],
                                    7: [[(1, 1), +1]]}

    def _pgs_model(self, t, p, coeff_mat, const_mat, varying_propensities):
        raise NotImplementedError

    def _pgs_model_reservoir_cumulative(self, t, p, coeff_mat, const_mat):
        """
        Right hand side of basic PGS model in vectorized form (V, T1, T2), used to compute the cumulative probability
        :parameter:
        t: float
        p: ndarray (self._shape, 3*n_strain, 1)
        coeff_mat: ndarray (3*n_strain, 3*n_strain)
        const_mat: ndarray (3*n_strain, 1)
        :return:
        model: ndarray (self._shape, 5*n_strain, 1)
            The right hand side of the PGS ODE
        """
        step = round(t / self._time_step) - 1
        # expand the coefficient matrix to match the dimension of p0
        coeff_term = coeff_mat[step].clone().repeat([p.shape[0]] + [1] * (len(self._shape) + 2))
        const_term = const_mat[step].clone()
        coeff_term[..., 2, 0] = torch.mul(coeff_term[..., 2, 0], (1 - p[..., 2, 0]))
        return torch.matmul(coeff_term, p) - const_term

    def compute_pr_cumulative(self, expo_tps=[]):
        """
        Compute the cumulative probability of reservoir using new approach (adding different initial point gradually)
        :parameter:
        expo_tps: list of exposure time points
            (have to record the cumulative probability for the corresponding exposure time)
        :return:
        result: ndarray (n_points, n_Sample, [n_regimen, ], 3, 1)
        """
        n_points = round(self._time_span[1] - self._time_span[0])
        # 1 indicate the number of current p0 point, this dimension is for future dimension expansion
        coeff_mat = torch.zeros([self._steps] + [1] + self._shape + [3, 3], dtype=torch.double)
        const_mat = torch.zeros([self._steps] + [1] + self._shape + [3, 1], dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        for idx in range(1, 8):
            if isinstance(propensity_dict[idx], float):  # add the constant terms into coeff_mat
                if idx == 7:  # update the const matrix with value of a7
                    const_mat[..., 1, 0] = propensity_dict[idx]
                for pos, sign in self.pos_dict_coeff[idx]:
                    coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + propensity_dict[idx] * sign
            else:  # store the info of time-varying propensity
                time_dependent_propensity.append([idx, self.pos_dict_coeff[idx]])
        for idx, propensity_info in time_dependent_propensity:  # add the time-varying propensities
            propensity_array = propensity_dict[idx][1:]    # get rid of the additional value at beginning
            if idx == 7:  # update const_mat with time-varying propensity
                const_mat[:, 0, ..., 1, 0] = propensity_array
            for pos, sign in propensity_info:  # update coeff_mat with time-varying propensity
                coeff_mat[:, 0, ..., pos[0], pos[1]] = coeff_mat[:, 0, ..., pos[0], pos[1]] + propensity_array * sign
        p0 = torch.zeros([1] + self._shape + [3, 1], dtype=torch.double)
        const_mat_original = const_mat.clone()
        res_expo = dict()       # for the result of different exposure time in expo_tps
        if expo_tps:
            expo_tps = np.array(expo_tps) - self._time_span[0]
        for i in range(n_points)[::-1]:
            kwarg = {'coeff_mat': coeff_mat, 'const_mat': const_mat}
            helpfunc = partial(self._pgs_model_reservoir_cumulative, **kwarg)
            p0 = type(self)._ode_solver(helpfunc, i+1, i, p0, self._time_step)
            # if one time point in expo_tps reached
            p0_new = torch.zeros([1] + self._shape + [3, 1], dtype=torch.double)
            p0 = torch.cat([p0[0], p0_new], 0)    # concatenate a new p0 [0,0,0]
            if i in expo_tps:
                # print(p0.shape)
                res_expo[i+self._time_span[0]] = torch.flip(p0, dims=[0])
            const_mat = torch.cat([const_mat, const_mat_original], 1)  # update const matrix, adjust to dim of p0
        return torch.flip(p0, dims=[0]), res_expo

    def _pgs_model_reservoir_distribution(self, t, p, n_reservoir, coeff_mat, const_mat, a6_array):
        """
        Right hand side of basic PGS model in vectorized form (3xn), used to compute the cumulative probability
        distribution of reservoir number
        :parameter:
        t: float
        p: ndarray (self._shape, 3*n_reservoir, 1)
        n_reservoir: number of reservoir (one dimension in p and const_mat)
        coeff_mat: ndarray (3*n_strain, 3*n_strain)
        const_mat: ndarray (3*n_strain, 1)
        :return:
        model: ndarray (self._shape, 3, n_reervoir)
            The right hand side of the PGS ODE
        """
        step = round(t / self._time_step) - 1
        if isinstance(a6_array, float):     # take the value of a6 for this step
            a6 = a6_array
        else:
            a6 = a6_array[step]
        # take the coeff and const mat of this step and expand the coefficient matrix to match the dimension of p0
        coeff_term = coeff_mat[step].repeat([p.shape[0]] + [1] * (len(self._shape) + 2))
        const_term = const_mat[step]
        # 1. step: tmp_p (3xn) = coeff_mat (3x3) * p (3xn)
        tmp_p = torch.matmul(coeff_term, p)
        # 2. step: tmp_p[2, :] = tmp_p[2, :] - a6 * p[2, :] * [p[0, :]] (diagonal matrix of P[0, :])
        # take the transpose of reversed p0 (for matrix multiplication later)
        p0_array = torch.transpose(torch.flip(p[..., [0], :], dims=[-1]), -2, -1)

        for i in range(n_reservoir):
            tmp_p[..., 2, i] = tmp_p[..., 2, i] - \
                               a6 * (p[..., [2], :i+1] @ p0_array[..., n_reservoir-i-1:, [0]])[..., 0, 0]
        # 3. step: tmp_p - const_mat
        return tmp_p - const_term

    def compute_cumulative_reservoir_distribution(self, n_reservoir, expo_tps=[]):
        """
        Compute the cumulative infection probability, same way as compute_pe_cumulative
        :parameter:
        n_reservoir: int
            upper bound of reservoir number. The distribution for range(0, n) reservoir will be computed.
        expo_tps: list of exposure time points
            (have to record the cumulative probability for the corresponding exposure time)
        :return:
        result: ndarray (n_points, n_Sample, [n_regimen, ], 3*n_strain, 1)
        """
        n_points = round(self._time_span[1] - self._time_span[0])
        # 1 indicate the number of current p0 point, this dimension is for future dimension expansion
        coeff_mat = torch.zeros([self._steps] + [1] + self._shape + [3, 3], dtype=torch.double)
        const_mat = torch.zeros([self._steps] + [1] + self._shape + [3, n_reservoir], dtype=torch.double)
        time_dependent_propensity = []
        propensity_dict = self._pd_interface.get_propensity()
        a6_array = propensity_dict[6]
        for idx in range(1, 8):
            if idx in self.pos_dict_dist_const:        # fill propensities into const_mat
                for pos, sign in self.pos_dict_dist_const[idx]:
                    if isinstance(propensity_dict[idx], float):     # add the constant terms into const_mat
                        const_mat[..., pos[0], pos[1]] = const_mat[..., pos[0], pos[1]] + propensity_dict[idx]*sign
                    else:      # update with time_varing propensities (get rid of the additional value at beginning)
                        const_mat[:, 0, ..., pos[0], pos[1]] = const_mat[:, 0, ..., pos[0],
                                                                         pos[1]] + propensity_dict[idx][1:] * sign
            for pos, sign in self.pos_dict_dist_coeff[idx]:     # fill propensities into coeff_mat
                if isinstance(propensity_dict[idx], float):  # add the constant terms into coeff_mat
                    coeff_mat[..., pos[0], pos[1]] = coeff_mat[..., pos[0], pos[1]] + propensity_dict[idx] * sign
                else:
                    coeff_mat[:, 0, ..., pos[0], pos[1]] = coeff_mat[:, 0, ..., pos[0],
                                                                     pos[1]] + propensity_dict[idx][1:] * sign
        # the extra 1 first dimension will be used for expansion later
        p0 = torch.zeros([1] + self._shape + [3, n_reservoir], dtype=torch.double)
        p0[..., :, 0] = 1
        p0_new = p0.clone()          # generate a new p0 template for extension
        const_mat_original = const_mat.clone()
        res_expo = dict()       # for the result of different exposure time in expo_tps
        if expo_tps:
            expo_tps = np.array(expo_tps) - self._time_span[0]
        for i in range(n_points)[::-1]:
            kwarg = {'n_reservoir': n_reservoir, 'coeff_mat': coeff_mat, 'const_mat': const_mat, 'a6_array': a6_array}
            helpfunc = partial(self._pgs_model_reservoir_distribution, **kwarg)
            p0 = type(self)._ode_solver(helpfunc, i+1, i, p0, self._time_step)
            # if one time point in expo_tps reached
            p0 = torch.cat([p0[0], p0_new.clone()], 0)    # concatenate a new p0 [0,0,0]
            if i in expo_tps:
                res_expo[i+self._time_span[0]] = torch.flip(p0, dims=[0])
            const_mat = torch.cat([const_mat, const_mat_original], 1)  # update const matrix, adjust to dim of p0
        return torch.flip(p0, dims=[0]), res_expo

        """kwarg = {'n_reservoir': n_reservoir, 'coeff_mat': coeff_mat, 'const_mat': const_mat, 'a6_array': a6_array}
        helpfunc = partial(self._pgs_model_reservoir_distribution, **kwarg)
        return type(self)._ode_solver(helpfunc, self._time_span[1], self._time_span[0], p0, self._time_step), res_expo"""
