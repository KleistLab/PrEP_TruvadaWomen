import numpy as np
import torch

from .utils import Drug


class RegimenGenerator(object):
    # TODO: a class to generate/rescale corresponding regimens
    pass


class Regimen(object):
    """
    Basic class of regimen (for arbitrary drug and dosing schemes with a given adherence level). The function of
    this class can be extended for more complicated situations.
    Parameters:
        drug: str, name of the drug (abbreviation)
        dose: mass of one dose [mg]
        period: the time between two doses [hr]
        time_span: tuple, time range of interest [hr] (both sides should better be the multiple of _period)

    Attributes:
        _drug: Drug (enum)
            enum member of Drug
        _dose: float
            mass of one dose [mg] (constant dose during the regimen)
        _number_doses: int
            count of doses in one regimen (i.e. times of drug taken at 100% adherence)
        _period: float
            time between two doses [hr] (should be constant during one regimen, better be integer)
        _time_span: tuple
            time range of interest [hr], _time_span[0] can be negative (time point before the first dose)
        _administration: str
            mode of administration, oral or implant. Default: oral.
        _adh_level: float
            value of adherence (0-1)
        _adh_pattern: 2-D array
            a 2-D arrray of 0 and 1 for drug taking, beginning from the first dose (time point 0). If pattern is given,
            dose, n_doses and adh_level will be ignored.
        _regimen_matrix: 2Darray (n_regimen, n_period)
            matrix of doses, generated according to _adh_level. 0 will be attached one/both ends if
            _time_span exceeds _number_doses*_period (first dose beginning from time point 0).
            This matrix is used by PK object.
    """
    def __init__(self, drug, period, time_span, dose, n_doses, adh_level, administration='oral', adh_pattern=None):
        self._drug = Drug[drug]
        self._dose = dose
        self._number_doses = n_doses
        self._adh_level = adh_level
        self._period = period
        self._time_span = time_span
        self._administration = administration
        self._adh_pattern = adh_pattern
        self._regimen_matrix = self.generate_regimen_matrix()

    def generate_regimen_matrix(self):
        """
        generate the matrix of drug taking pattern, with 0 for no taking and self._dose for drug taking.
        :return:
        pattern: ndarray (, n_periods)
        """
        if self._adh_pattern is not None:
            d1, d2 = np.array(self._adh_pattern).shape       # length of each dimension of the given adherence pattern
        else:
            d1, d2 = 1, 0                           # one regimen, d1 = 1, d2 not important (unknown actually)
        pre_array = np.array([[] for _ in range(d1)])
        post_array = np.array([[] for _ in range(d1)])
        if self._time_span[0] < 0:
            pre_array = np.zeros((d1, abs(self._time_span[0] // self._period)))
        if self._adh_pattern is not None:
            pattern = np.array(self._adh_pattern) * self._dose
            if self._time_span[1] > d2 * self._period:
                post_array = np.zeros((d1, int(self._time_span[1] / self._period - d2)))
            elif self._time_span[1] < d2 * self._period:
                pattern = pattern[:, :int(self._time_span[1] / self._period)]
        else:
            pattern = np.random.choice([0, self._dose], size=self._number_doses,
                                       p=[1 - self._adh_level, self._adh_level])
            pattern = np.expand_dims(pattern, axis=0)
            if self._time_span[1] > self._number_doses * self._period:
                post_array = np.zeros((d1, int(self._time_span[1] / self._period - self._number_doses)))
        pattern = np.concatenate((pre_array, pattern), axis=1)
        pattern = np.concatenate((pattern, post_array), axis=1)
        return torch.tensor(pattern)

    def decode_regimen_matrix(self):
        raise NotImplementedError

    def get_regimen_matrix(self):
        return self._regimen_matrix

    def get_period(self):
        return self._period

    def get_timespan(self):
        return self._time_span

    def get_number_doses(self):
        return self._number_doses

    def get_administration(self):
        return self._administration

    def get_dose(self):
        return self._dose

    def get_drug(self):
        return self._drug

    def get_drug_name(self):
        return self._drug.name

    def get_drug_class(self):
        return self._drug.drug_class

    def get_molecular_weight(self):
        return self._drug.molecular_weight

    def get_hill_coefficient(self):
        return self._drug.m

    def get_ic50(self):
        return self._drug.IC_50
