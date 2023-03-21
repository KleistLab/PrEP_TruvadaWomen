from os.path import exists
from pathlib import Path
import pickle

from .ode_solver import euler
from .pd import PharmacoDynamicsInterface
from .pgs import AbstractPgSystem, PgSystemExtinction, PgSystemInfection, PgSystemMacrophageIncluded, \
    PgSystemReservoirNewApproach
from .pk import AbstractPharmacokinetics
from .utils import ViralDynamicParameter, ExtinctionCalculator

# TODO: rearrange different regimens
# TODO: bug: time span of PK and PGS are different if input timespan is not integer times of the periods


class EfficacyPredictor(object):
    """
    Class to estimate the prophylactic efficacy of a regimen /regimens

    Attributes:
    -------------
    _time_step_pk: dict[drug] = float
        step size [hr] for solving PK / PGS ODE . None denotes default (0.02), can be changed by setter methods.
        if _time_Step_pk is smaller than _time_step_pgs: _time_step_pgs / _time_step_pk should be an integer.
    _ode_solver_pk: dict[drug] = callable
        function to solve ODE for PK / PGS. None denotes default (euler), can be changed be setter.
    _time_step_pk: float
        step size [hr] for solving PK / PGS ODE . None denotes default (0.02), can be changed by setter methods.
        if _time_Step_pk is smaller than _time_step_pgs: _time_step_pgs / _time_step_pk should be an integer.
    _ode_solver_pk: callable
        function to solve ODE for PK / PGS. None denotes default (euler), can be changed be setter.
    _time_span: tuple
        time span [hr] of computation (the time span of all regimens should be equal, otherwise
        the regimens must be rearranged)
    regimens: array-like
         list of Regimen objects,
    files: array-like
        list of file names which contains the PK parameters, must be in corresponding order as regimens
    _pk_objects: array-like
        list of Pharmacokinetics objects, corresponds to regimens
    _pd_object: PharmacodynamicsBoundary object
        PharmacodynamicsBoundary object generated from _pk_objects
    _pgs_object: PgSystem object
        PgSystem object generated from pd_object
    _pd_class:  PharmacodynamicsBoundary_
        the Pharmacodynamics class used (viral dynamics dependent)
    _pgs_class: PgSystem_
        the pgs class used (viral dynamics dependent)
    _pe_function_non_medicated: callable
        function to calculate the extinction probability with absence of any drug
    _pe: ndarray (n_steps, [, n_regimens], n_samples, 3, 1)
        matrix of extinction probabilities
    _pi: ndarray (n_steps, [, n_regimens], n_samples, 3, 1)
        matrix of infection probabilities
    _phi: ndarray (n_steps, [, n_Regimens], n_samples, 1, 1)
        matrix of prophylactic efficacy
    _vd_parameter_file: str
        name of the file of viral dynamic parameters, file must be placed in ./config/
    _cdf_pe: ndarray (n_points, [, n_regimens], n_samples, 3, 1)
        matrix of cumulative extinction probability with a fixed exposure time
    _cdf_pi: ndarray (n_points, [, n_regimens], n_samples, 3, 1)
        matrix of cumulative infection probability with a fixed exposure time
    _cdf_pr: ndarray (n_points, [, n_regimens], n_samples, 3, 1)
        matrix of cumulative probability of reservoir establishment with a fixed exposure time
    """
    def __init__(self):
        self._time_step_pk = dict()
        self._ode_solver_pk = dict()
        self._time_step_pgs = 0.02
        self._ode_solver_pgs = euler
        self._time_span = None
        self.regimens = []
        self.files = []
        self._pk_objects = []
        self._pd_interface = None
        self._pgs_object = None
        self._pgs_class = PgSystemExtinction                     # class of pgs
        self._pe_function_non_medicated = ExtinctionCalculator.get_pe_basic
        self._pe = None
        self._pi = None
        self._phi = None
        self._cdf_pe = None
        self._cdf_pi = None
        self._cdf_pr = None
        self._pr_distribution = None
        self._vd_parameter_file = 'config.ini'
        ViralDynamicParameter.set_vd_parameters(file=self._vd_parameter_file)

    @classmethod
    def save_object(cls, obj, filename):
        """
        save the object in a local file

        Parameters:
        filename: str
            name of the file in which the object will be saved
        """
        if not isinstance(obj, cls):
            raise SystemExit('Object is not an EfficacyPredictor object.\n')
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
            print('Object saved in {}'.format(filename))

    @classmethod
    def load_object(cls, filename):
        """
        load objects from a file as a generator

        Parameters:
        filename: str
            name of the file in which the object is saved
        """
        if not exists(filename):
            raise SystemExit('File does not exist, please check and try again.\n')
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    def _reset(self):
        """
        reset the object variables
        """
        self._pk_objects = []
        self._pd_interface = None
        self._pgs_object = None
        # self._pe = None
        # self._pi = None
        # self._phi = None

    def _set_pgs_class(self, pgs_class):
        """
        set PGS class used, must match self._pd_class
        """
        assert issubclass(pgs_class, AbstractPgSystem), 'PGS class must be a subclass of AbstractPgSystem'
        self._pgs_class = pgs_class
        print('PGS class changed')

    def set_pk_time_step(self, drug, timestep):
        """
        set time step for PK object of 'drug', must be called after corresponding regimen is given.
        """
        self._time_step_pk[drug] = timestep
        self._reset()
        print('Time step for PK changed, please run the computation again.')

    def set_pk_ode_solver(self, drug, ode_solver):
        """
        set ODE solver for PK object of 'drug, must be called after corresponding regimen is given.
        """
        self._ode_solver_pk[drug] = ode_solver
        self._reset()
        print('ODE solver for PK changed, please run the computation again. ')

    def set_pgs_time_step(self, timestep):
        """
        set time step for PGS.
        """
        self._time_step_pgs = timestep
        AbstractPgSystem.set_time_step(self._time_step_pgs)
        self._reset()
        print('Time step for PGS changed, please run the computation again.')

    def set_pgs_ode_solver(self, ode_solver):
        """
        set ODE solver for PK.
        """
        self._ode_solver_pgs = ode_solver
        AbstractPgSystem.set_ode_solver(self._ode_solver_pgs)
        self._reset()
        print('ODE solver for PK changed, please run the computation again. ')

    def set_pe_function(self, pe_function):
        """
        set the function to calculate the extinction probability with absence of drug (match self._pd_class)
        """
        self._pe_function_non_medicated = pe_function
        self._reset()

    def change_vd_file(self, file_name):
        """
        set the file that contains the viral dynamic parameters
        """
        file = Path('./config/' + file_name)
        assert file.is_file(), 'File does not exit'
        self._reset()
        self._vd_parameter_file = file_name
        ViralDynamicParameter.set_vd_parameters(file=self._vd_parameter_file)
        print('Viral dynamic parameters changed, please run the computation again. ')

    def add_regimen(self, regimen):
        """
        add new regimen, one regimen for one drug, so that the efficacy of drugs combination can be computed.
        If the time span of regimen is not the same as self._time_span, rearrange
        the self.regimens and then set self._time_span
        """
        self.regimens.append(regimen)
        if not self._time_span and self._time_span != regimen.get_timespan():
            self._rearrange_regimens()
        self._time_span = regimen.get_timespan()
        self._reset()
        print('A new regimen added. please run the computation again. ')

    def add_sample_files(self, file):
        """
        add the file containing PK-parameters of patients for corresponding drug.
        The order must much the order of self.regimens.
        If default PK parameters are used for all regimens, it's no need to input None explicitly; otherwise
        None must hold the corresponding place (between two input files).
        """
        self.files.append(file)
        self._reset()
        print('Input file changed, please run the computation again. ')

    def _rearrange_regimens(self):
        """
        rearrange the regimens of self.regimens if their period / time span are not equal.
        """
        if not self.regimens:
            raise SystemExit('No regimen available, please add at least one regimen for computation\n')
        elif len(self.regimens) > 1:
            # TODO: rescale the regimens if their period/time span are not equal. If it's a specific combination
            # like Truvada, the adherence pattern of each drug should be same, and time span should be same long
            pass

    def compute_concentration(self):
        """
        compute the PK profile: corresponding PK objects are generated and assign to self._pk_objects
        """
        self._reset()
        if len(self.files) < len(self.regimens):  # file list shorter than regimen list, make up with None
            self.files = self.files + [None] * (len(self.regimens) - len(self.files))
        elif len(self.files) > len(self.regimens):
            raise SystemExit('Given input files are more than type of drugs, please check the input.\n')
        ratio = 1
        for regimen, file in zip(self.regimens, self.files):
            if regimen.get_drug_name() in self._ode_solver_pk and regimen.get_drug_name() in self._time_step_pk:
                if self._time_step_pk[regimen.get_drug_name()] < self._time_step_pgs:
                    ratio = round(self._time_step_pgs / self._time_step_pk[regimen.get_drug_name()])
                self._pk_objects.append(AbstractPharmacokinetics.get_pk_class(regimen,
                                                                              file, ratio,
                                                                              self._ode_solver_pk[regimen.get_drug_name()],
                                                                              self._time_step_pk[regimen.get_drug_name()]))
            elif regimen.get_drug_name() in self._ode_solver_pk:
                self._pk_objects.append(AbstractPharmacokinetics.get_pk_class(regimen,
                                                                              file, ratio,
                                                                              self._ode_solver_pk[regimen.get_drug_name()]))
            elif regimen.get_drug_name() in self._time_step_pk:
                if self._time_step_pk[regimen.get_drug_name()] < self._time_step_pgs:
                    ratio = round(self._time_step_pgs / self._time_step_pk[regimen.get_drug_name()])
                self._pk_objects.append(AbstractPharmacokinetics.get_pk_class(regimen,
                                                                              file, ratio, time_step=
                                                                              self._time_step_pk[regimen.get_drug_name()]))
            else:
                self._pk_objects.append(AbstractPharmacokinetics.get_pk_class(regimen, file, ratio))

    def compute_drug_effect(self, file=None, reservoir=False, macrophage=False):
        """
        compute the drug effect (eta) profile: self.strain_manager will be initialized (pd object generated)
        :parameter:
        file: str
            file name of possible file for PD computation. Currently only used for Truvada: file of MMOA matrix
        """
        if not self._pd_interface:
            if not self._pk_objects:
                self.compute_concentration()
            self._pd_interface = PharmacoDynamicsInterface(self._pk_objects, file, reservoir, macrophage)

    def compute_extinction_probability(self, reservoir=False):
        """
        Compute the extinction probability profile and assign the result to self._pe
        ATTENTION: here the reservoir indicates if latent and long-lived cells are considered as infection during
            computation of extinction
        """
        self.compute_drug_effect()
        if self._pgs_class is not PgSystemExtinction:
            self._set_pgs_class(PgSystemExtinction)
        self._pgs_object = self._pgs_class(self._pd_interface, self._time_span, reservoir=reservoir)
        self._pe = self._pgs_object.compute_pe()

    def compute_efficacy(self, reservoir=False):
        """
        Compute the prophylactic efficacy and assign the result to self._phi
        ATTENTION: here the reservoir indicates if latent and long-lived cells are considered as infection during
            computation of extinction
        """
        if self._pe is None:        # extinction probability not computed
            self.compute_extinction_probability(reservoir=reservoir)
        pe_no_drug = self._pe_function_non_medicated()
        self._phi = 1 - (1 - self._pe[..., 0, 0]) / (1 - pe_no_drug)

    def compute_cumulative_extinction_probability(self, reservoir=False):
        """
        Compute the cumulative probability and assign the result to self._cdf_pe
        ATTENTION: here the reservoir indicates if latent and long-lived cells are considered as infection during
            computation of extinction
        """
        if isinstance(self._pgs_object, PgSystemExtinction):         # extinction probability already computed
            result = self._pgs_object.compute_pe_cumulative()
        else:
            self.compute_drug_effect()
            self._pgs_object = self._pgs_class(self._pd_interface, self._time_span, reservoir=reservoir)
            result = self._pgs_object.compute_pe_cumulative()
        self._cdf_pe = result

    def compute_infection_probability(self, macrophage=False):
        """
        Compute the infection probability profile and assign the result to self._pi
        :parameter:
        macrophage: boolean, if the macrophage is considered in the viral dynamics
        (currently only available for infection probability)
        """

        self.compute_drug_effect(macrophage=macrophage)
        if macrophage:
            self._set_pgs_class(PgSystemMacrophageIncluded)
        elif self._pgs_class is not PgSystemInfection:
            self._set_pgs_class(PgSystemInfection)
        self._pgs_object = self._pgs_class(self._pd_interface, self._time_span)
        self._pi = self._pgs_object.compute_pi()

    def compute_cumulative_infection_probability(self, macrophage=False):
        """
        Compute the cumulative probability and assign the result to self._cdf_pe
        """
        self.compute_drug_effect(macrophage=macrophage)
        if macrophage:
            self._set_pgs_class(PgSystemMacrophageIncluded)
        elif self._pgs_class is not PgSystemInfection:
            self._set_pgs_class(PgSystemInfection)
        self._pgs_object = self._pgs_class(self._pd_interface, self._time_span)
        self._cdf_pi = self._pgs_object.compute_pi_cumulative()

    def compute_cumulative_reservoir_probability(self, expo_tps=[]):
        """
        compute the cumulative probability of reservoir establishment when exposure occurs at self._time_span[0].
        Currently it returns same result as compute_cumulative_infection_probability.
        This function is implemented using the new approach in PGS class PgSystemReservoirNewApproach.
        :parameter
        expo_tps: list
            time points for which we want to observe as exposure time. If given, the cumulative probability for
            different exposure time will be returned as a dictionary
        """
        # ATTENTION: here reservoir is used so that the new approach can be called
        # to compute the probability of reservoir (propensity a1-a7 calculated)
        self.compute_drug_effect(reservoir=True)
        if self._pgs_class is not PgSystemReservoirNewApproach:
            self._set_pgs_class(PgSystemReservoirNewApproach)
        self._pgs_object = self._pgs_class(self._pd_interface, self._time_span)
        self._cdf_pr, cdf_expo = self._pgs_object.compute_pr_cumulative(expo_tps)
        return cdf_expo

    def compute_cumulative_reservoir_distribution(self, n_reservoir, expo_tps=[]):
        """
        compute the cumulative probability distribution of reservoir establishment when exposure occurs at
        self._time_span[0].
        This function is implemented using the new approach in PGS class PgSystemReservoirNewApproach.
        :parameter
        n_reservoir: int
            upper bound of reservoir number. The distribution for range(0, n) reservoir will be computed.
        expo_tps: list
            time points for which we want to observe as exposure time. If given, the cumulative probability for
            different exposure time will be returned as a dictionary
        """
        # ATTENTION: here reservoir is used so that the new approach can be called
        # to compute the probability of reservoir (propensity a1-a7 calculated)
        self.compute_drug_effect(reservoir=True)
        if self._pgs_class is not PgSystemReservoirNewApproach:
            self._set_pgs_class(PgSystemReservoirNewApproach)
        self._pgs_object = self._pgs_class(self._pd_interface, self._time_span)
        self._pr_distribution, cdf_expo = self._pgs_object.compute_cumulative_reservoir_distribution(n_reservoir,
                                                                                                     expo_tps)
        return cdf_expo

    def set_concentration_proportion(self, drug, proportion):
        """
        Adjust the target PK profile with a parameter proportion
        Example: TDF and FTC have different local concentration compared to concentration PBMC (target in PK system)
        to compute the local concentration of different organ, this function can be used.
        !! This function must be called after compute_concentration and before subsequent computation
        !! This function only change the pk._concentration_profile_target,
        !! but pk._concentration_profile_whole stays unchanged.
        :parameter:
        drug: str, name of the drug
        proportion: float, the proportion that the target concentration will change
        """
        if not self._pk_objects:
            self.compute_concentration()
        for pk in self._pk_objects:
            if pk.regimen.get_drug_name() == drug:
                pk.set_concentration(proportion)

    def set_pd_file(self, file, reservoir=False, macrophage=False):
        """
        Change the setup file in pd (currently only for Truvada MMOA file) and initialize the pd interface object.
        """
        print("PD file changed")
        self.compute_drug_effect(file, reservoir, macrophage)

    def get_concentration(self, drug):
        for pk in self._pk_objects:
            if pk.regimen.get_drug_name() == drug:
                return pk.get_concentration_whole()

    def get_drug_effect(self):
        if not self._pd_interface:
            self.compute_drug_effect()
        if len(self._pd_interface.get_pd_objects()) == 1:
            return self._pd_interface.get_pd_objects()[0].get_drug_effect(), self._pd_interface.get_propensity()
        else:
            # drug = input('Desired drug:')
            pass
            # TODO: return the drug effect profile for the given drug

    def get_extinction_probability(self):
        return self._pe

    def get_efficacy(self):
        return self._phi

    def get_cumulative_extinction_probability(self):
        return self._cdf_pe

    def get_infection_probability(self):
        return self._pi

    def get_cumulative_infection_probability(self):
        return self._cdf_pi

    def get_cumulative_reservoir_probability(self):
        return self._cdf_pr

    def get_reservoir_probability_distribution(self):
        return self._pr_distribution
