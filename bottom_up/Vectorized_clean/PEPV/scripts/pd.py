import os

import pandas as pd
import torch
from scipy.interpolate import griddata

from .utils import DrugClass, ViralDynamicParameter

ViralDynamicParameter.set_vd_parameters()


class PharmacoDynamicsInterface(object):
    """
    Class to compute the reaction propensities
    Parameters:
    pk_objects: array-like
        an array contains all PK objects (corresponding to each drug)
    reservoir: boolean
        if reservoir is considered. If the case, add a7 and reduce a5 respectively
    macrophage: boolean
        if macrophage and latent is considered. If the case, add a7-a12 and modify a1 a5.
    file: str
        file name of possible file for PD computation. Currently only used for Truvada: file of MMOA matrix
    """
    drugCombination = {'FTC': {'TDF': 'Truvada'}}

    def __init__(self, pk_objects, file=None, reservoir=False, macrophage=False):
        self._pk_objects = pk_objects
        self._file = file
        self._reservoir = reservoir
        self._macrophage = macrophage
        self._pd_objects = self._map_pk_to_pd()
        self._propensity_dict = self._combine_pd()

    @staticmethod
    def check_drug_combination(drug_names):
        """
        Check if a certain combination exists for two (or more) drugs (e.g. Truvada)

        :parameter:
        drug_names: array-like
            list of str of drug names
        :return:
        name: str
            name of drug combination (None if no such combination exists)
        """
        # TODO: return the name of drug combination or None if no combination available (using self.drugCombination)
        drug_names.sort()
        if drug_names[0] in PharmacoDynamicsInterface.drugCombination:
            combi = PharmacoDynamicsInterface.drugCombination[drug_names[0]]
            for drug in drug_names[1:]:
                if drug in combi:
                    combi = combi[drug]
                else:
                    return None
            return combi
        else:
            return None

    def _get_pd_class(self, pk_objects):
        """
        get the corresponding PD object for a drug/ drug combination represented by _pk_objects

        :parameter:
        _pk_objects: array-like
            array of pk objects
        :return:
        Pharmacodynamics_ : Pharmacodynamics object
            corresponding PD object
        """

        if len(pk_objects) > 1:
            name = PharmacoDynamicsInterface.check_drug_combination([pk.regimen.get_drug_name() for
                                                                     pk in pk_objects])
            if not name:
                raise SystemExit('Data for this drug combination does not exist\n')
            else:
                # TODO call the corresponding PD for combination
                if name == 'Truvada':
                    return PharmacodynamicsTruvada(pk_objects, self._file, self._reservoir, self._macrophage)
        else:
            if pk_objects[0].regimen.get_drug_class() is DrugClass.CRA:
                return PharmacodynamicsCRA(pk_objects, self._reservoir, self._macrophage)
            elif pk_objects[0].regimen.get_drug_class() is DrugClass.InI:
                return PharmacodynamicsInI(pk_objects, self._reservoir, self._macrophage)
            elif pk_objects[0].regimen.get_drug_class() is DrugClass.PI:
                return PharmacodynamicsPI(pk_objects, self._reservoir, self._macrophage)
            else:
                return PharmacodynamicsRTI(pk_objects, self._reservoir, self._macrophage)

    def _map_pk_to_pd(self):
        """
        generate a list which contains all PD objects corresponds to self._pk_objects.
        (if drug combination exists, generate one corresponding PD object)

        :return:
        pd_list: array-like
            list of pd objects
        """
        pd_list = list()
        pk_dict = dict()
        for pk_object in self._pk_objects:
            if pk_object.regimen.get_drug_class() is DrugClass.InI:
                if 5 not in pk_dict:
                    pk_dict[5] = list()         # drug that impact a5
                pk_dict[5].append(pk_object)
            elif pk_object.regimen.get_drug_class() is DrugClass.PI:
                if 6 not in pk_dict:
                    pk_dict[6] = list()         # drug that impact a5
                pk_dict[6].append(pk_object)
            else:
                if 1 not in pk_dict:
                    pk_dict[1] = list()  # drug that impact a1
                pk_dict[1].append(pk_object)
        for pk_list in pk_dict.values():
            pd_list.append(self._get_pd_class(pk_list))
        return pd_list

    def _combine_pd(self):
        """
        Update self._propensity_dict based on self._pd_objects, i.e. the pd objects for each drug/drug combination
        :return:
        a0: dict
            dictionary of propensities
        """
        a0 = self._pd_objects[0].get_propensities()
        for pd_object in self._pd_objects[1:]:
            propensities = pd_object.get_propensities()
            for key, value in propensities.items():
                if isinstance(value, torch.Tensor):
                    a0[key] = value
        return a0

    def get_propensity(self):
        return self._propensity_dict

    def get_pd_objects(self):
        return self._pd_objects


class AbstractPharmacodynamics(object):
    """
    Abstract class of pharmacodynamics. Aim is to compute the propensities a1-a6 according to each drug type.
     A strain with certain genotype can also be considered by giving the change of IC50, m and replicative capacity.

    Parameters
    -----------
    pk_objects: array-like
        array containing pk objects, each object contains the pk profile for one drug
        if drug combination is taken (actually only one drug will be accepted and handled now,
        this parameter is for further extension for drug combinations consist of more than
        one drug of same drug class, e.g. Truvada (two RTIs)
    reservoir: boolean
        indicate if latent reservoir is included. If True, there will be 7 reactions (viral dynamics can be extended)
        and new approach will be used for cumulative probability (PGS class: PgSystemReservoirNewApproach)
    macrophage: boolean
        indicate if the macrophage is included. If True, there will be 12 reactions
   Attributes
   ------------
   _pk_objects: array like
   _propensities: dict
        contain the propensity array a1-a6
    """
    def __init__(self, pk_objects, reservoir=False, macrophage=False):
        self._pk_objects = pk_objects
        self._macrophage = macrophage
        self._reservoir = reservoir
        self._propensities = self._calculate_propensity_constant()
        if len(self._pk_objects) == 1:
            pk_object = self._pk_objects[0]
            m = pk_object.regimen.get_hill_coefficient()
            ic_50 = pk_object.regimen.get_ic50()
            pk_profile = pk_object.get_concentration()
            self._eta = pk_profile[..., 0] ** m / (ic_50 ** m + pk_profile[..., 0] ** m)

    def _calculate_propensity_constant(self):
        """
        Calculate the reaction propensities a1-a6 with absence of drug considering replicative capacity [/hr]
        if macrophage is considered, the reactions will be extended to 1-12
        """
        a = dict()
        a[1] = (ViralDynamicParameter.CL + ViralDynamicParameter.CL_T * ViralDynamicParameter.T_u) / 24    # v -> *
        a[2] = (ViralDynamicParameter.delta_PIC_T + ViralDynamicParameter.delta_T1) / 24  # T1 -> *
        a[3] = ViralDynamicParameter.delta_T2 / 24  # T2 -> *
        a[4] = ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u / 24  # V -> T1
        a[5] = ViralDynamicParameter.k_T / 24  # T1 -> T2
        a[6] = ViralDynamicParameter.N_T / 24  # T2 -> T2 + V
        if self._reservoir:
            a[7] = ViralDynamicParameter.P_La5 * ViralDynamicParameter.k_T / 24
            a[5] = ViralDynamicParameter.k_T * (1 - ViralDynamicParameter.P_La5) / 24
        elif self._macrophage:
            a[1] = (ViralDynamicParameter.CL + ViralDynamicParameter.CL_T * ViralDynamicParameter.T_u +
                    ViralDynamicParameter.CL_M * ViralDynamicParameter.M_u) / 24
            a[5] = ViralDynamicParameter.k_T * (1 - ViralDynamicParameter.P_La5) / 24
            a[7] = ViralDynamicParameter.P_La5 * ViralDynamicParameter.k_T / 24       # T1 -> Tl
            a[8] = ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u / 24      # V -> M1
            a[9] = (ViralDynamicParameter.delta_PIC_M + ViralDynamicParameter.delta_M1) / 24      # M1 -> *
            a[10] = ViralDynamicParameter.delta_M2 / 24        # M2 -> *
            a[11] = ViralDynamicParameter.k_M / 24             # M1 -> M2
            a[12] = ViralDynamicParameter.N_M / 24             # M2 -> M2 + V
        return a

    def _compute_distinct_propensities(self):
        """
        Compute the propensities that are impacted by the drug
        """
        raise NotImplementedError

    def get_drug_effect(self):
        return self._eta

    def get_propensities(self):
        return self._propensities


class PharmacodynamicsInI(AbstractPharmacodynamics):
    """
    Class of pharmacodynamics model of InI
    """
    def __init__(self, pk_objects, reservoir=False, macrophage=False):
        super().__init__(pk_objects, reservoir, macrophage)
        if self._reservoir:
            self._propensities[5], self._propensities[7] = self._compute_distinct_propensities()
        elif self._macrophage:
            self._propensities[5], self._propensities[7], self._propensities[11] = self._compute_distinct_propensities()
        else:
            self._propensities[5] = self._compute_distinct_propensities()

    def _compute_distinct_propensities(self):
        """
        Compute the propensity a5 of the mutant strain
        :return:
        a5: ndarray
        """
        a5 = (1 - self._eta) * ViralDynamicParameter.k_T / 24
        if self._reservoir:
            a7 = a5 * ViralDynamicParameter.P_La5
            a5 = a5 * (1 - ViralDynamicParameter.P_La5)
            return a5, a7
        elif self._macrophage:
            a7 = a5 * ViralDynamicParameter.P_La5
            a5 = a5 * (1 - ViralDynamicParameter.P_La5)
            a11 = (1 - self._eta) * ViralDynamicParameter.k_M / 24
            return a5, a7, a11
        else:
            return a5


class PharmacodynamicsCRA(AbstractPharmacodynamics):
    """
    Class of pharmacodynamics model of CRA, for one mutant strain against CRA
    """
    def __init__(self, pk_objects, reservoir=False, macrophage=False):
        super().__init__(pk_objects, reservoir, macrophage)
        if self._macrophage:
            self._propensities[1], self._propensities[4], self._propensities[8] = self._compute_distinct_propensities()
        else:
            self._propensities[1], self._propensities[4] = self._compute_distinct_propensities()

    def _compute_distinct_propensities(self):
        """
        Compute the propensity a1 and a4 of the mutant strain
        :return:
        a1: ndarray
        a4: ndarray
        """

        a4 = (1 - self._eta) * ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u / 24
        if self._macrophage:
            a1 = (ViralDynamicParameter.CL + (1 - self._eta) * ViralDynamicParameter.CL_T * ViralDynamicParameter.T_u +
                  (1 - self._eta) * ViralDynamicParameter.CL_M * ViralDynamicParameter.M_u) / 24
            a8 = (1 - self._eta) * ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u / 24
            return a1, a4, a8
        else:
            a1 = (1 - self._eta) * ViralDynamicParameter.CL_T * ViralDynamicParameter.T_u
            return a1, a4


class PharmacodynamicsRTI(AbstractPharmacodynamics):
    """
    Class of pharmacodynamics model of RTI, for one mutant strain
    """
    def __init__(self, pk_objects, reservoir=False, macrophage=False):
        super().__init__(pk_objects, reservoir, macrophage)
        if self._macrophage:
            self._propensities[1], self._propensities[4], self._propensities[8] = self._compute_distinct_propensities()
        else:
            self._propensities[1], self._propensities[4] = self._compute_distinct_propensities()

    def _compute_distinct_propensities(self):
        """
        Compute the propensity a1 and a4 of the mutant strain
        :return:
        a1: ndarray
        a4: ndarray
        """
        a4 = (1 - self._eta) * ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u / 24

        if self._macrophage:
            a1 = (ViralDynamicParameter.CL + (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u +
                  (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u) / 24
            a8 = (1 - self._eta) * ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u / 24
            return a1, a4, a8
        else:
            a1 = (ViralDynamicParameter.CL + (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u) / 24
            return a1, a4


class PharmacodynamicsPI(AbstractPharmacodynamics):
    """
    Class of pharmacodynamics model of PI, for one mutant strain against PI
    """
    def __init__(self, pk_objects, reservoir=False, macrophage=False):
        super().__init__(pk_objects, reservoir, macrophage)
        if self._macrophage:
            self._propensities[6], self._propensities[12] = self._compute_distinct_propensities()
        else:
            self._propensities[6] = self._compute_distinct_propensities()

    def _compute_distinct_propensities(self):
        """
        Compute the propensity a6 of the mutant strain
        :return:
        a6: ndarray
        """
        a6 = (1 - self._eta) * ViralDynamicParameter.N_T / 24
        if self._macrophage:
            a12 = (1 - self._eta) * ViralDynamicParameter.N_M / 24
            return a6, a12
        else:
            return a6


class PharmacodynamicsTruvada(AbstractPharmacodynamics):
    """
    Class of pharmacodynamics model of Truvada, for one mutant strain against PI
    """

    def __init__(self, pk_objects, mmoa_file=None, reservoir=False, macrophage=False):
        super().__init__(pk_objects, reservoir, macrophage)

        if self._macrophage:
            self._propensities[1], self._propensities[4], self._propensities[8] = \
                self._compute_distinct_propensities(file=mmoa_file)
        else:
            self._propensities[1], self._propensities[4] = self._compute_distinct_propensities(file=mmoa_file)

    def _compute_distinct_propensities(self, file=None, interpolation=True):
        if self._pk_objects[0].regimen.get_drug_name() == 'FTC':
            c_ftc_new = self._pk_objects[0].get_concentration()[..., 0]
            c_tfv_new = self._pk_objects[1].get_concentration()[..., 0]
        else:
            c_ftc_new = self._pk_objects[1].get_concentration()[..., 0]
            c_tfv_new = self._pk_objects[0].get_concentration()[..., 0]
        if interpolation:
            mmoa_file = file or os.path.join(os.path.split(__file__)[0], '../Data/modMMOA_FTC_TDF_zero_extended.csv')
            try:
                conc_effect_matrix = pd.read_csv(mmoa_file)
            except FileNotFoundError:
                raise SystemExit('MMOA file for Truvada not found')
            c_ftc = torch.tensor(conc_effect_matrix['FTC'], dtype=torch.float32)
            c_tfv = torch.tensor(conc_effect_matrix['TFV'], dtype=torch.float32)
            c = torch.stack((c_ftc, c_tfv), dim=1)
            eta = torch.tensor(conc_effect_matrix['eps'], dtype=torch.float32)
            eta_new = griddata(c, eta, (c_ftc_new, c_tfv_new), method='cubic')
        else:
            c = torch.log10(torch.stack((c_ftc_new, c_tfv_new), dim=-1))
            model_file = file or os.path.join(os.path.split(__file__)[0], '../Data/model.pt')
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 500),
                torch.nn.Sigmoid(),
                torch.nn.Linear(500, 100),
                torch.nn.Sigmoid(),
                torch.nn.Linear(100, 50),
                torch.nn.Sigmoid(),
                torch.nn.Linear(50, 1)
            )
            try:
                model.load_state_dict(torch.load(model_file))
            except FileNotFoundError:
                raise SystemExit('NN model not found')
            eta_new = model(c)
            eta_new = torch.clamp(eta_new, min=0, max=1)
        self._eta = 1 - torch.tensor(eta_new, dtype=torch.double)
        a4 = (1 - self._eta) * ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u / 24

        if self._macrophage:
            a1 = (ViralDynamicParameter.CL + (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u +
                  (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u) / 24
            a8 = (1 - self._eta) * ViralDynamicParameter.beta_M * ViralDynamicParameter.M_u / 24
            return a1, a4, a8
        else:
            a1 = (ViralDynamicParameter.CL + (1 / ViralDynamicParameter.rho - (1 - self._eta)) *
                  ViralDynamicParameter.beta_T * ViralDynamicParameter.T_u) / 24
            return a1, a4
