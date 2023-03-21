import configparser
import os
from enum import Enum

import pandas as pd

# TODO: unit of IC50: TDF and FTC in umol, others in nmol, please unify the unit!!


class DrugClass(Enum):
    """
    Enum class to represent the class of drug
    """
    PI = 'PI'
    InI = 'InI'
    NRTI = 'NRTI'
    RTI = 'RTI'
    CRA = 'CRA'


class Drug(Enum):
    """
    Enum class to represent drug, contains drug class, molecular weight, IC50 and hill coefficient of
    each drug.
    Attributes:
        drug_class: DrugClass
        molecular_weight: float
            molecular weight (g/mol)
        IC_50: float
            drug concentration at the target site (blood or intracellular, ect.)
            where the target process is inhibited by 50%  (nM)
        m: float
            hill coefficient
    """
    MVC = (DrugClass.CRA, None, 5.06, 0.61)
    EFV = (DrugClass.RTI, 315.67, 10.8, 1.69)
    NVP = (DrugClass.RTI, None, 116, 1.55)
    DLV = (DrugClass.RTI, None, 336, 1.56)
    ETR = (DrugClass.RTI, None, 8.59, 1.81)
    RPV = (DrugClass.RTI, None, 7.73, 1.92)
    RAL = (DrugClass.InI, None, 25.5, 1.1)
    EVG = (DrugClass.InI, None, 55.6, 0.95)
    DTG = (DrugClass.InI, 419.38, 89, 1.3)      # nM for IC50
    ATV = (DrugClass.PI, None, 23.9, 2.69)
    APV = (DrugClass.PI, None, 262, 2.09)
    DRV = (DrugClass.PI, None, 45, 3.61)
    IDV = (DrugClass.PI, None, 130, 4.53)
    LPV = (DrugClass.PI, None, 70.9, 2.05)
    NFV = (DrugClass.PI, None, 327, 1.81)
    SQV = (DrugClass.PI, None, 88, 3.68)
    TPV = (DrugClass.PI, None, 483, 2.51)
    TDF = (DrugClass.NRTI, 635.5, 0.1, 1)       # uM for IC50
    TDF_vag = (DrugClass.NRTI, 635.5, 0.156909539547242, 1)    # used to change the ic50 of TDF
    TDF_col = (DrugClass.NRTI, 635.5, 0.0941663630292149, 1)
    FTC = (DrugClass.NRTI, 247.2, 0.82, 1)      # uM for IC50
    ISL = (DrugClass.NRTI, 293.258, 0.440029, 1)        # uM for IC50
    # 3TC = (DrugClass.NRTI, None, None, 1)

    def __init__(self, drug_class, molecular_weight, ic50, m):
        self.drug_class = drug_class
        self.molecular_weight = molecular_weight
        self.IC_50 = ic50
        self.m = m


class ViralDynamicParameter:
    """
    Class which contains the viral dynamic parameters
    """
    CL = None
    rho = None
    beta_T = None
    lambda_T = None
    delta_T = None
    delta_T1 = None
    delta_T2 = None
    delta_PIC_T = None
    k_T = None
    N_T = None
    CL_T = None
    T_u = None
    beta_M = None
    lambda_M = None
    delta_M = None
    delta_M1 = None
    delta_M2 = None
    delta_PIC_M = None
    k_M = None
    N_M = None
    CL_M = None
    M_u = None
    P_Ma4 = None
    P_La5 = None

    @classmethod
    def set_vd_parameters(cls, file='config.ini'):
        """
        Read the viral dynamic parameters from a file.
        :param
        file: str, name of file
        """
        config = configparser.ConfigParser()
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../config', file))
        config.read(filepath)
        cls.CL = float(config['viraldynamics']['CL'])
        cls.rho = float(config['viraldynamics']['rho'])
        cls.beta_T = float(config['viraldynamics']['beta_T'])
        cls.lambda_T = float(config['viraldynamics']['lambda_T'])
        cls.delta_T = float(config['viraldynamics']['delta_T'])
        cls.delta_T1 = float(config['viraldynamics']['delta_T1'])
        cls.delta_T2 = float(config['viraldynamics']['delta_T2'])
        cls.delta_PIC_T = float(config['viraldynamics']['delta_PIC_T'])
        cls.k_T = float(config['viraldynamics']['k_T'])
        cls.N_T = float(config['viraldynamics']['N_T'])
        cls.CL_T = (1 / cls.rho - 1) * cls.beta_T     # clearance rate of unsuccessful infected virus
        cls.T_u = cls.lambda_T / cls.delta_T        # steady state level of uninfected T-cells
        cls.beta_M = float(config['viraldynamics']['beta_M'])
        cls.lambda_M = float(config['viraldynamics']['lambda_M'])
        cls.delta_M = float(config['viraldynamics']['delta_M'])
        cls.delta_M1 = float(config['viraldynamics']['delta_M1'])
        cls.delta_M2 = float(config['viraldynamics']['delta_M2'])
        cls.delta_PIC_M = float(config['viraldynamics']['delta_PIC_M'])
        cls.k_M = float(config['viraldynamics']['k_M'])
        cls.N_M = float(config['viraldynamics']['N_M'])
        cls.CL_M = (1 / cls.rho - 1) * cls.beta_M  # clearance rate of unsuccessful infected virus
        cls.M_u = cls.lambda_M / cls.delta_M  # steady state level of uninfected T-cells
        cls.P_Ma4 = float(config['viraldynamics']['P_Ma4'])
        cls.P_La5 = float(config['viraldynamics']['P_La5'])


class ExtinctionCalculator:
    """
    Class to calculate the extinction probability with absence of drug.
    """
    @classmethod
    def get_pe_basic(cls):
        """
        Calculate the extinction probability with absence of drug for the basic viral dynamics.
        :return:
        PE: float
            extinction probability with absence of drug.
        """
        return 0.90177743
