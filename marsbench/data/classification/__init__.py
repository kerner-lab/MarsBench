"""
Classification datasets for Mars surface image classification tasks.
"""
from .Atmospheric_Dust_Classification_EDR import Atmospheric_Dust_Classification_EDR
from .Atmospheric_Dust_Classification_RDR import Atmospheric_Dust_Classification_RDR
from .BaseClassificationDataset import BaseClassificationDataset
from .Change_Classification_CTX import Change_Classification_CTX
from .Change_Classification_HiRISE import Change_Classification_HiRISE
from .DoMars16k import DoMars16k
from .Frost_Classification import Frost_Classification
from .Landmark_Classification import Landmark_Classification
from .Multi_Label_MER import Multi_Label_MER
from .Surface_Classification import Surface_Classification

__all__ = [
    "BaseClassificationDataset",
    "DoMars16k",
    "Landmark_Classification",
    "Surface_Classification",
    "Frost_Classification",
    "Atmospheric_Dust_Classification_RDR",
    "Atmospheric_Dust_Classification_EDR",
    "Change_Classification_HiRISE",
    "Change_Classification_CTX",
    "Multi_Label_MER",
]
