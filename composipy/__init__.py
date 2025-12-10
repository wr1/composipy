import os
import sys

__version__ = "1.5.2rc1"
from .core.material import OrthotropicMaterial, IsotropicMaterial
from .core.property import LaminateProperty
from .core.structure import PlateStructure
from .core.strength import LaminateStrength