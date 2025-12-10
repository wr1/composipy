#!/usr/bin/env python3
"""Example script demonstrating Nastran PCOMP card generation using composipy.

This script creates a laminate and generates the corresponding Nastran PCOMP card.
"""

import sys
import os

# Add the parent directory to the path to import composipy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from composipy import OrthotropicMaterial, LaminateProperty
from composipy.nastranapi.pcomp_generator import build_pcomp

# Define material properties (example values)
E1 = 129500  # MPa
E2 = 9370    # MPa
v12 = 0.38
G12 = 5240   # MPa
thickness = 0.2  # mm

# Create orthotropic material
ply = OrthotropicMaterial(E1, E2, v12, G12, thickness)

# Define stacking sequence (angles in degrees)
stacking = [45, -45, 0, 90, 0, 90, -45, 45]

# Create laminate
laminate = LaminateProperty(stacking, ply)

# Prepare inputs for build_pcomp
sequence = stacking
midi = 1  # Single material ID
thicknesses = ply.thickness  # Single thickness
pid = 1

# Generate Nastran PCOMP card
pcomp_card = build_pcomp(sequence, midi, thicknesses, pid)

# Print the PCOMP card
print("Nastran PCOMP Card:")
print(pcomp_card)
