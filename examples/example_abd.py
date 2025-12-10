#!/usr/bin/env python3
"""Example script demonstrating ABD matrix computation using composipy.

This script creates a laminate and computes its ABD stiffness matrix.
"""

import sys
import os

# Add the parent directory to the path to import composipy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from composipy import OrthotropicMaterial, LaminateProperty

# Define material properties (example values)
E1 = 129500  # MPa
E2 = 9370    # MPa
v12 = 0.38
G12 = 5240   # MPa
thickness = 0.2  # mm

# Create orthotropic material
ply = OrthotropicMaterial(E1, E2, v12, G12, thickness)

# Define stacking sequence (angles in degrees)
stacking = [90, 0, 90]

# Create laminate
laminate = LaminateProperty(stacking, ply)

# Compute and print ABD matrix
print("ABD Matrix:")
print(laminate.ABD)
print("\nA Matrix (extension):")
print(laminate.A)
print("\nB Matrix (coupling):")
print(laminate.B)
print("\nD Matrix (bending):")
print(laminate.D)
