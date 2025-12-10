#!/usr/bin/env python3
"""Example script evaluating coupling terms (B matrix) in ABD for various laminates.

This script creates a series of laminates, including symmetric and asymmetric ones,
and evaluates the coupling terms in the ABD matrix using rich formatting.
"""

import sys
import os
import numpy as np
from rich.console import Console
from rich.table import Table

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

# Define a series of stacking sequences: symmetric and asymmetric
stacking_sequences = [
    [0, 0],  # Symmetric
    [90, 90],  # Symmetric
    [0, 90, 90, 0],  # Symmetric
    [45, -45, -45, 45],  # Symmetric
    [0],  # Single ply, symmetric
    [0, 90],  # Asymmetric
    [45, 0],  # Asymmetric
    [0, 45, 90],  # Asymmetric
    [30, -30, 60],  # Asymmetric
    [0, 45, -45, 90, 30],  # Asymmetric
    [10, 20, 30, 40, 50],  # Asymmetric
    [0, 0, 0, 90],  # Asymmetric
    [45, 45, -45, -45],  # Symmetric
    [0, 45, 90, 135],  # Asymmetric
    [60, 30, 0, -30, -60],  # Asymmetric
]

console = Console()

console.print("[bold blue]Evaluating Coupling Terms (B Matrix) in ABD for Various Laminates[/bold blue]")
console.print("\n")

# Summary table
summary_table = Table(title="Summary of Coupling Terms")
summary_table.add_column("Stacking Sequence", style="cyan", no_wrap=True)
summary_table.add_column("Max |B|", justify="right")
summary_table.add_column("Is Symmetric", justify="center")

for stacking in stacking_sequences:
    # Create laminate
    laminate = LaminateProperty(stacking, ply)
    
    # Get ABD matrix
    ABD = laminate.ABD
    
    # Get B matrix
    B = laminate.B
    
    # Calculate max absolute value of B
    max_coupling = np.max(np.abs(B))
    
    # Check if symmetric (B should be zero for symmetric laminates)
    is_symmetric = "Yes" if np.allclose(B, 0, atol=1e-10) else "No"
    
    summary_table.add_row(str(stacking), f"{max_coupling:.2e}", is_symmetric)
    
    # Full ABD matrix table
    abd_table = Table(title=f"ABD Matrix for Stacking: {stacking}")
    abd_table.add_column("", style="dim")
    for j in range(6):
        abd_table.add_column(f"Col {j+1}", justify="right")
    
    for i in range(6):
        row = [f"Row {i+1}"] + [f"{ABD[i,j]:.2e}" for j in range(6)]
        abd_table.add_row(*row)
    
    console.print(abd_table)
    console.print("\n")

console.print(summary_table)
