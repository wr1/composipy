'''
Test of PCOMP generator

Test the build_pcomp function for generating Nastran PCOMP cards.
'''

import pytest

from composipy.nastranapi.pcomp_generator import build_pcomp


def test_build_pcomp_simple():
    sequence = [45, -45, 0, 90]
    midi = [1, 1, 1, 1]
    ti = [0.1, 0.1, 0.1, 0.1]
    pid = 1
    result = build_pcomp(sequence, midi, ti, pid)
    expected = "PCOMP,1,,,,,,+\n+,1,0.1,45,YES,1,0.1,-45,NO+\n+,1,0.1,0,NO,1,0.1,90,YES+\n"
    assert result == expected


def test_build_pcomp_with_z0():
    sequence = [0, 90]
    midi = [2, 2]
    ti = [0.2, 0.2]
    pid = 2
    z0 = 0.5
    result = build_pcomp(sequence, midi, ti, pid, z0=z0)
    expected = "PCOMP,2,0.5,,,,,+\n+,2,0.2,0,YES,2,0.2,90,YES+\n"
    assert result == expected


def test_build_pcomp_fiber_sout():
    sequence = [45, -45]
    midi = [1, 1]
    ti = [0.1, 0.1]
    pid = 1
    result = build_pcomp(sequence, midi, ti, pid, sout='FIBER')
    expected = "PCOMP,1,,,,,,+\n+,1,0.1,45,YES,1,0.1,-45,YES+\n"
    assert result == expected
