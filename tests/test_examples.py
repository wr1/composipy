'''
Test of example scripts

Test that the example scripts run without error.
'''

import subprocess
import sys
import os


def test_example_abd():
    script_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_abd.py')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'ABD Matrix:' in result.stdout


def test_example_nastran():
    script_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_nastran.py')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Nastran PCOMP Card:' in result.stdout


def test_example_coupling():
    script_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_coupling.py')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Evaluating Coupling Terms' in result.stdout
