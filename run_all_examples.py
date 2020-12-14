# runs all EnergyFlow examples

import os
import subprocess
import sys

import energyflow as ef

run_animation_example = False

PYTHON = 'python' if sys.version_info[0] == 2 else 'python3'
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'examples')
EXAMPLES = ef.utils.ALL_EXAMPLES

if not run_animation_example:
    EXAMPLES = EXAMPLES[:-1]

for example in EXAMPLES:
    command = "{} {} ''".format(PYTHON, os.path.join(EXAMPLE_DIR, example))
    print(command)
    subprocess.call(command.split())
