from setuptools import setup, find_packages
import energyflow 

setup(
name='EnergyFlow',
version=energyflow.__version__,
description='Python implementation of the Energy Flow basis',
author='Patrick T. Komiske III',
author_email='pkomiske@mit.edu',
url='https://pkomiske.github.io/EnergyFlow',
project_urls={'GitHub': 'https://github.com/pkomiske/EnergyFlow'},
license='GPL-3.0',
install_requires=['numpy>=1.12.0', 'six>=1.10.0'],
extras_require={'generation': ['python-igraph']},
setup_requires=['pytest-runner'],
tests_require=['pytest'],
keywords=['physics', 'jets', 'energy flow', 'correlator', 'multigraph', 'EFP'],
packages=find_packages(),
package_data={'': ['data/*']}
)
