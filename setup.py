import sys

from setuptools import setup, find_packages

import energyflow 

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name='EnergyFlow',
    version=energyflow.__version__,
    description='Python package for the Energy Flow suite of tools',
    long_description='See the [EnergyFlow website](https://energyflow.network).',
    long_description_content_type='text/markdown',
    author='Patrick T. Komiske III',
    author_email='pkomiske@mit.edu',
    url='https://energyflow.network',
    project_urls={'GitHub': 'https://github.com/pkomiske/EnergyFlow'},
    license='GPL-3.0',
    install_requires=['numpy>=1.14.0', 'six>=1.10.0'],
    extras_require={'generation': ['python-igraph'], 
                    'examples': ['keras', 'tensorflow', 'scikit-learn', 'matplotlib']},
    setup_requires=pytest_runner,
    tests_require=['pytest'],
    keywords=['physics', 'jets', 'energy flow', 'correlator', 'multigraph', 'polynomial', 
              'EFP', 'EFN', 'PFN', 'Energy Flow Network', 'Particle Flow Network', 'architecture'],
    packages=find_packages(),
    package_data={'': ['data/*']}
)
