from setuptools import setup, find_packages

setup(name='EnergyFlow',
      version='0.3.1',
      description='Python implementation of the energy flow basis',
      author='Patrick T. Komiske III',
      author_email='pkomiske@mit.edu',
      url='https://github.com/pkomiske/EnergyFlow',
      download_url='https://github.com/pkomiske/EnergyFlow/archive/v0.3.0.tar.gz',
      license='GPL-3.0',
      install_requires=['numpy>=1.12.0', 'six>=1.10.0'],
      extras_require={'generation': ['python-igraph']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      package_data={'': ['data/*']},
      keywords=['physics', 'jets', 'energy flow', 'correlator', 'multigraph'],
      packages=find_packages())
