from setuptools import setup, find_packages

setup(name='EnergyFlow',
      version='0.2.0',
      description='Energy Flow Basis Implementation',
      author='Patrick T. Komiske III',
      author_email='pkomiske@mit.edu',
      url='https://github.com/pkomiske/EnergyFlow',
      download_url='https://github.com/pkomiske/EnergyFlow/archive/v0.2.0.tar.gz',
      license='GPL-3.0',
      install_requires=['numpy>=1.12.0'],
      extras_require={'generation': ['igraph']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      package_data={'':['data/*']},
      packages=find_packages())
