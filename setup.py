import re
import sys

from setuptools import setup

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open('energyflow/__init__.py', 'r') as f:
     __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(
    version=__version__,
    setup_requires=pytest_runner
)
