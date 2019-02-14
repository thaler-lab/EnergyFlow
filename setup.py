import re
import sys

from setuptools import setup

with open('energyflow/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(version=__version__)
