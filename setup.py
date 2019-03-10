"""Setup script for autograd."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy']

setup(
    name='autograd',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('autograd')],
    description='Automatic Differentiation for CS 189 at UC Berkeley')