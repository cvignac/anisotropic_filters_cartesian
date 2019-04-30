
from setuptools import setup
from setuptools import find_packages

setup(name='anisotropic_filters_cartesian',
      version='0.1',
      description='Anisotropic filters for Cartesian product graphs',
      author='Clément Vignac',
      author_email='clement.vignac@epfl.ch',
      download_url='https://github.com/cvignac/anisotropic_filters_cartesian',
      license='CC BY-NC',
      install_requires=['numpy',
                        'torch',
                        'scipy',
                        'matplotlib'
                        ],
packages=find_packages())
