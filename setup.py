from setuptools import setup, find_packages
from phaseconfig import __version__

setup(
    name='phaseconfig',
    version=__version__,
    author='Andrei Shkarin',
    author_email='andrei.shkarin@kit.edu',
    url='http://ufo.kit.edu',
    packages=find_packages(),
    scripts=['bin/phaseconfig'],
    install_requires=['tifffile', "numpy"],
    #long_description=open('README.md').read(),
)
