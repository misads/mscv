import re
import setuptools
from setuptools import setup

with open('mscv/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mscv',
    version=__version__,
    description='mscv - A foundational python library for computer vision research.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/misads/mscv',
    author='Haoyu Xu',
    author_email='xuhaoyu@tju.edu.cn',
    license='MIT',
    install_requires=[
        "numpy",
        "utils-misc>=0.0.5"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
