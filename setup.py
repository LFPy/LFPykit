#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''lfpy-forward-models setuptools file

'''

import setuptools
import os

d = {}
exec(open(os.path.join('lfpy_forward_models', 'version.py')).read(), None, d)
version = d['version']


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='lfpy_forward_models',
    version=version,
    author='LFPy-team',
    author_email='lfpy@users.noreply.github.com',
    description='Electrostatic forward models for multicompartment neuron models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LFPy/lfpy_forward_models',
    download_url='https://github.com/LFPy/lfpy_forward_models/tarball/v{}'.format(version),
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 2 - Pre-Alpha',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.4',
        ],
    extras_require={'tests': ['pytest'],
                    'docs': ['sphinx', 'numpydoc', 'sphinx_rtd_theme'],
                    },
    dependency_links=[],
    provides=['lfpy_forward_models']
)
