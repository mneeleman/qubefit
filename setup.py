#!/usr/bin/env python
# from __future__ import absolute_import, division, print_function
#
# Find the version
#
import re
def get_version():

    version_file = './qubefit/__init__.py'
    ver = 'unknown'
    with open(version_file, "r") as f:
        for line in f.readlines():
            mo = re.match('__version__ = "(.*)"', line)
            if mo:
                ver = mo.group(1)
    return ver
#
# Standard imports
#
import glob, os
from distutils.extension import Extension
#
# setuptools' sdist command ignores MANIFEST.in
#
#from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()
#
# THESE SETTINGS NEED TO BE CHANGED FOR EVERY PRODUCT.
#
setup_keywords['name'] = 'qubefit'
setup_keywords['description'] = 'Qubefit Software'
setup_keywords['author'] = 'Marcel Neeleman'
setup_keywords['author_email'] = 'neeleman@mpia.de'
setup_keywords['license'] = 'MIT'
setup_keywords['url'] = 'https://github.com/mneeleman/qubefit'
#
# END OF SETTINGS THAT NEED TO BE CHANGED.
#
setup_keywords['version'] = get_version()
#
# Use README.rst as long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.  These are automated, & should
# be left alone unless you are an expert.
#
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>3.7.0)']
# setup_keywords['install_requires'] = ['Python (>2.7.0)']
setup_keywords['zip_safe'] = False
#setup_keywords['use_2to3'] = True
setup_keywords['packages'] = find_packages()
#setup_keywords['package_dir'] = {'':'py'}
#setup_keywords['cmdclass'] = {'version': DesiVersion, 'test': DesiTest, 'sdist': DistutilsSdist}
#etup_keywords['test_suite']='{name}.tests.{name}_test_suite.{name}_test_suite'.format(**setup_keywords)
setup_keywords['setup_requires']=['pytest-runner']
setup_keywords['tests_require']=['pytest']

#
# Add internal data directories.
#

data_files = []

# walk through the data directory, adding all files
data_generator = os.walk('qubefit/data')
for path, directories, files in data_generator:
    for f in files:
        data_path = '/'.join(path.split('/')[1:])
        data_files.append(data_path + '/' + f)
setup_keywords['package_data'] = {'qubefit': data_files,
                                  '': ['*.rst', '*.txt', '*.yaml']}
setup_keywords['include_package_data'] = True

#
# adding scripts
#

entry_points = {}
entry_points['console_scripts'] = ['qfgui = qubefit.scripts.qfgui:main',
                                   'qubemom = qubefit.scripts.qubemom:main']
setup_keywords['entry_points'] = entry_points
#
# Run setup command.
#
setup(**setup_keywords)

