import os

from distutils.core import setup, Extension

import numpy as np

from Cython.Distutils import build_ext


descr = 'Fast algorithm with dual extrapolation for the Lasso'

version = None
with open(os.path.join('celer', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'celer'
DESCRIPTION = descr
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mathurinm/celer.git'
VERSION = version
URL = 'https://mathurinm.github.io/celer'

setup(name='celer',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['celer'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('celer.lasso_fast',
                    sources=['celer/lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.cython_utils',
                    sources=['celer/cython_utils.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.logreg_fast',
                    sources=['celer/logreg_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.PN_logreg',
                    sources=['celer/PN_logreg.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.multitask_fast',
                    sources=['celer/multitask_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
          Extension('celer.group_lasso_fast',
                    sources=['celer/group_lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"]),
      ],
      )
