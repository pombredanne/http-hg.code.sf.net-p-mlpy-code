from distutils.core import setup, Extension
from distutils.sysconfig import *
from distutils.util import *

import os
import os.path
import numpy

data_files = []

# Include gsl dlls for the win32 distribution
if get_platform() == "win32":
    dlls = ["mlpy\gslwin\libgsl-0.dll",
            "mlpy\gslwin\libgslcblas-0.dll"]
    data_files += [("Lib\site-packages\mlpy", dlls)]

## Extra compile args
extra_compile_args = ['-Wno-strict-prototypes']

# Python include
py_include = get_python_inc()

# NumPy include
numpy_lib = os.path.dirname(numpy.__file__)
numpy_include = os.path.join(numpy_lib, 'core/include')

# Base include
base_include = [py_include, numpy_include]

scripts = []

ext_modules = [Extension("mlpy.gsl", ["mlpy/gsl/gsl.c"],
                         libraries=['gsl', 'gslcblas', 'm']),
               Extension("mlpy.liblinear",
                         ["mlpy/liblinear/liblinear/linear.cpp",
                          "mlpy/liblinear/liblinear/tron.cpp",
                          "mlpy/liblinear/liblinear.c",
                          "mlpy/liblinear/liblinear/blas/daxpy.c",
                          "mlpy/liblinear/liblinear/blas/ddot.c",
                          "mlpy/liblinear/liblinear/blas/dnrm2.c",
                          "mlpy/liblinear/liblinear/blas/dscal.c"],
                         include_dirs=base_include),
               Extension("mlpy.libsvm",
                         ["mlpy/libsvm/libsvm/svm.cpp",
                          "mlpy/libsvm/libsvm.c"],
                         include_dirs=base_include),
               Extension("mlpy.libml",
                         ["mlpy/libml/src/alloc.c",
                          "mlpy/libml/src/dist.c",
                          "mlpy/libml/src/get_line.c",
                          "mlpy/libml/src/matrix.c",
                          "mlpy/libml/src/mlg.c",
                          "mlpy/libml/src/nn.c",
                          "mlpy/libml/src/parser.c",
                          "mlpy/libml/src/read_data.c",
                          "mlpy/libml/src/rn.c",
                          "mlpy/libml/src/rsfn.c",
                          "mlpy/libml/src/sampling.c",
                          "mlpy/libml/src/sort.c",
                          "mlpy/libml/src/svm.c",
                          "mlpy/libml/src/tree.c",
                          "mlpy/libml/src/trrn.c",
                          "mlpy/libml/src/ttest.c",
                          "mlpy/libml/src/unique.c",
                          "mlpy/libml/libml.c"],
                         include_dirs=base_include),
               Extension("mlpy.kmeans",
                         ["mlpy/kmeans/c_kmeans.c",
                          "mlpy/kmeans/kmeans.c"],
                         include_dirs=base_include,
                         libraries=['gsl', 'gslcblas', 'm']),
               Extension("mlpy.kernel",
                         ["mlpy/kernel/c_kernel.c",
                          "mlpy/kernel/kernel.c"],
                         include_dirs=base_include,
                         libraries=['m']),
               Extension("mlpy.canberra",
                         ["mlpy/canberra/c_canberra.c",
                          "mlpy/canberra/canberra.c"],
                         include_dirs=base_include,
                         libraries=['m']),
               Extension("mlpy.adatron",
                         ["mlpy/adatron/c_adatron.c",
                          "mlpy/adatron/adatron.c"],
                         include_dirs=base_include,
                         libraries=['m']),
               Extension("mlpy.findpeaks",
                         ["mlpy/findpeaks/findpeaks.c"],
                         include_dirs=base_include),
               Extension("mlpy.c_findpeaks",
                         ["mlpy/findpeaks/c_findpeaks.c"],
                         extra_compile_args=extra_compile_args,
                         include_dirs=base_include),
               Extension("mlpy.wavelet._dwt",
                         ["mlpy/wavelet/dwt.c"],
                         extra_compile_args=extra_compile_args,
                         include_dirs=base_include,
                         libraries=['gsl', 'gslcblas', 'm']),
               Extension("mlpy.wavelet._uwt",
                         ["mlpy/wavelet/uwt.c"],
                         extra_compile_args=extra_compile_args,
                         include_dirs=base_include,
                         libraries=['gsl', 'gslcblas', 'm']),
               Extension("mlpy.hcluster.chc",
                         ["mlpy/hcluster/hc.c"],
                         extra_compile_args=extra_compile_args,
                         include_dirs=base_include,
                         libraries=['m']),
               Extension("mlpy.bordacount.cborda",
                         ["mlpy/bordacount/borda.c"],
                         extra_compile_args=extra_compile_args,
                         include_dirs=base_include)
               ]


packages=['mlpy', 'mlpy.wavelet', 'mlpy.hcluster',
          'mlpy.bordacount']


setup(name = 'mlpy',
      version='3.0a',
      requires=['numpy (>=1.3.0)', 'scipy (>=0.7.0)', 'gsl (>=1.14)'],
      description='mlpy - Machine Learning Py - ' \
          'High-Performance Python Package for Predictive Modeling',
      author='mlpy Developers',
      author_email='davide.albanese@gmail.com',
      packages=packages,
      url='',
      download_url='',
      license='GPLv3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Programming Language :: C',
                   'Programming Language :: Python',
                   'Programming Language :: Cython',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   ],
      ext_modules=ext_modules,
      scripts=scripts,
      data_files=data_files
      )
