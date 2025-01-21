from distutils.extension import Extension

from Cython.Build import cythonize
from setuptools import setup

#  python setup_branched_ssg_matcher.py  build_ext --inplace or CC=clang python setup_branched_ssg_matcher.py  build_ext --inplace
setup(ext_modules=cythonize(Extension("branched_ssg_matcher", ["branched_ssg_matcher.pyx"], extra_compile_args=["-O3", '-std=c++20'], language="c++")))  # requires gcc11
