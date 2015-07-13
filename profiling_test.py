__author__ = 'lposti'


import pstats
import cProfile
from fJmodel.fJmodel import FJmodel

f = FJmodel('fJmodel/examples/Hernq_ngc6125_dp0.5-1.0_dz0.5-1.0_c0.1_4.out')

cProfile.runctx("f.project(0.)", globals(), locals(), "profile.prof")

s = pstats.Stats("profile.prof")
s.strip_dirs().sort_stats("time").print_stats()