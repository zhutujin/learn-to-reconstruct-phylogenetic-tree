'''Collection of modules for reconstructions of phylogenetic trees

Maximum likelihood criterion is used to calculate tree scores, and
reinforcement learning performs the stepwise addition tree constructions
'''
from . import TreeIO
from .Tree import Tree, Clade