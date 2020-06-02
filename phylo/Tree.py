'''Tree class'''
from Bio.Phylo import BaseTree


class Tree(BaseTree.Tree):
    '''Tree class.
    
    add some tree manipulation methods for stepwise addition
    construction'''

    # def __init__(self):
    #     pass

    # get parent of target clade
    def get_parent(self, target=None):
        path = self.get_path(target)
        if isinstance(path, list) and len(path) > 0:
            path = [self.root] + path
            return path[-2]
        else:
            # raise ValueError('target %s is not in this tree' % repr(t))
            return None

    def get_descendents(self):
        descendents = self.get_terminals() + self.get_nonterminals()
        descendents.remove(self.root)
        return descendents

    # add a new sequence to tree
    def add_clade(self, target=None, new_clade=None, internal_clade_name=None):
        '''Add a new clade to Tree.
        This function only work for unrooted tree. There is a special case:
        unrooted tree's root have at more 3 child.'''
        if target is self.root:
            raise IndexError('can not add clade to root.')
        parent_target = self.get_parent(target)
        # if parent_target is None:
        #     raise ValueError('target %s is not in this tree' % repr(target))
        # if not isinstance(new_clade, Clade):
        #     raise ValueError('%s (type %s) is not a valid Clade object' % 
        #                      (new_clade, type(new_clade))
        # if not isinstance(internal_clade_name, str) or len(internal_clade_name) == 0:
        #     raise ValueError('internal_clade_name must be a non-empty str')

        # add clade to root's next generation
        if parent_target is self.root and len(self.root.clades) < 3:
                parent_target.clades.append(new_clade)
        else:
            # other conditions
            # half_length = None
            # if target.branch_length is not None:
            #     half_length = target.branch_length / 2.0
            #     target.branch_length = half_length
            internal_clade = Clade(name=internal_clade_name,
                                   clades=[target, new_clade])

            # Remove target from parent clade, add internal clade into parent clade
            parent_target.clades.remove(target)
            parent_target.clades.append(internal_clade)


    # remove sequence and connected internal clade from tree
    def undo_add_clade(self, target=None, new_clade=None):
        if self.get_parent(target) is self.root:
            self.root.clades.remove(new_clade)
        else:
            # if target.branch_length is not None:
            #     target.branch_length += self.get_parent(target).branch_length
            parent_target = self.get_parent(self.get_parent(target))
            parent_target.clades.remove(self.get_parent(target))
            parent_target.clades.append(target)


Clade = BaseTree.Clade
# class Clade(BaseTree.Clade):
#     pass