from phylo import TreeIO, Clade, Tree
from nets.models import AttentionModel
from utils import torch_load_cpu, get_kmer, co2tree, cluster_co2tree, PAMLml

from tqdm import tqdm
from Bio.Phylo.NewickIO import Writer
import torch

from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


class TreeSearcher(object):
    '''Base class for tree searching methods.'''

    def search(self, msa):
        pass


class GreedySPTreeSearcher(TreeSearcher):
    '''Greedy stepwise addition construction of phylogenetic tree.'''

    def __init__(self, estimator):
        # TODO initiate a estimator using _tmp_msa_file, _tmp_tree_file, and
        # ctl_file
        self.estimator = estimator

    def step_search(self, sub_msa, tree, new_clade):
        '''Each step of tree search.
        '''
        opt_lnLK = float('-inf')
        opt_tree = None
        targets = tree.get_descendents()
        for target in tqdm(targets):
            # add the new clade to target
            # TODO draw a picture
            tree.add_clade(target, new_clade)

            lnLK, tmp_tree = self.estimator.estimate_parameter(sub_msa, tree)
            if lnLK > opt_lnLK:
                opt_lnLK = lnLK
                opt_tree = tmp_tree
            
            # delete the new clade from target
            tree.undo_add_clade(target, new_clade)

        return opt_lnLK, opt_tree

    def search(self, msa=None, out_file='./greedySP.out'):
        '''Stepwise search to construct phlogenetic tree.'''
        if msa is None or len(msa) < 1:
            return 1.0, None
        init_msa = msa
        lnLK = 1.0
        # initiate tree using the first sequences
        tree = Tree(rooted=False)
        tree.root.clades.append(Clade(name=init_msa[0].name))
        if len(init_msa) == 1:
            return lnLK, tree

        with open(out_file, 'wt') as f_write:
            f_write.write('Order: ' +
                          ' -> '.join([record.name for record in init_msa]))
            for index in tqdm(range(1, len(init_msa))):
                new_clade = Clade(name=init_msa[index].name)
                lnLK, tree = self.step_search(init_msa[:index+1], tree, new_clade)
                f_write.write('\n\nAdded sp. %s, %s [lnLK: %s]\n' % (
                    index, new_clade.name, lnLK
                ))
                newick_writer = Writer([tree])
                f_write.write(next(newick_writer.to_strings()))
            f_write.write('\n\nFinished greedy stepwise addition construction.\n')
            f_write.write('lnLK: %s\n' % lnLK)
            f_write.write('Tree: %s' % next(Writer([tree]).to_strings()))

        return lnLK, tree


class SPTreeConstructor(object):
    '''Stepwise addition Tree constructor.'''

    def __init__(self, searcher):
        '''Initialize the class.'''
        self.searcher = searcher

    # TODO print some message about result
    def build_tree(self, msa, out_file):
        '''Build the tree.

        :Parameters:
            msa: MultipleSeqAlignment object
            result: file path to save the result
        '''
        return self.searcher.search(msa, out_file)

    def __call__(self, msa, out_file):
        '''Make SPTreeConstructor object callable.'''
        self.build_tree(msa, out_file)


class RLTreeConstructor(object):
    '''Tree constructor using trained REINFORCE model.'''
    def __init__(self,
                 paml_path,
                 load_path,
                 node_dim,
                 embedding_dim=128,
                 n_encode_layers=3):
        '''Initialize model.'''
        self.model = AttentionModel(node_dim=node_dim, embedding_dim=embedding_dim, n_encode_layers=n_encode_layers)

        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
        self.model.load_state_dict(load_data['model'])
        self.model.set_decode_type('greedy')

        if paml_path is None:
            self.paml = None
        else:
            self.paml = PAMLml(paml_command=paml_path, del_tmp_dir=True)

    def _get_kmer_vecs(self, seqs_batch, k_mer=6):
        '''Get k-mer representation of sequences.
        param:
            seqs_batch: list of sequence list
        '''
        vecs_batch = [[get_kmer(seq, k_mer) for seq in seqs] for seqs in seqs_batch]
        return vecs_batch

    def infer_tree(self, seqs, k_mer=6, optim_branch=True):
        lnlk_tree = self.infer_trees([seqs], k_mer=k_mer, optim_branch=optim_branch)[0]
        return lnlk_tree

    def infer_trees(self, seqs_batch, k_mer=6, optim_branch=True):
        '''inferring tree for batch of seuences.'''
        vecs_batch = self._get_kmer_vecs(seqs_batch, k_mer=k_mer)

        vecs_tensor = torch.tensor(vecs_batch, dtype=torch.float)
        _, _, circular_orders = self.model(vecs_tensor, return_pi=True)
        circular_orders = circular_orders.tolist()

        trees = [co2tree(vecs, co) for vecs, co in zip(vecs_batch, circular_orders)]

        if self.paml is None or not optim_branch:
            return trees
        else:
            # optimaize the branch length by PAML
            aligns = [MultipleSeqAlignment([SeqRecord(Seq(s, generic_dna), name=str(i+1), id=str(i+1))
                                            for i, s in enumerate(msa)])
                    for msa in seqs_batch]
            lnlk_trees = [self.paml.estimate_parameter(align, tree) for align, tree in zip(aligns, trees)]
            return lnlk_trees


class TSPTreeConstructor(object):
    '''Tree constructor using trained REINFORCE model.'''
    def __init__(self, tsp_func, paml_path=None):
        '''Initialize PAML'''
        self.tsp_func = tsp_func
        if paml_path is None:
            self.paml = None
        else:
            self.paml = PAMLml(paml_command=paml_path, del_tmp_dir=True)

    def _get_kmer_vecs(self, seqs_batch, k_mer=6):
        '''Get k-mer representation of sequences.
        param:
            seqs_batch: list of sequence list
        '''
        vecs_batch = [[get_kmer(seq, k_mer) for seq in seqs] for seqs in seqs_batch]
        return vecs_batch

    def infer_tree(self, seqs, k_mer=6, optim_branch=True):
        lnlk_tree = self.infer_trees([seqs], k_mer=k_mer, optim_branch=optim_branch)[0]
        return lnlk_tree

    def infer_trees(self, seqs_batch, k_mer=6, optim_branch=True):
        '''inferring tree for batch of seuences.'''
        vecs_batch = self._get_kmer_vecs(seqs_batch, k_mer=k_mer)
        circular_orders = [self.tsp_func(vecs) for vecs in vecs_batch]

        trees = [cluster_co2tree(vecs, co) for vecs, co in zip(vecs_batch, circular_orders)]
        if self.paml is None or not optim_branch:
            return trees
        else:
            # optimaize the branch length by PAML
            aligns = [MultipleSeqAlignment([SeqRecord(Seq(s, generic_dna), name=str(i+1), id=str(i+1))
                                            for i, s in enumerate(msa)])
                    for msa in seqs_batch]
            lnlk_trees = [self.paml.estimate_parameter(align, tree) for align, tree in zip(aligns, trees)]
            return lnlk_trees
            
