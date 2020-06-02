'''Utilities for Rlphy phylogenetic tree constructor.'''
import os
import shutil
from io import StringIO

from phylo import TreeIO

from Bio.Phylo.PAML.baseml import Baseml
from Bio.Align import MultipleSeqAlignment


def aln_write(msa=None, out=None):
    '''save Bio.Align.MultipleSeqAlignment object to phylip format file with
    supporting PAML.'''
    with open(out, 'w') as f_write:
        f_write.write('%s %s' % (len(msa), msa.get_alignment_length()))
        for seq_record in msa:
            f_write.write('\n%s  %s' % (seq_record.name, str(seq_record.seq)))


# use tempfile to create a temporary dir
class PAMLml(Baseml):
    '''An interface to PAML, estimate the parameters of substitutions model
    and branch lengths'''

    # TODO set runmode
    def __init__(self, paml_command='baseml', tmp_dir='./paml_tmp_dir',
                 ctl_file=None, del_tmp_dir=True):
        '''Don't set seqfile, treefile and outfile in ctl_file.'''
        Baseml.__init__(self, working_dir=tmp_dir)

        # default options of PAML baseml
        self._options = {'noisy': 2,  # 0,1,2,3: how much rubbish on the screen
                         'verbose': 0,  # 1: detailed output, 0: concise output
                         'runmode': 0,  # 0: user tree; 1: semi-automatic; 2: automatic
                                        # 3: StepwiseAddition; (4,5):PerturbationNNI
                         'model': 4,  # 0:JC69, 1:K80, 2:F81, 3:F84, 4:HKY85
                                      # 5:T92, 6:TN93, 7:REV, 8:UNREST, 9:REVu; 10:UNRESTu
                         'model_options': None,
                         'Mgene': 0,  # 0:rates, 1:separate; 2:diff pi, 3:diff kapa, 4:all diff
                         'ndata': 1,  # number of data sets
                         'clock': 0,  # 0:no clock, 1:clock; 2:local clock; 3:CombinedAnalysis
                         'fix_kappa': 0,  # 0: estimate kappa; 1: fix kappa at value below
                         'kappa': 5,  # initial or fixed kappa
                         'fix_alpha': 0,  # 0: estimate alpha; 1: fix alpha at value below
                         'alpha': 0.5,  # initial or fixed alpha, 0:infinity (constant rate)
                         'Malpha': 0,  # 1: different alpha's for genes, 0: one alpha
                         'ncatG': 5,  # # of categories in the dG, AdG, or nparK models of rates
                         'fix_rho': None,  # 0: estimate rho; 1: fix rho at value below
                         'rho': None,  # 0: estimate rho; 1: fix rho at value below
                         'nparK': 0,  # rate-class models. 1:rK, 2:rK&fK, 3:rK&MK(1/K), 4:rK&MK
                         'nhomo': 1,  # 0 & 1: homogeneous, 2: kappa for branches, 3: N1, 4: N2, 5: user
                         'getSE': 0,  # 0: don't want them, 1: want S.E.s of estimates
                         'RateAncestor': 0,  # (0,1,2): rates (alpha>0) or ancestral states
                         'Small_Diff': 7e-6,
                         'cleandata': 1,  # remove sites with ambiguity data (1:yes, 0:no)?
                         'icode': None,  # (RateAncestor=1 for coding genes, "GC" in data)
                         'fix_blength': None,  # 0: ignore, -1: random, 1: initial, 2: fixed
                         'method': 1}  # 0: simultaneous; 1: one branch at a time

        self.command = paml_command
        if ctl_file is not None:
            self.read_ctl_file(ctl_file)
        self.del_tmp_dir = del_tmp_dir

        self.alignment = os.path.join(self.working_dir, 'tmp_aln.phy')
        self.tree = os.path.join(self.working_dir, 'tmp_tree.tree')

    def _create_tmp_dir(self):
        '''Create tmporary dir for PAML baseml.'''
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

    def _set_msa(self, msa=None):
        '''Set multiple sequence alignment file.'''
        self._create_tmp_dir()
        aln_write(msa, self.alignment)

    def _set_tree(self, tree_obj=None):
        '''Set tree file.'''
        self._create_tmp_dir()
        if tree_obj is None:
            open(self.tree, 'a').close()
        else:
            TreeIO.write(tree_obj, self.tree, 'newick')

    def _set_out_file(self, out_file=None):
        if out_file is None:
            self._create_tmp_dir()
            self.out_file = os.path.join(self.working_dir, 'tmp.out')
        else:
            self.out_file = out_file

    def estimate_parameter(self, msa=None, tree=None, out_file=None,
                           model=4, rate_ancestor=1):
        '''Return log_LK, tree -> with branch length.
        '''
        self.set_options(runmode=0) # estimate parameter

        self._set_msa(msa)
        self._set_tree(tree)
        self._set_out_file(out_file)
        self.set_options(model=model)
        # set RateAncestor=1 for ancestral sequence reconstruction
        self.set_options(RateAncestor=rate_ancestor)

        mle_result = self.run(command=self.command, verbose=False)
        # TODO use different format for TreeIO.read
        return mle_result['lnL'], TreeIO.read(StringIO(mle_result['tree']), 'newick')

    def __del__(self):
        if self.del_tmp_dir and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
