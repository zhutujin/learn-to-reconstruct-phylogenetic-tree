import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import random
import pickle
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from utils import get_co_len, get_kmer
from seq_embedding.seq2vec import preprocess_seq
from Bio.SeqIO.FastaIO import SimpleFastaParser
from gensim.models.doc2vec import Doc2Vec


class PHYLO(object):

    NAME = 'phylo'

    def __init__(self, seq_file=None, vec_path=None, k_mer=5, seq2vec=None):
        # assert isinstance(seq2vec, Doc2Vec), 'seq2vec must be instance of Doc2Vec'

        # self._seq2vec = seq2vec
        # self.infer_vec = self._seq2vec.infer_vector

        if vec_path is not None:
            self.vectors = torch.load(vec_path)
        elif seq_file is not None:
            with open(seq_file, 'rt') as f_read:
                self.vectors = torch.tensor([get_kmer(seq, k_mer=k_mer)
                                            for _, seq in SimpleFastaParser(f_read)],
                                            dtype=torch.float)
        else:
            raise RuntimeError('vec_path or seq_file is required!')

    @staticmethod
    def get_costs(vectors, pi):
        '''
        vectors: (batch_size, phy_size, node_dim)
        pi: (batch_size, phy_size)'''
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather vectors in order of tour
        d = vectors.gather(1, pi.unsqueeze(-1).expand_as(vectors))

        # k-mer vector distance
        # d = d / 0.25
        dists = torch.cat((torch.norm(d[:, 1:] - d[:,:-1], dim=-1),
                          torch.norm(d[:, -1] - d[:, 0], dim=-1).unsqueeze(-1)), dim=1)
        return dists.sum(dim=-1)


    def make_dataset(self, *args, **kwargs):
        return PHYLODataset(self, *args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePHYLO.initialize(*args, **kwargs)


class PHYLODataset(Dataset):

    def __init__(self, problem, phy_size=20, num_samples=1000):
        '''
        mode: "train" (default) or "validation".'''
        super(PHYLODataset, self).__init__()

        assert phy_size * num_samples <= len(problem.vectors), 'do not have enough datas'

        # sub_idx_seqs = random.sample(list(enumerate(problem.seqs)), phy_size * num_samples)

        # sub_idx, sub_seqs = list(zip(*sub_idx_seqs))
        sub_idx = random.sample(range(len(problem.vectors)), phy_size * num_samples)

        sub_vectors = problem.vectors[torch.tensor(sub_idx)]

        # self.seqs = [sub_seqs[i:i+phy_size]
        #                 for i in range(0, phy_size * num_samples, phy_size)]
        self.vectors = [sub_vectors[i:i+phy_size]
                        for i in range(0, phy_size * num_samples, phy_size)]

        self.size = len(self.vectors)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return {
        #     'vec': self.vectors[idx]
        # }
        return self.vectors[idx]


class StatePHYLO(NamedTuple):
    # Fixed input
    loc: torch.Tensor

    # If this state contains multiple copies for the same instance, then for memory efficiency
    # the loc tensor are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
            )
        return super(StatePHYLO, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StatePHYLO(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def update(self, selected):
        '''
        selected: (batch_size)
        '''

        # Update the state
        # prev_a: (batch_size, 1)
        prev_a = selected[:, None]  # Add dimension for step

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def construct_solutions(self, actions):
        return actions
