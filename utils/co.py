import copy
import math
import numpy as np
import itertools
import gurobipy as gp

from phylo import Clade, Tree, TreeIO
# from seq_embedding import preprocess_seq

import torch
import torch.nn.functional as F
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from sklearn.metrics.pairwise import cosine_distances


def k_mer_dist(v1, v2):
    return np.linalg.norm(v1 - v2)  # Euclidean distance
    # return 1.0 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # cosine distance

def delta(vecs, co, i):
    length = len(co)
    return abs(k_mer_dist(vecs[co[(i-1)%length]], vecs[co[i]]) +
               k_mer_dist(vecs[co[(i+1)%length]], vecs[co[(i+2)%length]]) -
               k_mer_dist(vecs[co[(i-1)%length]], vecs[co[(i+1)%length]]) -
               k_mer_dist(vecs[co[i]], vecs[co[(i+2)%length]]))


def cluster_co2tree(kmer_vecs, circular_order):
    '''Given k-mer vectors and circular order, calculate the phylogenetic tree.
    param:
        kmer_vecs: (phy_size, kmer_dim) -> list of numpy.array
        circular_order: (phy_size) -> list
    '''
    # assert len(kmer_vecs) == len(circular_order), 'size of vectors and co must be equal'

    co = circular_order

    # delta(s_i) = |d(s_{i-1}, s_{i}) + d(s_{i+1},s_{i+2}) - d(s_{i-1}, s_{i+1})
    #   - d(s_i, s_{i+2})|
    d = [[delta(kmer_vecs, co, i), Clade(name=str(co[i]+1))]
         for i in range(len(co))]
    # print(co)
    length = len(co)
    for i in range(len(co)-3):
        # print('length of co:', length)
        min_i = d.index(min(d, key=lambda item: item[0]))
        # print('min_i:', min_i)
        inter_clade = Clade(clades=[d[min_i][1], d[(min_i+1)%length][1]])

        del co[(min_i+1)%length]
        del d[(min_i+1)%length]
        length -= 1
        if min_i == length:
            min_i -= 1

        # print(co)

        d[(min_i-1)%length][0] = delta(kmer_vecs, co, (min_i-1)%length)
        d[min_i] = [delta(kmer_vecs, co, min_i), inter_clade]
        d[(min_i+1)%length][0] = delta(kmer_vecs, co, (min_i+1)%length)

    # combine last three clades
    root = Clade(clades=[d[0][1], d[1][1], d[2][1]])
    tree = Tree(root=root, rooted=False)
    return tree


def point_to_path_dist(x, y, z):
    return (k_mer_dist(x, y) + k_mer_dist(x, z) - k_mer_dist(y, z)) / 2.0


def co2tree(vecs, circular_order):
    # assert len(vecs) == len(circular_order), 'size of vectors and co must be equal'

    co = circular_order
    
    # initialization: add the first two items to tree, rooted at co[0]
    root = Clade(name=str(co[0]+1))
    clade_k = Clade(name=str(co[1]+1), branch_length=k_mer_dist(vecs[co[0]], vecs[co[1]]))
    root.clades = [clade_k]
    tree = Tree(root=root, rooted=False)

    for k in range(1, len(co)-1):
        clade_k_next = Clade(name=str(co[k+1]+1))
        path_k_to_1 = tree.get_path(clade_k)[::-1] + [tree.root]
        delta_k_next = point_to_path_dist(vecs[co[k]], vecs[co[0]], vecs[co[k+1]])

        if delta_k_next < 0:
            raise ValueError('delta_k_next < 0')

        s = 0.0
        i = -1
        while s < delta_k_next and i < len(path_k_to_1)-2:
            i += 1
            s += path_k_to_1[i].branch_length
            # if s > delta_k_next:
            #     break

        if s > delta_k_next:
            clade_k_next.branch_length = point_to_path_dist(vecs[co[k+1]], vecs[co[0]], vecs[co[k]])
            path_k_to_1[i].branch_length = delta_k_next - s + path_k_to_1[i].branch_length
            inter_clade = Clade(branch_length=s-delta_k_next, clades=[path_k_to_1[i], clade_k_next])
            path_k_to_1[i+1].clades.remove(path_k_to_1[i])
            path_k_to_1[i+1].clades.append(inter_clade)
        else:
            # i -= 1
            # last_e = (delta_k_next - s + path_k_to_1[i].branch_length)
            # clade_k_next.branch_length = point_to_path_dist(vecs[co[k+1]], vecs[co[0]], vecs[co[k]])
            # path_k_to_1[i].branch_length = last_e / 2.0
            # inter_clade = Clade(branch_length=last_e / 2.0, clades=[path_k_to_1[i], clade_k_next])
            # path_k_to_1[i+1].clades.remove(path_k_to_1[i])
            # path_k_to_1[i+1].clades.append(inter_clade)

            ## S < delta_k_next, the tree will not be a bifurcated tree
            clade_k_next.branch_length = point_to_path_dist(vecs[co[k+1]], vecs[co[0]], vecs[co[k]])
            root.clades.append(clade_k_next)

        clade_k = clade_k_next

    # reroot tree
    if len(root.clades) == 1:
        tree.root = root.clades[0]
        root.clades.clear()
        tree.root.clades.append(root)
    else:
        tree.root.name = None
        tree.root.clades.append(Clade(name=str(co[0]+1), branch_length=0.0))

    return tree


def get_co_len(msa, circular_order):
    '''
    Scoring an circular order.
    param:
        msa: list of string(sequence)
    '''
    co = circular_order
    assert len(msa) > 3
    assert len(msa) == len(co), 'length of msa and circular order must be equal'

    calculator = DistanceCalculator('blastn')

    pa_scores = [calculator._pairwise(msa[co[i]], msa[co[i+1]])
                 for i in range(len(co) - 1)]
    pa_scores.append(calculator._pairwise(msa[co[-1]], msa[co[0]]))

    return sum(pa_scores)


def get_kmer(seq, k_mer=5):
    alphabet = 'ATCG'
    kmer_dict = {''.join(p): 0 for p in itertools.product(alphabet, repeat=k_mer)}
    seq = seq.replace('-', '')

    for i in range(len(seq) - k_mer + 1):
        kmer_dict[seq[i:i+k_mer]] += 1

    vec = np.array(list(kmer_dict.values()), dtype=np.float64)

    return vec / (len(seq) - k_mer + 1)


def get_co_len_vec(msa, circular_order):
    '''
    Scoring an circular order.
    param:
        msa: list of string(sequence)
    '''
    co = torch.tensor(circular_order, dtype=torch.long)
    assert len(msa) > 3
    assert len(msa) == len(co), 'length of msa and circular order must be equal'

    vectors = torch.tensor([get_kmer(seq, k_mer=6) for seq in msa])

    # Gather vectors in order of tour
    d = vectors.gather(0, co.unsqueeze(-1).expand_as(vectors))

    # Calculate the cosine similarity
    cos_sim = torch.cat((F.cosine_similarity(d[1:], d[:-1], dim=1),
                        F.cosine_similarity(d[-1], d[0], dim=0).unsqueeze(-1)), dim=0)

    # Calculate the cosine distance
    total_length = (1.0 - cos_sim) / 2.0

    # print(pa_scores)

    return total_length.sum().item()


def gurobi_tsp(vecs):
    '''
    calculate circular order using gurobi solver.
    param:
        vecs: list of numpy.array
    return:
        tour -> list: circular order
    '''
    # vecs = [get_kmer(seq, k_mer) for seq in seqs]
    n = len(vecs)

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(gp.quicksum(model._vars[i, j]
                                    for i, j in itertools.combinations(tour, 2))
                            <= len(tour) - 1)
                    
    # Given a tuplelist of edges, find the shortest subtour
    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)
        while unvisited: # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for _, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of distance between each pair of seq
    dist = {(i, j): k_mer_dist(vecs[i], vecs[j])
            for i in range(n) for j in range(i)}
    # print(len(dist))
    # print(dist)

    m = gp.Model()
    m.Params.outputFlag = False

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=gp.GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j] # edge in opposite direction

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))

    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)

    # Retrieve solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return tour