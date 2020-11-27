import os
import random
import numpy as np
import subprocess
from tempfile import TemporaryDirectory
from Bio.Align import MultipleSeqAlignment
from Bio.SeqIO.FastaIO import SimpleFastaParser
import dendropy
from dendropy.calculate import treecompare
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from phylo import TreeIO
from utils import PAMLml, get_kmer, torch_load_cpu, gurobi_tsp, co2tree, cluster_co2tree
from nets.models import AttentionModel


def create_benchmark_dataset(seq_file: str, trainset_dir: str, testset_dir: int,
                             num_testset=10, size_testset=10, size_phy=15):
    num_subset = size_testset * size_phy

    with open(seq_file, 'rt') as f_read:
        items = list(SimpleFastaParser(f_read))

    random.shuffle(items)

    for i in range(num_testset):
        seqs_dir = os.path.join(testset_dir, str(i+1))
        if not os.path.exists(seqs_dir):
            os.makedirs(seqs_dir)
        sub_items = items[i*num_subset:(i+1)*num_subset]
        for j in range(size_testset):
            with open(os.path.join(seqs_dir, '{}.fasta'.format(j+1)), 'wt') as f_write:
                for k in range(j*size_phy, (j+1)*size_phy):
                    f_write.write('>{}\n{}\n'.format(sub_items[k][0], sub_items[k][1]))

    if not os.path.exists(trainset_dir):
        os.makedirs(trainset_dir)
    with open(os.path.join(trainset_dir, 'train.fasta'), 'wt') as f_write:
        for des, seq in items[num_subset*num_testset:]:
            f_write.write('>{}\n{}\n'.format(des, seq))


def infer_kmer_vector(seq_file, pickle_file, k_mer=5):
    print('Inferring k-mer vector with k={}'.format(k_mer))
    with open(seq_file, 'rt') as f_read:
        vectors = [get_kmer(seq, k_mer=k_mer) for _, seq in tqdm(SimpleFastaParser(f_read))]
    
    vectors = torch.tensor(vectors, dtype=torch.float)
    torch.save(vectors, pickle_file)


def rename_fasta(testset_dir, num_testset=10, size_testset=10):
    for i in range(num_testset):
        for j in range(size_testset):
            msa_file = os.path.join(testset_dir, str(i+1), '%d.fasta' % (j+1))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            with open(os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1)), 'wt') as f_write:
                for idx, seq in enumerate(seq_list):
                    f_write.write('>{}\n{}\n'.format(idx+1, seq))


def create_Ref_tree(raxml_MPI: str, raxml: str, msa: list, save_path=None):
    with TemporaryDirectory() as dirname:
        with open(os.path.join(dirname, 'msa.fasta'), 'wt') as f_write:
            for i, seq in enumerate(msa):
                f_write.write('>%d\n%s\n' %(i+1, seq))

        try:
            # Rapid bootstrap with GTRGAMMA model, number of RUN: autoMR
            out_bytes = subprocess.run([raxml_MPI,
                                        '-m', 'GTRGAMMA',
                                        '-p', '12345',
                                        '-x', '12345',
                                        '-N', 'autoMR',
                                        '-s', 'msa.fasta',
                                        '-n', 'boot'],
                                        cwd=dirname,
                                        stdout=subprocess.DEVNULL,
                                        # stderr=subprocess.DEVNULL
                                        )
            
            # Build consensus trees using bootstrap replicates as Reference tree
            out_bytes = subprocess.run([raxml,
                                        '-m', 'GTRGAMMA',
                                        '-J', 'MR',
                                        '-z', 'RAxML_bootstrap.boot',
                                        '-n', 'consenus'],
                                        cwd=dirname,
                                        stdout=subprocess.DEVNULL,
                                        # stderr=subprocess.DEVNULL
                                        )
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit(-1)

        # return Reference tree
        ref_tree = os.path.join(dirname, 'RAxML_MajorityRuleConsensusTree.consenus')
        return TreeIO.read(ref_tree, 'newick')


def create_Ref_of_testset(raxml_MPI, raxml, testset_dir, num_testset, size_testset):
    for i in range(num_testset):
        for j in range(size_testset):
            print('Test[%d],Num[%d]: creating refernce tree...' %(i+1, j+1))
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            tree = create_Ref_tree(raxml_MPI, raxml, seq_list)
            # print(tree)
            tree_file = os.path.join(testset_dir, str(i+1), '%d_ref.nwk' % (j+1))
            TreeIO.write([tree], tree_file, 'newick')


def megacc(megacc_cmd, method_type, mao_file, testset_dir, num_testset, size_testset):
    for i in range(num_testset):
        print('testset[{}] Inferring tree by megacc ...'.format(i+1))
        for j in range(size_testset):
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            out_file = os.path.join(testset_dir, str(i+1), '%d_%s.nwk' %(j+1, method_type))

            out_bytes = subprocess.run([megacc_cmd,
                                        '-a', mao_file,
                                        '-d', msa_file,
                                        '-o', out_file,
                                        '-n',
                                        '-s'],
                                        stdout=subprocess.DEVNULL,
                                        # stderr=subprocess.DEVNULL
                                       )


def megacc_dist(megacc_cmd, method_type, mao_file, testset_dir, num_testset, size_testset):
    for i in range(num_testset):
        print('testset[{}] Inferring tree from distance matrix ...'.format(i+1))
        for j in range(size_testset):
            dist_file = os.path.join(testset_dir, str(i+1), '%d_dist.meg' % (j+1))
            out_file = os.path.join(testset_dir, str(i+1), '%d_dist_%s.nwk' % (j+1, method_type))

            out_bytes = subprocess.run([megacc_cmd,
                                        '-a', mao_file,
                                        '-d', dist_file,
                                        '-o', out_file,
                                        '-n',
                                        '-s'],
                                        stdout=subprocess.DEVNULL,
                                        # stderr=subprocess.DEVNULL
                                       )


def infer_tree_rl(model_path, node_dim, embedding_dim, n_encode_layers,
             testset_dir, num_testset, size_testset, k_mer=6, out_filename=''):
    model = AttentionModel(node_dim=node_dim, embedding_dim=embedding_dim, n_encode_layers=n_encode_layers)
    print('  [*] Loading data from {}'.format(model_path))
    load_data = torch_load_cpu(model_path)
    model.load_state_dict(load_data['model'])
    # del load_data
    model.set_decode_type('greedy')

    for i in range(num_testset):
        print('testset[{}] Inferring tree by rl ...'.format(i+1))
        for j in range(size_testset):
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            out_file = os.path.join(testset_dir, str(i+1), '%d_rl%s.nwk' % (j+1, out_filename))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            vecs = [get_kmer(seq, k_mer=k_mer) for seq in seq_list]
            vecs_tensor = torch.tensor(vecs, dtype=torch.float).unsqueeze(0)

            _, _, circular_orders = model(vecs_tensor, return_pi=True)
            co = circular_orders.tolist()[0]

            tree = co2tree(vecs, co)
            # tree = cluster_co2tree(vecs, co)
            TreeIO.write([tree], out_file, 'newick')
            # TreeIO.draw_ascii(tree)


def infer_tree_tsp(testset_dir, num_testset, size_testset, k_mer=6):
    for i in range(num_testset):
        print('testset[{}] Inferring tree by tsp ...'.format(i+1))
        for j in range(size_testset):
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            out_file = os.path.join(testset_dir, str(i+1), '%d_tsp.nwk' % (j+1))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            vecs = [get_kmer(seq, k_mer=k_mer) for seq in seq_list]

            co = gurobi_tsp(vecs)

            tree = cluster_co2tree(vecs, co)
            # tree = co2tree(vecs, co)
            TreeIO.write([tree], out_file, 'newick')


def kmer2distance(testset_dir, num_testset, size_testset, size_phy, k_mer=6):
    for i in range(num_testset):
        print('testset[{}] calculate the distance matrix ...'.format(i+1))
        for j in range(size_testset):
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            out_file = os.path.join(testset_dir, str(i+1), '%d_dist.meg' % (j+1))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            vecs = np.array([get_kmer(seq, k_mer=k_mer) for seq in seq_list])
            dist = euclidean_distances(vecs, vecs)

            with open(out_file, 'wt') as f_write:
                # mega distance format
                f_write.write('#mega\n')
                f_write.write('!TITLE cosine distance matrix of k-mer vectors;\n')
                f_write.write('!Format DataType=distance NTaxa={};\n'.format(size_phy))
                f_write.write('\n')

                f_write.write('\n'.join('#{}'.format(k+1) for k in range(size_phy)))
                f_write.write('\n\n\n')

                # distance matrix
                for r in range(1, size_phy):
                    f_write.write(' '.join(str(a) for a in dist[r, :r]))
                    f_write.write('\n')



def compare_lnlk(paml_path, method_types, testset_dir, num_testset, size_testset, out_filename=None, silent=False):
    if out_filename is None:
        out_filename = 'lnlk.csv'
    paml = PAMLml(paml_command=paml_path, del_tmp_dir=True)
    total_result_file = os.path.join(testset_dir, out_filename)
    total_contents = []

    for i in range(num_testset):
        print('testset[{}] Calculating ln likelihood ...'.format(i+1))

        total_content = []

        result_file = os.path.join(testset_dir, str(i+1), out_filename)
        with open(result_file, 'wt') as f_write:
            for method_type in method_types:
                s = 0
                contents = [method_type]
                for j in range(size_testset):
                    msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
                    tree_file = os.path.join(testset_dir, str(i+1), '%d_%s.nwk' % (j+1, method_type))
                    with open(msa_file, 'rt') as f_read:
                        seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
                    msa = MultipleSeqAlignment([SeqRecord(Seq(seq, generic_dna), name=str(idx+1), id=str(idx+1))
                                                for idx, seq in enumerate(seq_list)])
                    tree = TreeIO.read(tree_file, 'newick')
                    lnlk, _ = paml.estimate_parameter(msa, tree)
                    s += - lnlk
                    contents.append(str(lnlk))
                f_write.write('{}\n'.format(','.join(contents)))
                
                if not silent:
                    # print(', '.join(contents), 'avg: ', s / size_testset)
                    print('{}: {}'.format(method_type, s / size_testset))

                total_content.append('{:.3f}'.format(s / size_testset))

        total_contents.append(total_content)
    
    total_contents = [list(t) for t in zip(*total_contents)]
    with open(total_result_file, 'wt') as f_write:
        for idx, method_type in enumerate(method_types):
            f_write.write('{},{}\n'.format(method_type, ','.join(total_contents[idx])))


def compare_rf_distance(ref, method_types, testset_dir, num_testset, size_testset, size_phy, out_filename=None, silent=False):
    if out_filename is None:
        out_filename = 'RF_dist.csv'
    # establish common taxon namespace
    tns = dendropy.TaxonNamespace()
    total_result_file = os.path.join(testset_dir, out_filename)
    total_contents = []

    for i in range(num_testset):
        print('testset[{}] Calculating RF distance ...'.format(i+1))
        total_content = []
        ref_trees = []
        for j in range(size_testset):
            ref_tree_file = os.path.join(testset_dir, str(i+1), '%d_%s.nwk' % (j+1, ref))
            ref_trees.append(dendropy.Tree.get(path=ref_tree_file, schema='newick',
                                               taxon_namespace=tns))

        result_file = os.path.join(testset_dir, str(i+1), out_filename)
        with open(result_file, 'wt') as f_write:
            f_write.write('reference: {}\n'.format(ref))
            for method_type in method_types:
                s = 0
                contents = [method_type]
                for j in range(size_testset):
                    tree_file = os.path.join(testset_dir, str(i+1), '%d_%s.nwk' % (j+1, method_type))
                    tree = dendropy.Tree.get(path=tree_file, schema='newick', taxon_namespace=tns)
                    rf_dist = treecompare.symmetric_difference(ref_trees[j], tree) / (2 * size_phy - 6)
                    s += rf_dist
                    contents.append(str(rf_dist))
                f_write.write('{}\n'.format(','.join(contents)))

                if not silent:
                    # print(', '.join(contents), 'avg: ', s / size_testset)
                    print('{}: {}'.format(method_type, s / size_testset))
                
                total_content.append('{:.3f}'.format(s / size_testset))

        total_contents.append(total_content)
    
    total_contents = [list(t) for t in zip(*total_contents)]
    with open(total_result_file, 'wt') as f_write:
        for idx, method_type in enumerate(method_types):
            f_write.write('{},{}\n'.format(method_type, ','.join(total_contents[idx])))


def cal_tree_varience(tree, dist, k):
    '''Calculate the proportion of explained variance, as given by a formula of Pruzanski,
     Tversky and Caroll.
     
     see paper: Circular orders of tree metrics, and their uses for the reconstruction 
     and fitting of phylogenetic trees'''
    d = []
    t = []
    for i in range(k - 1):
        for j in range(i + 1, k):
            t.append(dist[i][j])
            start = next(tree.find_clades(terminal=True, name=str(i+1)))
            to = next(tree.find_clades(terminal=True, name=str(j+1)))
            d.append(tree.distance(start, to))
    d = np.array(d)
    t = np.array(t)
    # print(d)
    # print(t)
    # print()
    d_mean = np.mean(d)
    return 100.0 * (1.0 - (np.sum((d - t)**2) / np.sum((d - d_mean)**2)))


def compare_variance(method_types, testset_dir, num_testset, size_testset, size_phy,
                        out_filename=None, k_mer=6, silent=False):
    if out_filename is None:
        out_filename = 'variance.csv'

    total_result_file = os.path.join(testset_dir, out_filename)
    total_results = {method_type: [] for method_type in method_types}

    for i in range(num_testset):
        print('testset[{}] Calculating varience ...'.format(i+1))
        results = {method_type: [] for method_type in method_types}

        for j in range(size_testset):
            # distance matrix
            msa_file = os.path.join(testset_dir, str(i+1), '%d_rn.fasta' % (j+1))
            with open(msa_file, 'rt') as f_read:
                seq_list = [seq for _, seq in SimpleFastaParser(f_read)]
            vecs = np.array([get_kmer(seq, k_mer=k_mer) for seq in seq_list])
            dist = euclidean_distances(vecs, vecs)

            for method_type in method_types:
                tree_file = os.path.join(testset_dir, str(i+1), '%d_%s.nwk' % (j+1, method_type))
                tree = TreeIO.read(tree_file, 'newick')
                # print('i: {},j: {}, method_type: {}'.format(i, j, method_type))
                var = cal_tree_varience(tree, dist, size_phy)
                results[method_type].append(var)
        
        avg = {method_type: np.mean(results[method_type]) for method_type in method_types}
        for method_type in method_types:
            total_results[method_type].append(avg[method_type])

        result_file = os.path.join(testset_dir, str(i+1), out_filename)
        with open(result_file, 'wt') as f_write:
            for method_type in method_types:
                ss = ['{:.2f}'.format(num) for num in results[method_type]]
                f_write.write('{},{}\n'.format(method_type, ','.join(ss)))
        
    with open(total_result_file, 'wt') as f_write:
        for method_type in method_types:
            ss = ['{:.2f}'.format(num) for num in total_results[method_type]]
            f_write.write('{},{}\n'.format(method_type, ','.join(ss)))


if __name__ == '__main__':
    seq_file = '/data/ztj_data/rlphy/data/sel03n_rm.fasta'
    trainset_dir = '/data/ztj_data/rlphy/data/trainset'
    testset_dir = '/data/ztj_data/rlphy/data/testset'
    num_testset=10
    size_testset=10
    size_phy=15
    # create_benchmark_dataset(seq_file, trainset_dir, testset_dir,
    #                          num_testset=num_testset, size_testset=size_testset, size_phy=size_phy)
    # rename_fasta(testset_dir, num_testset=num_testset, size_testset=size_testset)

    ## infer kmer vector
    train_seq_file = '/data/ztj_data/rlphy/data/trainset/train.fasta'
    pickle_file = '/data/ztj_data/rlphy/data/trainset/k_mer_6.pt'
    k_mer = 6
    # infer_kmer_vector(train_seq_file, pickle_file, k_mer=k_mer)


    raxml_MPI = '/data/ztj_data/rlphy/tools/RAxML-8.2.12/bin/raxmlHPC-MPI-AVX2'
    raxml = '/data/ztj_data/rlphy/tools/RAxML-8.2.12/bin/raxmlHPC-AVX2'
    # create_Ref_of_testset(raxml_MPI, raxml, testset_dir, num_testset, size_testset)


    # infer tree by NJ, UPGMA, Minimum Evolution
    megacc_cmd = '/usr/bin/megacc'
    method_types = ['nj', 'upgma', 'me']
    mao_files = [
        '/data/ztj_data/rlphy/infer_NJ_nucleotide.mao',
        '/data/ztj_data/rlphy/infer_UPGMA_nucleotide.mao',
        '/data/ztj_data/rlphy/infer_ME_nucleotide.mao'
    ]
    # method_types = ['me']
    # mao_files = ['/data/ztj_data/rlphy/infer_ME_nucleotide.mao']
    # method_types = ['ml']
    # mao_files = ['/data/ztj_data/rlphy/infer_ML_nucleotide.mao']
    # for method_type, mao_file in zip(method_types, mao_files):
    #     megacc(megacc_cmd, method_type, mao_file, testset_dir, num_testset, size_testset)


    # infer tree by RL
    model_path = '/data/ztj_data/rlphy/kmer6_models/phylo_15/run_20200529T101747/epoch-99.pt'
    node_dim = 4 ** k_mer
    embedding_dim = 128
    n_encode_layers = 2
    # infer_tree_rl(model_path, node_dim, embedding_dim, n_encode_layers,
    #               testset_dir, num_testset, size_testset, k_mer=k_mer)


    # infer tree by tsp
    # infer_tree_tsp(testset_dir, num_testset, size_testset, k_mer=k_mer)


    ## Benchmark 1:  RL vs NJ, UPGMA and ME
    paml_path = '/data/ztj_data/rlphy/tools/paml4.9/bin/baseml'
    # compare_lnlk(paml_path, ['ref', 'rl', 'tsp', 'me', 'nj', 'upgma'], testset_dir, num_testset, size_testset)
    # compare_rf_distance('ref', ['rl', 'tsp', 'me', 'nj', 'upgma'], testset_dir, num_testset, size_testset, size_phy)


    ## Benchmark 2:  RL vs distance matrix based NJ, UPGMA and ME
    # 1. calculate the k-mer distance matrix, and save as mega format.
    # kmer2distance(testset_dir, num_testset, size_testset, size_phy)

    # 2. infer tree by NJ, UPGMA, Minimum Evolution using k-mer distance matrix.
    method_types = ['nj', 'upgma', 'me']
    mao_files = [
        '/data/ztj_data/rlphy/infer_NJ_distances.mao',
        '/data/ztj_data/rlphy/infer_UPGMA_distances.mao',
        '/data/ztj_data/rlphy/infer_ME_distances.mao'
    ]
    # for method_type, mao_file in zip(method_types, mao_files):
    #     megacc_dist(megacc_cmd, method_type, mao_file, testset_dir, num_testset, size_testset)

    # 3. compare nrf distances.
    # compare_rf_distance('ref', ['rl_128_2', 'tsp', 'dist_nj', 'dist_upgma', 'dist_me'], 
    #                     testset_dir, num_testset, size_testset, size_phy, out_filename='RF_dist_2.csv')

    # 4. compare lnlk.
    # compare_lnlk(paml_path, ['ref', 'rl', 'tsp', 'dist_nj', 'dist_upgma', 'dist_me'],
    #              testset_dir, num_testset, size_testset, out_filename='lnlk_2.csv')


    ## Benchmark 3:  results of RL in different parameters(k-mer size, embedding_dim and n_encode_layers)
    # infer tree by RL (kmer == 6)
    node_dim = 4 ** 6
    embedding_dim_list = [128, 256, 512]
    n_encode_layers_list = [2, 3, 4, 5]
    model_filename = [
        'run_20200529T101747',
        'run_20200529T110049',
        'run_20200529T114522',
        'run_20200529T123106',
        'run_20200529T131832',
        'run_20200529T140217',
        'run_20200529T144548',
        'run_20200529T153044',
        'run_20200529T161753',
        'run_20200529T170233',
        'run_20200529T174824',
        'run_20200529T183735',
    ]
    # for i, embedding_dim in enumerate(embedding_dim_list):
    #     for j, n_encode_layers in enumerate(n_encode_layers_list):
    #         model_path = os.path.join('/data/ztj_data/rlphy/kmer6_models/phylo_15/',
    #                                   model_filename[i*4+j], 'epoch-99.pt')
    #         out_filename = '_{}_{}'.format(embedding_dim, n_encode_layers)
    #         infer_tree_rl(model_path, node_dim, embedding_dim, n_encode_layers,
    #                       testset_dir, num_testset, size_testset, k_mer=6, out_filename=out_filename)
    
    ## compare nrf distances of different parameters
    method_types = ['rl_{}_{}'.format(ed, nel) for ed in embedding_dim_list for nel in n_encode_layers_list]
    method_types.extend(['tsp', 'dist_nj', 'dist_upgma', 'dist_me'])
    compare_rf_distance('ref', method_types, testset_dir, num_testset,
                        size_testset, size_phy, out_filename='RF_dist_3.csv')
    
    ## compare lnlk
    compare_lnlk(paml_path, method_types, testset_dir, num_testset,
                 size_testset, out_filename='lnlk_3.csv')


    ## Benchmark 4: Calculate the proportion of explained variance
    # embedding_dim_list = [128, 256, 512]
    # n_encode_layers_list = [2, 3, 4, 5]
    # method_types = ['rl_{}_{}'.format(ed, nel) for ed in embedding_dim_list for nel in n_encode_layers_list]
    # method_types.extend(['dist_nj', 'dist_upgma', 'dist_me'])
    # compare_variance(method_types, testset_dir, num_testset,
    #                  size_testset, size_phy, out_filename='varience.csv')