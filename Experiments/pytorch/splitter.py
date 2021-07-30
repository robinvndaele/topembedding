import numpy as np
from scipy import sparse

from sklearn.metrics import roc_auc_score


def from_csr_matrix_to_edgelist(csr_A):
    csr_A = sparse.csr_matrix(csr_A)
    t_list = csr_A.indices
    h_list = np.zeros_like(t_list).astype(int)
    for i in range(csr_A.shape[0]):
        h_list[csr_A.indptr[i]:csr_A.indptr[i+1]] = i
    return np.vstack((h_list, t_list)).T

def split_mst(A):
    mst_A = sparse.csgraph.minimum_spanning_tree(A)

    mst_A = (mst_A + mst_A.T).astype(bool)
    rest_A = A - mst_A

    mst_E = from_csr_matrix_to_edgelist(sparse.triu(mst_A, 1))
    rest_E = from_csr_matrix_to_edgelist(sparse.triu(rest_A, 1))
    np.random.shuffle(rest_E)
    return mst_E, rest_E

def sample_neg_edges(A, n_edges):
    n_nodes = A.shape[0]
    portion = 1.5
    while True:
        sample_E = np.random.randint(n_nodes, size=(int(portion*n_edges), 2))
        sample_A  = sparse.csr_matrix((np.ones(len(sample_E)),
                                      (sample_E[:,0], sample_E[:,1])),
                                      shape=(n_nodes,n_nodes)).astype(bool)
        neg_A = sparse.triu(sample_A.astype(int) - A.astype(int), 1) > 0
        if np.sum(neg_A) > n_edges:
            break
        else:
            portion += 0.5
    neg_E = from_csr_matrix_to_edgelist(neg_A)
    np.random.shuffle(neg_E)
    return neg_E[:n_edges]

def split_pos_edges(A, cut_off):
    E_a, E_b = split_mst(A)
    split_E = np.vstack((E_a, E_b[:cut_off-len(E_a)]))
    if cut_off < len(E_a):
        raise ValueError("cut_off < len(E_a)")
    rest_E = E_b[cut_off-len(E_a):]
    return split_E, rest_E

def split_neg_edges(A, cut_off):
    E = sample_neg_edges(A, int(np.sum(A)/2))
    return E[:cut_off], E[cut_off:]

def split_edges(A, cut_off):
    pos_E_a, pos_E_b = split_pos_edges(A, cut_off)
    neg_E_a, neg_E_b = split_neg_edges(A, cut_off)
    return pos_E_a, neg_E_a, pos_E_b, neg_E_b

def construct_adj_matrix(E, n):
    E = np.array(E)
    A  = sparse.csr_matrix((np.ones(len(E)), (E[:,0], E[:,1])),
                           shape=(n,n)).astype(bool)
    A = (A + A.T).astype(bool)
    return A

def label_edges(E, label):
    return np.hstack((E, np.tile(label, (len(E),1))))

def compute_tr_val_split(A, portion):
    cutt_off = int(np.sum(sparse.triu(A, 1))*portion)
    tr_pos_E, tr_neg_E, val_pos_E, val_neg_E = split_edges(A, cutt_off)

    tr_A = construct_adj_matrix(tr_pos_E, A.shape[0])
    tr_E = np.vstack((label_edges(tr_pos_E, 1),
                      label_edges(tr_neg_E, 0)))
    val_E = np.vstack((label_edges(val_pos_E, 1),
                      label_edges(val_neg_E, 0)))
    return tr_A, tr_E, val_E