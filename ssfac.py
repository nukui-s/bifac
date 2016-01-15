"""Semi-Supervised Matrix Factorization on TensorFlow

Authoers: NUKUI Shun
License: GNU General Public License Version 2

"""

from __future__ import division
import pandas as pd
from itertools import chain
import numpy as np
import tensorflow as tf
from clippedgrad import ClippedAdagradOptimizer

class SSFac(object):
    """Class for NMF-based Community Detection for Unipartite Networks
        with pair-wise constraints
    """

    def __init__(self, edge_list, K, weights=None, constraints=None,
                learning_rate=0.1, lambda_c = 1.0, threads=8):
        self.num_node = max(chain.from_iterable(edge_list)) + 1
        self.edge_list = edge_list
        if weights:
            self.weights = np.array(weights, dtype=np.float32)
        else:
            self.weights = np.ones(len(edge_list), dtype=np.float32)
        self.sum_weight = np.sum(self.weights)
        self.K = K
        self.constraints = constraints
        if constraints:
            self.constraint_pairs = cp = [(i, j) for i, j, w in constraints]
            self.constraint_weights = np.array([w for i, j, w in constraints],
                                                dtype=np.float32)
        self.lambda_c = lambda_c
        self.threads = threads
        self.learning_rate = learning_rate
        self._setup_graph()

    def _setup_graph(self):
        N = self.num_node
        weights = self.weights
        sum_weight = self.sum_weight
        K = self.K
        lambda_c = self.lambda_c
        edge_list = self.edge_list
        rev_edge_list = [(n2, n1) for n1, n2 in edge_list]
        edge_list = edge_list + rev_edge_list
        weights = np.append(weights, weights)
        lr = self.learning_rate

        self._graph = tf.Graph()
        with self._graph.as_default():

            with tf.name_scope("X"):
                self.X = X = tf.sparse_to_dense(output_shape=[N,N],
                                                sparse_values=weights,
                                                sparse_indices=edge_list)

            with tf.name_scope("D"):
                self.D = D = tf.diag(tf.reduce_sum(X, reduction_indices=1))

            u_mean = tf.to_float(tf.sqrt(2.0*sum_weight) / N * K)
            initializer_u = tf.random_uniform_initializer(minval=0.0,
                                                          maxval=2*u_mean)
            self.U = U = tf.get_variable(name="U", shape=[N, K],
                                        initializer=initializer_u)
            tf.histogram_summary("U", U)

            if self.constraints:
                c_pairs = self.constraint_pairs
                c_pairs = c_pairs + [(n2, n1) for n1, n2 in c_pairs]
                c_weights = self.constraint_weights
                c_weights = np.append(c_weights, c_weights)
            else:
                c_pairs = [(0, 0)]
                c_weights = [0.0]
            I_N = tf.diag(tf.ones([N]), name="I_N")
            with tf.name_scope("C"):
                #must-link is positive and cannot-link is negative
                self.C_ = C_ = I_N + tf.sparse_to_dense(output_shape=[N, N],
                                              sparse_indices=c_pairs,
                                              sparse_values=c_weights)
                C_suminv = tf.diag(1.0 / tf.reduce_sum(C_,
                                                       reduction_indices=1))
                self.C = C = tf.matmul(C_suminv, C_,
                                       a_is_sparse=True, b_is_sparse=True)
            with tf.name_scope("B"):
                self.B = B = I_N - C

            #Ht = tf.transpose(H)
            #XHt = tf.matmul(X, Ht, a_is_sparse=True)
            #BtB = tf.matmul(B, B, a_is_sparse=True, b_is_sparse=True,
            #                transpose_a=True)
            #self.BtBU = BtBU = tf.matmul(BtB, U, a_is_sparse=True)
            #HtH = tf.matmul(H, Ht)
            #UHHt = tf.matmul(U, HtH)
            #U_new = (XHt / (UHHt + lambda_c*BtBU)) * U
            #self.update_U = U.assign(U_new)

            #Ut = tf.transpose(U)
            #UtX = tf.matmul(Ut, X, b_is_sparse=True)
            #UtU = tf.matmul(Ut, U)
            #tUH = tf.matmul(UtU, H)
            #HBtB = tf.matmul(H, BtB, b_is_sparse=True)
            #H_new = (UtX / (UtUH + lambda_c*HBtB)) * H
            #self.update_H = H.assign(H_new)

            #UH = tf.matmul(U, H)
            Y = tf.matmul(U, U, transpose_b=True)
            self.error = err = tf.nn.l2_loss(X - Y) / N

            self.cnst_err = cnst_err = \
                            tf.nn.l2_loss(tf.matmul(B, U, a_is_sparse=True))
            self.total_cost = total_cost = err + lambda_c*cnst_err
            self.optimize = ClippedAdagradOptimizer(lr).minimize(total_cost)
            tf.scalar_summary("error", err)
            tf.scalar_summary("constraint_err", cnst_err)
            tf.scalar_summary("total_cost", self.total_cost)
            self.summary = tf.merge_all_summaries()

            self.init_op = tf.initialize_all_variables()
            config = tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                    intra_op_parallelism_threads=self.threads)
            self.sess = tf.Session(config=config)

    def run_mu(self, iter_max=100, logdir=None, stop_threshold=0.1):
        sess = self.sess
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        else:
            writer = None
        sess.run(self.init_op)
        pre_cost = 0
        for i in range(iter_max):
            sess.run(self.update_U)
            sess.run(self.update_H)
            cost, sm = sess.run([self.total_cost, self.summary])
            print("cost:",cost)
            if writer:
                writer.add_summary(sm, i)
            if np.abs(pre_cost - cost) < stop_threshold:
                break
            pre_cost = cost
        U = sess.run(self.U)
        return U

    def run(self, iter_max=100, logdir=None, stop_threshold=0.01):
        sess = self.sess
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        else:
            writer = None
        sess.run(self.init_op)
        pre_cost = 0
        for i in range(iter_max):
            cost, sm, _ = sess.run([self.total_cost, self.summary,
                                    self.optimize])
            if writer:
                writer.add_summary(sm, i)
            if np.abs(pre_cost - cost) < stop_threshold:
                break
            pre_cost = cost
        print("cost:",cost)
        U = sess.run(self.U)
        return U

    def get_hard_communities(self):
        sess = self.sess
        #z = sess.run(self.z)
        U = sess.run(tf.nn.relu(self.U))
        U = U / U.sum(axis=0)
        #zU = U * z
        zU = U
        soft_com = zU.T / zU.sum(axis=1)
        hard_com = soft_com.argmax(axis=0)
        return hard_com

def get_constarints(ans, npairs, strength):
    nn = len(ans)
    cst = []
    while True:
        while(1):
            n1 = np.random.randint(0,nn)
            n2 = np.random.randint(0,nn)
            if n1 != n2: break
        c1 = ans[n1]
        c2 = ans[n2]
        if c1 == c2:
            cst.append((n1,n2,strength))
        if len(cst) == npairs: break
    return cst

def get_constarints_all(ans, nn, npairs, strength):
    import random
    from itertools import combinations
    nodes = list(range(len(ans)))
    random.shuffle(nodes)
    cst_nodes = nodes[:nn]
    cst = []
    for n1, n2 in combinations(cst_nodes, 2):
        c1 = ans[n1]
        c2 = ans[n2]
        if c1 == c2:
            cst.append((n1,n2,strength))
    random.shuffle(cst)
    return cst[:npairs]

def check_satisfy(com, cst, U, verbose=False):
    stf = 0
    nn = len(cst)
    for n1, n2, c in cst:
        c1 = com[n1]
        c2 = com[n2]
        if (c1==c2)^(c<0):
            stf += 1
        if verbose:
            print((c1==c2),U[n1],U[n2])
    print(stf,nn)

def get_candidates(com, ans, npairs=100):
    N = len(ans)
    import random
    cand = []
    while True:
        n1 = random.randint(0, N-1)
        n2 = random.randint(0, N-1)
        c1_1 = com[n1]
        c1_2 = com[n2]
        c2_1 = ans[n1]
        c2_2 = ans[n2]
        if (c1_1 != c1_2)and(c2_1 == c2_2):
            cand.append((n1,n2))
            if len(cand) >= npairs:
                break
    return cand

if __name__ == '__main__':
    import os
    import time
    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score
    os.system("rm -rf logkarate")
    #elist = pd.read_pickle("data/karate.pkl")
    #ans = pd.read_pickle("data/karate_com.pkl")
    elist, ans = pd.read_pickle("data/dolphin.pkl")
    pairs = pd.read_pickle("pair_candidates_dol.pkl")
    C = [(n1,n2,10.0) for n1, n2 in pairs][:20]
    #C = get_constarints_all(ans,50,10,10.0)
    model = SSFac(elist, 2, constraints=C, lambda_c=1.0)
    start = time.time()
    U = model.run(iter_max=2000, logdir="logkarate", stop_threshold=0.001)
    end = time.time()
    com = model.get_hard_communities()
    nmi = normalized_mutual_info_score(com, ans)
    check_satisfy(com, C, U)
    print("NMI",nmi)
    print("time", end-start)
    #print(U)
    print("ans",np.array(ans))
    print("res",com)
