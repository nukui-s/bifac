"""BiFAC on TensorFlow

Authoers: NUKUI Shun
License: GNU General Public License Version 2

"""

from __future__ import division
import pandas as pd
from itertools import chain
import numpy as np
import tensorflow as tf
from clippedgrad import ClippedAdagradOptimizer
from clippedgrad import ClippedGDOptimizer

class ComFac(object):
    """Class for NMF-based Community Detection for Unipartite Networks
        with pair-wise constraints
    """

    def __init__(self, edge_list, K, weights=None, constraints=None,
                learning_rate=0.1, threads=8):
        self.num_node = max(chain.from_iterable(edge_list)) + 1
        self.edge_list = edge_list
        if weights:
            self.weights = np.array(weights, dtype=np.float32)
        else:
            self.weights = np.ones(len(edge_list), dtype=np.float32)
        self.sum_weight = np.sum(self.weights)
        self.learning_rate = learning_rate
        self.K = K
        self.constraints = constraints
        if constraints:
            self.constraint_pairs = cp = [(i, j) for i, j, w in constraints]
            self.constraint_nodes = list(set(chain.from_iterable(cp)))
            self.constraint_weights = np.array([w for i, j, w in constraints],
                                              dtype=np.float32)
        self.threads = threads
        self._setup_graph()

    def _setup_graph(self):
        N = self.num_node
        weights = self.weights
        sum_weight = self.sum_weight
        lr = self.learning_rate
        K = self.K
        edge_list = self.edge_list
        e1_list = [e1 for e1, e2 in edge_list]
        e2_list = [e2 for e1, e2 in edge_list]

        self._graph = tf.Graph()
        with self._graph.as_default():

            with tf.name_scope("X"):
                self.X = X = tf.sparse_to_dense(output_shape=[N,N],
                                                sparse_values=weights,
                                                sparse_indices=edge_list)

            initializer_z = tf.truncated_normal_initializer(
                                                        mean=sum_weight/K,
                                                        stddev=1.0)
            self.z = z = tf.get_variable(name="z", shape=[K],
                                        initializer=initializer_z)

            initializer_u = tf.random_uniform_initializer(minval=0.0,
                                                          maxval=2/N)
            self.U = U = tf.get_variable(name="U", shape=[N, K],
                                        initializer=initializer_u)

            tf.histogram_summary("z", z)
            tf.histogram_summary("U", U)

            #weight for soft constraint of sum(U_i) = 1
            self.lambda_u = lambda_u = tf.placeholder("float", name="lambda_u")

            with tf.name_scope("l2_loss"):
                diag_z = tf.diag(z)
                Ut = tf.transpose(U)
                Y = tf.matmul(U, tf.matmul(diag_z, Ut))
                self.l2_loss = l2_loss = tf.nn.l2_loss(X - Y) / (N*N)
                tf.scalar_summary("l2_loss", l2_loss)

            with tf.name_scope("normalize_U"):
                #norlization term for U w.r.t column
                U_colsum = tf.reduce_sum(U, reduction_indices=[0])
                u_normlization = tf.nn.l2_loss(U_colsum - 1.0)
                tf.scalar_summary("U_normalization", u_normlization)


            pair_constraints = 0
            if self.constraints:
                pairs = self.constraint_pairs
                nodes = self.constraint_nodes
                nn = len(nodes)
                c_weights = self.constraint_weights
                indmap = {nid: ind for ind, nid in enumerate(nodes)}
                maped_pairs = [(indmap[i], indmap[j]) for i, j in pairs]
                with tf.name_scope("C"):
                    #must-link is positive and cannot-link is negative
                    self.C = C = tf.sparse_to_dense(output_shape=[nn, nn],
                                                    sparse_indices=maped_pairs,
                                                    sparse_values=c_weights)
                #extract a subset of U with constraint nodes
                self.U_c = U_c = tf.gather(U, indices=nodes,
                                          name="U_constraint")
                with tf.name_scope("pair_constraints"):
                    Uc_T = tf.transpose(U_c)
                    #normalize U w.r.t row
                    Uc_sum = tf.reduce_sum(U_c, reduction_indices=1)
                    Uc_n = Uc_T / Uc_sum
                    ones = tf.ones(shape=[nn,1])
                    for k in range(K):
                        Uc_k = tf.reshape(Uc_n[k,:], shape=[1,nn])
                        #deprecate vectors like [1,2,3]->[[1,2,3],[1,2,3]]
                        Uc_dep = tf.matmul(ones, Uc_k)
                        Uc_dep_T = tf.transpose(Uc_dep)
                        Uc_dif = Uc_dep - Uc_dep_T
                        pair_constraints += tf.reduce_sum(C*tf.pow(Uc_dif, 2))
                    pair_constraints /= len(pairs)
            self.pair_constraints = pair_constraints

            with tf.name_scope("cost_func"):
                self.cost = cost = l2_loss + \
                                   lambda_u*u_normlization + \
                                   pair_constraints

            self.opt = ClippedAdagradOptimizer(lr).minimize(cost)
            #self.opt = ClippedGDOptimizer(lr).minimize(cost)
            #self.opt = tf.train.AdagradOptimizer(lr).minimize(cost)



            tf.scalar_summary("cost", cost)
            self.summary = tf.merge_all_summaries()

            init_op = tf.initialize_all_variables()
            config = tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                    intra_op_parallelism_threads=self.threads)
            self._sess = tf.Session(config=config)
            self._sess.run(init_op)

    def run(self, iter_max=100, init_lambda=0, lambda_step=0.1,
            stop_threshold=0.1, max_lambda=1.0, logdir=None):
        sess = self._sess
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        else:
            writer = None
        pre_loss = np.infty
        lambda_u = init_lambda
        for i in range(iter_max):
            feed_dict = {self.lambda_u: lambda_u}
            loss, pc, sm, _ = sess.run([self.cost,
                                    self.pair_constraints,
                                    self.summary,
                                    self.opt],
                                    feed_dict=feed_dict)
            print(pc)
            if writer:
                writer.add_summary(sm, i)
            if np.abs(pre_loss - loss) < stop_threshold:
                break
            pre_loss = loss
            if lambda_u < max_lambda:
                lambda_u += lambda_step
        U = sess.run(self.U)
        z = sess.run(self.z)
        print("loss",loss)
        return z, U

    def get_hard_communities(self):
        sess = self._sess
        z = sess.run(self.z)
        U = sess.run(tf.nn.relu(self.U))
        U = U / U.sum(axis=0)
        zU = U * z
        soft_com = zU.T / zU.sum(axis=1)
        hard_com = soft_com.argmax(axis=0)
        return hard_com


if __name__ == '__main__':
    import os
    import time
    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score
    os.system("rm -rf logkarate")
    elist = pd.read_pickle("karate.pkl")
    C = [(1,2,-0.001)]
    model = ComFac(elist, 2, learning_rate=0.1, threads=2, constraints=C)
    start = time.time()
    z, U = model.run(logdir="logkarate",stop_threshold=10e-8,
                    lambda_step=0.01, iter_max=500, max_lambda=0.3)
    end = time.time()
    Y = np.matmul(U, np.matmul(np.diag(z),U.T))
    U_sum = U.sum(axis=0)
    com = model.get_hard_communities()
    ans = pd.read_pickle("karate_com.pkl")
    nmi = normalized_mutual_info_score(com, ans)
    print("NMI",nmi)
    print("U_sum",U_sum)
    print("time", end-start)
    print(U)
    print(com)
