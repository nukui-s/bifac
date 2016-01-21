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


            with tf.name_scope("l2_loss"):
                diag_z = tf.diag(z)
                Ut = tf.transpose(U)
                Y = tf.matmul(U, tf.matmul(diag_z, Ut))
                l2_loss = tf.reduce_sum(-X * tf.log(Y+10e-12) + Y) / N /N
                self.l2_loss = l2_loss
                tf.scalar_summary("l2_loss", l2_loss)

            #weight for soft constraint of sum(U_i) = 1
            self.lambda_u = lambda_u = tf.placeholder("float", name="lambda_u")
            with tf.name_scope("normalize_U"):
                #norlization term for U w.r.t column
                U_colsum = tf.reduce_sum(U, reduction_indices=[0])
                u_normalization = tf.nn.l2_loss(U_colsum - 1.0)
                tf.scalar_summary("U_normalization", u_normalization)

            with tf.name_scope("cost_func"):
                self.cost = cost = l2_loss + lambda_u*u_normalization

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


    def run(self, iter_max=100,lambda_step=0.1,
            stop_threshold=0.1, max_lambda=1.0, logdir=None,
            lambda_c_step=0.0005, max_lambda_c=0.01):
        sess = self._sess
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        else:
            writer = None
        pre_loss = np.infty
        lambda_u = 0.0
        lambda_c = lambda_c_step
        for i in range(iter_max):
            feed_dict = {self.lambda_u: lambda_u}
            loss, sm, _ = sess.run([self.cost,
                                    self.summary,
                                    self.opt],
                                    feed_dict=feed_dict)
            if writer:
                writer.add_summary(sm, i)
            if i > 2000:
                lambda_u += 1
            if np.abs(pre_loss - loss) < stop_threshold:
                lambda_c = lambda_c_step
            pre_loss = loss
            if lambda_u < max_lambda:
                lambda_u += lambda_step
            #if lambda_c < max_lambda_c:
            #    lambda_c += lambda_c_step
        U = sess.run(self.U)
        z = sess.run(self.z)
        print("loss",loss)
        return z, U

    def get_hard_communities(self):
        sess = self._sess
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
    for _ in range(npairs):
        while(1):
            n1 = np.random.randint(0,nn)
            n2 = np.random.randint(0,nn)
            if n1 != n2: break
        c1 = ans[n1]
        c2 = ans[n2]
        if c1 == c2:
            cst.append((n1,n2,strength))
        else:
            cst.append((n1,n2,-strength))
    return cst

def get_constarints_all(ans, nn, strength):
    import random
    from itertools import combinations
    nodes = list(range(nn))
    random.shuffle(nodes)
    cst_nodes = nodes[:nn]
    cst = []
    for n1, n2 in combinations(cst_nodes, 2):
        c1 = ans[n1]
        c2 = ans[n2]
        if c1 == c2:
            cst.append((n1,n2,strength))
        else:
            cst.append((n1,n2,-strength))
    return cst

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

if __name__ == '__main__':
    import os
    import time
    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score
    os.system("rm -rf logkarate")
    #elist = pd.read_pickle("data/karate.pkl")
    #ans = pd.read_pickle("data/karate_com.pkl")
    elist, ans = pd.read_pickle("data/dolphin.pkl")
    C = get_constarints(ans,30,1.0)
    model = ComFac(elist, 2, learning_rate=1.0, threads=8, constraints=C)
    start = time.time()
    z, U = model.run(logdir="logkarate",stop_threshold=10e-12,
                    lambda_step=0, iter_max=5000, max_lambda=1000,
                    max_lambda_c=10000, lambda_c_step=0.0)
    end = time.time()
    #Y = np.matmul(U, np.matmul(np.diag(z),U.T))
    U_sum = U.sum(axis=0)
    com = model.get_hard_communities()
    nmi = normalized_mutual_info_score(com, ans)
    print("NMI",nmi)
    print("U_sum",U_sum)
    print("time", end-start)
    #print(U)
    print("ans",np.array(ans))
    print("res",com)
