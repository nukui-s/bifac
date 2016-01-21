"""BiFAC on TensorFlow

Authoers: NUKUI Shun
License: GNU General Public License Version 2

"""

import os
from itertools import chain
import numpy as np
import tensorflow as tf

class BiFac(object):
    """class for BiFac algorithm

        Matrix-Factorization-based community detection algorithm

        Arguments (edge_list, K)
        -------------------

        edge_list : A list of tuple of nodeID(1st mode), nodeID(2nd mode), weight
                        nodeID must be zero origin

        K : The number of communities

        -------------------
    """

    def __init__(self, edge_list, weights, K, learning_rate=0.01):
        self.edge_list = edge_list
        indices = np.array(edge_list)
        self.n1 = indices[:,0].max() + 1
        self.n2 = indices[:,1].max() + 1
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.array(weights, dtype=np.float32)
        self.setup_graph()

    def setup_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            n1, n2 = self.n1, self.n2
            edge_list = self.edge_list
            edge_list = edge_list
            weights = self.weights
            K = self.K
            lr = self.learning_rate
            self.X = X = tf.sparse_to_dense(output_shape=[n1, n2],
                                            sparse_values=weights,
                                            sparse_indices=edge_list)

            init_phi = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            self.Phi1 = Phi1 = tf.get_variable(name="Phi1", shape=[n1, K],
                                               initializer=init_phi)
            self.Phi2 = Phi2 = tf.get_variable(name="Phi2", shape=[n2, K],
                                               initializer=init_phi)
            init_omega = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            self.omega = omega = tf.get_variable(name="omega", shape=[K],
                                               initializer=init_omega)

            Phi1_2 = tf.pow(Phi1, 2)
            Phi2_2 = tf.pow(Phi2, 2)
            omega_2 = tf.pow(omega, 2)
            self.Theta1 = Theta1 = Phi1_2 / tf.reduce_sum(Phi1_2, 0)
            self.Theta2 = Theta2 = Phi2_2 / tf.reduce_sum(Phi2_2, 0)
            self.z = z = omega_2 / tf.reduce_sum(omega_2)

            Y = tf.matmul(Theta1, tf.matmul(tf.diag(z), Theta2, transpose_b=True))
            sum_weight = weights.sum()
            self.loss = loss = tf.nn.l2_loss((X / sum_weight) - Y)
            tf.scalar_summary("loss", loss)
            self.summary = tf.merge_all_summaries()
            self.opt = tf.train.AdamOptimizer(lr).minimize(loss)

            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()

    def optimize(self, logdir=None, max_steps=1000, stop_threshold=0.001):
        sess = self.sess
        sess.run(self.init_op)
        pre_loss = 10000
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        for step in range(max_steps):
            loss, sm, _ = sess.run([self.loss, self.summary, self.opt])
            if abs(loss - pre_loss) < stop_threshold:
                break
            if logdir:
                writer.add_summary(sm, step)
        return loss

    def get_theta1(self):
        return self.sess.run(self.Theta1)

    def get_theta2(self):
        return self.sess.run(self.Theta2)

    def get_z(self):
        return self.sess.run(self.z)

    def get_loss(self):
        return self.sess.run(self.loss)


if __name__ == '__main__':
    import pandas as pd
    os.system("rm -rf bifaclog")
    #edge_list = [(0,0), (0,1), (1,1),(1,0),(2,2),(2,3),(3,2),(3,3)]
    #weights = np.ones(len(edge_list))
    edge_list = pd.read_pickle("data/edge_list.pkl")
    weights = pd.read_pickle("data/weight.pkl")
    K = 2
    bifac = BiFac(edge_list, weights, K)
    bifac.optimize(logdir="bifaclog")
