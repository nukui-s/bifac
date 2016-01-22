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

    def __init__(self, K, edge_list, weights, constraint=None,
                learning_rate=0.01, lambda_r=10e-9, lambda_c=0.1):
        self.edge_list = edge_list
        indices = np.array(edge_list)
        self.n1 = indices[:,0].max() + 1
        self.n2 = indices[:,1].max() + 1
        self.K = K
        self.constraint = constraint
        self.learning_rate = learning_rate
        self.weights = np.array(weights, dtype=np.float32)
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.setup_graph()
        self.initialize_variables()

    def initialize_variables(self):
        self.sess.run(self.init_op)

    def setup_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            n1, n2 = self.n1, self.n2
            edge_list = self.edge_list
            edge_list = edge_list
            weights = self.weights
            constraint = self.constraint
            K = self.K
            lr = self.learning_rate
            lambda_r = self.lambda_r
            lambda_c = self.lambda_c
            eps = 0.0

            self.X = X = tf.sparse_to_dense(output_shape=[n1, n2],
                                            sparse_values=weights,
                                            sparse_indices=edge_list)

            init_phi = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            self.Phi1 = Phi1 = tf.get_variable(name="Phi1", shape=[n1, K],
                                               initializer=init_phi)
            self.Phi2 = Phi2 = tf.get_variable(name="Phi2", shape=[n2, K],
                                               initializer=init_phi)
            init_omega = tf.random_normal_initializer(mean=1.0/K, stddev=10e-8)
            self.omega = omega = tf.get_variable(name="omega", shape=[K],
                                                initializer=init_omega)

            #Phi1_2 = tf.pow(Phi1, 2)
            #Phi2_2 = tf.pow(Phi2, 2)
            #omega_2 = tf.pow(omega, 2)
            Phi1_2 = tf.abs(Phi1)
            Phi2_2 = tf.abs(Phi2)
            omega_2 = tf.abs(omega)
            Phi1_sqsum = tf.reduce_sum(Phi1_2, 0)
            Phi2_sqsum = tf.reduce_sum(Phi2_2, 0)
            omega_sqsum = tf.reduce_sum(omega_2)
            self.Theta1 = Theta1 = Phi1_2 / (Phi1_sqsum + eps)
            self.Theta2 = Theta2 = Phi2_2 / (Phi2_sqsum + eps)
            self.z = z = omega_2 / (omega_sqsum + eps)

            with tf.name_scope("constraint"):
                if constraint:
                    constraint = constraint + [(j,i) for i, j in constraint]
                    n_cnst = len(constraint)
                    self.C = C = tf.sparse_to_dense(output_shape=[n1, n1],
                                                sparse_values=tf.ones([n_cnst]),
                                                sparse_indices=constraint)
                    sim = tf.matmul(Theta1, Theta1, transpose_b=True)
                    self.cnst_reward = cnst_reward = tf.reduce_sum(C * sim)
                else:
                    self.cnst_err = cnst_err = 0.0

            Y = tf.matmul(Theta1, tf.matmul(tf.diag(z), Theta2,
                                            transpose_b=True))
            sum_weight = weights.sum()
            self.reglr = reglr = tf.nn.l2_loss(Phi1) + \
                                   tf.nn.l2_loss(Phi2) + \
                                   tf.nn.l2_loss(omega)
            self.recon_err = recon_err = tf.nn.l2_loss((X / sum_weight) - Y)
            self.cost = cost = recon_err + lambda_r * reglr + \
                                        lambda_c * cnst_reward
            self.opt = tf.train.AdamOptimizer(lr).minimize(cost)

            tf.histogram_summary("z", z)
            tf.histogram_summary("omega", omega)
            tf.histogram_summary("Phi1", Phi1)
            tf.histogram_summary("Phi2", Phi2)
            tf.histogram_summary("Theta1", Theta1)
            tf.histogram_summary("Theta2", Theta2)
            tf.scalar_summary("cost", cost)
            tf.scalar_summary("reglr", reglr)
            tf.scalar_summary("recon_err", recon_err)
            tf.scalar_summary("cnst_reward", cnst_reward)
            self.summary = tf.merge_all_summaries()

            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()

    def optimize(self, logdir=None, max_steps=1000, stop_threshold=0.001):
        sess = self.sess
        sess.run(self.init_op)
        pre_cost = -1000
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        for step in range(max_steps):
            cost, sm, _ = sess.run([self.cost, self.summary, self.opt])
            if abs(cost - pre_cost) < stop_threshold:
                break
            if logdir:
                writer.add_summary(sm, step)
        z = self.get_z()
        theta1 = self.get_theta1()
        theta2 = self.get_theta2()
        return cost, z, theta1, theta2

    def get_theta1(self):
        return self.sess.run(self.Theta1)

    def get_theta2(self):
        return self.sess.run(self.Theta2)

    def get_z(self):
        return self.sess.run(self.z)

    def get_cost(self):
        return self.sess.run(self.cost)

    def get_recon_err(self):
        return self.sess.run(self.recon_err)

    def get_reglr(self):
        return self.sess.run(self.reglr)

    def get_hard_community(self, mode=None):
        Theta1 = self.get_theta1()
        Theta2 = self.get_theta2()
        z = self.get_z()
        com1_prob = Theta1 * z
        com2_prob = Theta2 * z
        com1 = com1_prob.argmax(axis=1)
        com2 = com2_prob.argmax(axis=1)
        coms = [com1, com2]
        if mode:
            return coms[mode]
        return com1, com2

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
