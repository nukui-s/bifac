"""BiFAC on TensorFlow

Authoers: NUKUI Shun
License: GNU General Public License Version 2

"""

import numpy as np
import tensorflow as tf

INFINITY = 10e+1

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
            num_node = []
            for q in range(2):
                nn = max(map(lambda x: x[q], edge_list)) + 1
                num_node.append(nn)

            #id pairs of edge_list and weights
            self.edge_list = edge_list
            weights = np.array(weights, dtype=np.float32)
            sum_weight = np.sum(weights)

            self.X = X = tf.SparseTensor(values=weights,
                                                        indices=edge_list,
                                                        shape=(num_node))

            initializer_z = tf.random_normal_initializer(mean=sum_weight/K,
                                                                            stddev=1.0)
            self.z = z = tf.get_variable(name="z", shape=[K],
                                                    initializer=initializer_z)

            self.U1 = U1 = tf.get_variable(name="U1", shape=[num_node[0], K],
                                                        initializer=tf.random_uniform_initializer())

            self.U2 = U2 = tf.get_variable(name="U2", shape=[num_node[1], K],
                                                        initializer=tf.random_uniform_initializer())

            U2_T = tf.transpose(U2)
            S = [z[k] * tf.matmul(
                                            tf.reshape(U1[:, k], shape=(num_node[0], 1)),
                                            tf.reshape(U2_T[k, :], shape=(1, num_node[1]))
                                            ) for k in range(K)]
            #(K * N1 * N2)-Tensor
            self.S = S = tf.pack(S)
            self.Y = Y = tf.reduce_sum(S, reduction_indices=[0])
            self.y_values = y_values = tf.pack([Y[index] for index in edge_list])

            self.x_values = x_values = X.values
            sum_X = tf.reduce_sum(x_values)
            sum_Y = tf.reduce_sum(Y)

            self.sum_MI = sum_MI = tf.reduce_sum(x_values * (tf.log(x_values / y_values)))
            KL_divergence = sum_MI - sum_X + sum_Y

            self.sum_u1_i = sum_u1_i = tf.reduce_sum(U1, reduction_indices=[0])
            self.norm_u1 = normalize_U1 = tf.reduce_sum(tf.pow(
                                         sum_u1_i - 1.0, 2))
            self.sum_u1 = sum_u1 = tf.pow(tf.reduce_sum(U1 - 1.0), 2)
            normalize_U2 = tf.reduce_sum(tf.pow(
                                        tf.reduce_sum(U2, reduction_indices=[0]) - 1, 2
                                        ))
            constraint_z = tf.pow(tf.reduce_sum(z) - sum_weight, 2)
            nonnegative_U1 = tf.reduce_sum(tf.abs(U1) - U1)
            nonnegative_U2 = tf.reduce_sum(tf.abs(U2) - U2)

            constraint = normalize_U1 + normalize_U2 + nonnegative_U1 + nonnegative_U2

            self.loss = loss = KL_divergence + (nonnegative_U1 + nonnegative_U2) * INFINITY + \
                                            (normalize_U1 + normalize_U2 + constraint_z)*2

            optimizer = tf.train.AdagradOptimizer(learning_rate)
            self.optimize = optimizer.minimize(loss)

            u1_hist = tf.histogram_summary("U1", U1)
            loss_summ = tf.scalar_summary("loss", loss)
            self.merged = tf.merge_all_summaries()

    def run(self, sess):
        writer = tf.train.SummaryWriter("bifaclog", sess.graph_def)
        tf.initialize_all_variables().run()
        print(self.loss.eval())
        print(self.U1.eval())
        for iter_ in range(10000):
            z, U1, U2, merged, _ = sess.run([self.z, self.U1, self.U2,
                                                            self.merged, self.optimize])
            if iter_ % 100 == 0:
                writer.add_summary(merged, iter_)
        print(self.loss.eval())
        print(self.x_values.eval())
        print(self.y_values.eval())

        print(self.sum_u1_i.eval())
        print(self.Y.eval())
        return z, U1, U2

if __name__ == '__main__':
    edge_list = [(0,0), (0,1), (1,1),(1,0),(2,2),(2,3),(3,2),(3,3)]
    weights = np.ones(len(edge_list))
    K = 2
    bifac = BiFac(edge_list, weights, K)
    sess = tf.InteractiveSession()
    z, u1, u2 = bifac.run(sess)
    print(z)
    print(u1)
    print(u2)
    a = tf.ones([3, 4,2])
    z = tf.constant([1., 2])
    y = tf.ones([4])
    #sess = tf.InteractiveSession()
    #sp = tf.SparseTensor(values=y,indices=edge_list, shape=(4,2,3))
    #tf.initialize_all_variables().run()
