"""BiFAC on TensorFlow

Authoers: NUKUI Shun
License: GNU General Public License Version 2

"""

import numpy as np
import tensorflow as tf

INFINITY = 10e+100

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

    def __init__(self, edge_list, K):
            num_node = []
            for q in range(2):
                nn = max(map(lambda x: x[q], edge_list)) + 1
                num_node.append(nn)

            #id pairs of edge_list and weights
            edge_list_ = [(e[0], e[1]) for e in edge_list]
            weights = np.array([e[2] for e in edge_list]).astype(np.float32)

            self.X = X = tf.sparse_to_dense(
                                    sparse_indices=edge_list_,
                                    output_shape=num_node,
                                    sparse_values=weights,
                                    )

            self.z = z = tf.get_variable(name="z", shape=[K],
                                                    initializer=tf.random_uniform_initializer())

            self.U1 = U1 = tf.get_variable(name="U1", shape=[num_node[0], K],
                                                        initializer=tf.random_uniform_initializer())

            self.U2 = U2 = tf.get_variable(name="U2", shape=[num_node[1], K],
                                                        initializer=tf.random_uniform_initializer())

            #z_diag = tf.diag(z)

            U2_T = tf.transpose(U2)
            U_multiple_ = [tf.matmul(
                                            tf.reshape(U1[:, k], shape=(num_node[0], 1)),
                                            tf.reshape(U2_T[k, :], shape=(1, num_node[1]))
                                            ) for k in range(K)]
            #(K * N1 * N2)-Tensor
            U_multiple = tf.pack(U_multiple_)
            self.Y = Y = tf.reduce_sum( z * U_multiple,
                                        reduction_indices=[0])

            KL_divergence = tf.reduce_sum(X * (tf.log(X) - tf.log(Y)) - X + Y)

            self.loss = loss = KL_divergence


if __name__ == '__main__':
    edge_list = [(0,0,1), (0,1,1), (1,1,1),(2,1,2)]
    K = 2
    bifac = BiFac(edge_list, K)
    a = tf.ones([4,3,2])
    z = tf.ones([2])
    y = tf.ones([3])

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
