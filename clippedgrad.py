import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_ops

class ClippedAdagradOptimizer(tf.train.AdagradOptimizer):
    """AdagradOptimizer clipped grad to non-negative value"""

    def _apply_dense(self, grad, var):
        acc = self.get_slot(var, "accumulator")
        grad_clipped = self._clip_grad(grad, var)
        return training_ops.apply_adagrad(
            var, acc, self._learning_rate_tensor, grad_clipped,
            use_locking=self._use_locking)

    def _apply_sparse(self, grad, var):
        acc = self.get_slot(var, "accumulator")
        grad_clipped = self._clip_grad(grad, var)
        return training_ops.apply_adagrad(
            var, acc, self._learning_rate_tensor, grad_clipped,
            use_locking=self._use_locking)

    def _clip_grad(self, grad, var):
        acc = self.get_slot(var, "accumulator")
        lr = self._learning_rate_tensor
        eps = 10e-4
        eta = 10e+8
        grad = tf.clip_by_value(grad, -0.01, 0.01)
        sqdiv = tf.pow(lr, 2) - tf.pow(var, 2)
        #if sqdiv < 0  the next var absolutely become positive
        penalty = tf.to_float(tf.less(sqdiv, 0)) * eta
        g_max = tf.div(var * tf.sqrt(sqdiv*acc), sqdiv + eps) + penalty
        grad_clipped = tf.minimum(grad, g_max)
        return grad_clipped

class ClippedGDOptimizer(tf.train.GradientDescentOptimizer):
    """GradientDescentOptimizer clipped grad to non-negative"""

    def _apply_dense(self, grad, var):
        lr = self._learning_rate_tensor
        grad_clipped = tf.minimum(var/lr, grad)
        return training_ops.apply_gradient_descent(
            var,
            self._learning_rate_tensor,
            grad_clipped,
            use_locking=self._use_locking).op

class ClippedAdamOptimizer(tf.train.AdamOptimizer):
    pass
