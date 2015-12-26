import tensorflow as tf
import numpy as np

INF = 10e+10

if __name__ == '__main__':
    x = tf.get_variable(name="x", shape=[1000,1000], initializer=tf.random_uniform_initializer())
    x_sum1 = tf.reduce_sum(x, reduction_indices=[0])
    loss = tf.reduce_sum(tf.pow(x_sum1 - 1.0, 2)) * INF
    loss_summ = tf.scalar_summary("loss", loss)
    #opt = tf.train.AdagradOptimizer(0.1).minimize(loss)
    opt = tf.train.AdagradOptimizer(0.1).minimize(loss)

    sess = tf.InteractiveSession()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("testlog", sess.graph_def)

    tf.initialize_all_variables().run()
    for i in range(100):
        #print(x.eval())
        #print(x_sum1.eval())
        _, summary_string = sess.run([opt, merged])
        writer.add_summary(summary_string, i)
