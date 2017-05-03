from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import time
import datetime
import sys
from distutil import *
from distmodel import *
import tensorflow as tf

FLAGS = None


def main(_):
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"worker": worker_hosts})
  # Create and start a server for the local task.
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  with tf.device('/job:worker/task:0'): # conv layer
    global_step = tf.Variable(0, trainable=False)
    image_size=32
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                            image_size, 3],
                                           dtype=tf.float32,
                                           stddev=1e-1), trainable=False)
    pool5_value = node1(images, FLAGS)

  with tf.device('/job:worker/task:1'): # fc layer
    labels = tf.Variable(tf.constant(0, dtype=tf.int32,shape=[FLAGS.batch_size]), trainable=False)
    logits = node2(pool5_value, FLAGS)
    loss = calculate_loss(logits, labels)
    loss_summary = tf.summary.scalar('loss',loss)
    default = tf.AggregationMethod.DEFAULT
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step, aggregation_method=default, colocate_gradients_with_ops=True)


  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.log_device_placement = False


  with tf.Session("grpc://%s"%worker_hosts[0], config=config) as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    total_start_time = time.time()
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time
      total_duration = time.time() - total_start_time

      if step % 10 == 0:
        sec_per_print = float(duration)
        format_str = ('%s: step = %d, loss = %.2f, total time = %.2f')
        print (format_str % (datetime.now(), step, loss_value, total_duration))
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)


if __name__ == "__main__":

    parser = arg_parser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
