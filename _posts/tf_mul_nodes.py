from tensorflow.bml.server_run import Server as server
import tensorflow as tf
import numpy as np
 
# creat worker server which will creat cluster and server automatically.
# methods for class Server:
#   get_cluster(): return cluster initialized by tf.train.ClusterSpec({"ps": ps_hosts_list, "worker": worker_hosts_list})
#   get_server(): return current server initialized by tf.train.Server()
#   get_task_index(): return current task index
#   get_current_rank(): return current node rank
#   get_total_rank(): return total number of nodes in cluster
 
serv = server("worker")
 
# specify device for variables. if you want assign devices automatically, use statement as follow,
# with tf.device(tf.train.replica_device_setter(cluster=serv.get_cluster())):
with tf.device("/job:ps/task:0/cpu:0"):
  W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
  b = tf.Variable(tf.zeros([1]))
 
with tf.device(tf.train.replica_device_setter(
               worker_device="/job:worker/task:%d" % serv.get_task_index(),
               cluster=serv.get_cluster())):
  x_data = np.random.rand(100).astype(np.float32)
  y_data = x_data * 0.1 + 0.3
  y = W * x_data + b
  loss = tf.reduce_mean(tf.square(y - y_data))
  global_step = tf.Variable(0)
  optimizer = tf.train.GradientDescentOptimizer(0.5)
  train = optimizer.minimize(loss, global_step=global_step)
 
is_chief = serv.get_task_index() == 0
if is_chief:
  print "current is chief worker."
 
sv = tf.train.Supervisor(is_chief=is_chief,
                         logdir="./log",
                         saver=tf.train.Saver(max_to_keep=100),
                         checkpoint_basename="test_model",
                         init_op=tf.global_variables_initializer())
sess_config = tf.ConfigProto(allow_soft_placement=True)
 
with sv.managed_session(serv.get_server().target, config=sess_config) as sess:
  W_init, b_init, step = sess.run([W, b, global_step])
  print W_init, b_init
  while not sv.should_stop() and step < 10000:
    _, step = sess.run([train, global_step])
    if is_chief and step % 10 == 0:
      sv.saver.save(sess, save_path=sv.save_path, global_step=step.astype(int))
    print "step: " % sess.run(global_step), sess.run(W), sess.run(b)
sv.stop()