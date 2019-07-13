# distributed-tf
distributed tensorflow example repository
( This example uses deprecated version. )

#### execution command
> python train.py --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name='worker' --task_index=0 --max_steps=4000 --batch_size=128 --learning_rate=0.001 --log_dir=/tmp

You can add the IP of the node to use as argument --worker_hosts to register the worker host list.
Define a model of the code you want to split and execute in distmodel.py. In train.py, you can create a distributed model called distmodel.py.

#### train.py

    def main:
        with tf.device('job:/worker/task:0'):
            # computation in node1
            
        with tf.device('job:/worker/task:1'):
            #  computation in node2
        
        ...
