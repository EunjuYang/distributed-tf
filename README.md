# distributed-tf
distributed tensorflow example repository

commandd
>> python train.py --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name='worker' --task_index=0 --max_steps=4000 --batch_size=128 --learning_rate=0.001 --log_dir=/tmp
