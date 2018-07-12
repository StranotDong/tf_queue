import __future__

import tensorflow as tf

# params
batch_size = 2
num_threads = 5
instance_queue_capacity = 10
min_after_dequeue = 2
num_iters = 100


# file queue
filename_queue = tf.train.string_input_producer(
	["data{0:d}.csv".format(i) for i in range(3)])

# read the file from file queue, 
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

# instance queue
instance_queue = tf.RandomShuffleQueue(
	capacity=instance_queue_capacity, 
	min_after_dequeue=min_after_dequeue,
	dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
	name="instance_q", shared_name="shared_instance_q")
enqueue_instance = instance_queue.enqueue([features, target])
dequeue_instance = instance_queue.dequeue_many(batch_size)


'''train ops'''
train_op = tf.reduce_sum(dequeue_instance[:][1])
''''''


# multithreads related
qr = tf.train.QueueRunner(instance_queue, [enqueue_instance] * num_threads)

with tf.Session() as sess:
	# Start populating the filename queue and instance queue	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	enqueue_threads = qr.create_threads(
		sess, coord=coord, start=True)
	# print(len(threads))
	
	for i in range(num_iters):
		if coord.should_stop():
		    break
		print(i, sess.run(train_op))

	coord.request_stop()
	coord.join(threads)
	coord.join(enqueue_threads)
