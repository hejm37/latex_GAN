import os
import numpy as np
from model import *
from util import *
from load import mnist_with_valid_set

iterations = 5
batch_size = 128
image_shape = [28,28,1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64

visualize_dim=196

dcgan_model = DCGAN(
		batch_size=batch_size,
		image_shape=image_shape,
		dim_z=dim_z,
		dim_W1=dim_W1,
		dim_W2=dim_W2,
		dim_W3=dim_W3,
		)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.global_variables_initializer().run()

# SMITH: Restore model session, maybe I should use relative path
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), "models/")))
print('Restore succefully from', os.path.join(os.getcwd(), "models/"))

flags = np.zeros(10)

for iteration in range(iterations):
	Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
	Y_np_sample = OneHot( np.random.randint(10, size=[visualize_dim]))
	generated_samples = sess.run(
			image_tf_sample,
			feed_dict={
				Z_tf_sample:Z_np_sample,
				Y_tf_sample:Y_np_sample
				})
	generated_samples = (generated_samples + 1.)/2.
	save_data_set(generated_samples, Y_np_sample, flags)
	# save_data_set_3C(generated_samples, Y_np_sample, flags)
