import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv
import cv2
import sys
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system("clear")

#creat a list of label according to the file name in the training directory
def get_image(file_path):
	ax = []
	hx = []
	nx = []
	sx = []
	label_of_ax = []
	label_of_hx = []
	label_of_nx = []
	label_of_sx = []

	for file in os.listdir(file_path):
		name = file.split('_')
		if name[0] == 'ax':
			ax.append(file_path + file)
			label_of_ax.append(int(0))
		elif name[0] == 'hx':
			hx.append(file_path + file)
			label_of_hx.append(int(1))
		elif name[0] == 'nx':
			nx.append(file_path + file)
			label_of_nx.append(int(2))
		elif name[0] == 'sx':
			sx.append(file_path + file)
			label_of_sx.append(int(x=3))
	image_list = np.hstack((ax, hx, nx, sx))
	label_list = np.hstack((label_of_ax, label_of_hx, label_of_nx, label_of_sx))
# dtype of an array should be the same , so image_list is an string list
# label_list is an integer list, both of them turn into string list whenever
# you want two different types of list into an array
	tmp = np.array([image_list, label_list]) 
	tmp = tmp.transpose()
	np.random.shuffle(tmp)
	
	image_list = list(tmp[:,0])#[first_row:last_row, column 0]
	label_list = list(tmp[:,1])#[first_row:last_row, column 1]
	label_list = [int(float(i)) for i in label_list]#transform float to integer for every label in label list
	return image_list, label_list

#get an training batch in batchsize
def get_batch(image, label, image_W, image_H, batch_size, capacity):
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)
	#put training batch into a queue
	input_Queue = tf.train.slice_input_producer([image, label])

	label = input_Queue[1]
	image_undecoded = tf.read_file(input_Queue[0])
	#decode an image according to its format
	image = tf.image.decode_jpeg(image_undecoded, channels = 3)
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	#standlize the image to a standard tensorflow form
	image = tf.image.per_image_standardization(image)
	image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)
	label_batch = tf.reshape(label_batch, [batch_size])
	return image_batch, label_batch

#CNN model using convolution and pooling to get the weights
def inference(image, batch_size, n_classes):
	with tf.variable_scope("conv1") as scope:
		weights = tf.get_variable("weights",
		shape = [5,5,3,4],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [4],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(image, weights, strides = [1,1,1,1], padding = "SAME")
		activation = tf.nn.bias_add(conv,bias)
		conv1 = tf.nn.relu(activation, name = scope.name)
	with tf.variable_scope("pooling_lrn") as scope:
		pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "SAME", name = "pooling1")
		norm1 = tf.nn.lrn(pool1,
		depth_radius = 4,
		bias = 1,
		alpha = 0.001/9,
		beta = 0.75,
		name = "norm1")
	with tf.variable_scope("conv2") as scope:
		weights = tf.get_variable("weights",
		shape = [3,3,4,8],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [8],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(norm1, weights, strides = [1,1,1,1], padding = "SAME")
		activation = tf.nn.bias_add(conv,bias)
		conv2 = tf.nn.relu(activation, name = "conv2")
	with tf.variable_scope("pooling2_lrn") as scope:
		norm2 = tf.nn.lrn(conv2,
		depth_radius = 4,
		bias = 1,
		alpha = 0.001/9,
		beta = 0.75,
		name = "norm2")
		pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "SAME", name = "pooling2")
	with tf.variable_scope("conv3") as scope:
		weights = tf.get_variable("weights",
		shape = [3,3,8,16],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [16],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(norm2, weights, strides = [1,1,1,1], padding = "SAME")
		activation = tf.nn.bias_add(conv,bias)
		conv3 = tf.nn.relu(activation, name = "conv3")
	with tf.variable_scope("pooling3_lrn") as scope:
		norm3 = tf.nn.lrn(conv3,
		depth_radius = 4,
		bias = 1,
		alpha = 0.001/9,
		beta = 0.75,
		name = "norm3")
		pool3 = tf.nn.max_pool(norm3, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "SAME", name = "pooling3")
	with tf.variable_scope("local4") as scope:
		reshape = tf.reshape(pool3, shape = [batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = tf.get_variable("weights",
		shape = [dim,128],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [128],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(reshape, weights) + bias, name = scope.name)
	with tf.variable_scope("local5") as scope:
		weights = tf.get_variable("weights",
		shape = [128,128],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [128],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		local5 = tf.nn.relu(tf.matmul(local4, weights) + bias, name = "local5")
	with tf.variable_scope("softmax_linear") as scope:
		weights = tf.get_variable("softmax_linear",
		shape = [128, n_classes],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		bias = tf.get_variable("bias",
		shape = [n_classes],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local5, weights), bias, name = "softmax_linear")
		#avoid getting overfitting during training
		softmax_linear = tf.nn.dropout(softmax_linear, 0.5)
		return softmax_linear
def losses(logits, labels):
	with tf.variable_scope("loss") as scope:
		#calculate the loss
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "xentropy_per_example")
		loss = tf.reduce_mean(cross_entropy, name = "loss")
		tf.summary.scalar(scope.name + "/loss", loss)
		return loss
def trainning(loss, learning_rate):
	with tf.name_scope("optimizer"):
		#minimize the loss using AdamOptmizer
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		global_step = tf.Variable(0, name = "global_step", trainable = False)
		train_op = optimizer.minimize(loss, global_step = global_step)
		return train_op
def evaluation(logits, labels):
	with tf.variable_scope("accuracy") as scope:
		#claculate the training accuracy
		correct = tf.nn.in_top_k(logits, labels, 1)
		correct = tf.cast(correct, tf.float16)
		accuracy = tf.reduce_mean(correct)
		tf.summary.scalar(scope.name + "/accuracy", accuracy)
		return accuracy
def run_trainning():
	train_dir = "/Users/enzocheng/Desktop/python/data/ok/"
	logs_train_dir = "/Users/enzocheng/Desktop/python/data/log/"
	
	train, train_label = get_image(train_dir)
	train_batch, train_label_batch = get_batch(train, train_label, image_W, image_H, batch_size, capacity)
	train_logits = inference(train_batch, batch_size, n_classes)
	train_loss = losses(train_logits, train_label_batch)
	train_op = trainning(train_loss, learning_rate)
	train_acc = evaluation(train_logits, train_label_batch)
	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
	saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	try:
		for step in range(max_step):
			if coord.should_stop():
				break
			_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
			if step%5 == 0:
				print("step %s, train loss = %s, train accuracy = %s"%(step, tra_loss, tra_acc))
				summary_str = sess.run(summary_op)
				train_writer.add_summary(summary_str, step)
			#for every 100 step write the log to the log directory
			if step%100 ==0 or (step+1) == max_step:
				checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
				saver.save(sess, checkpoint_path, global_step = step)
	except tf.errors.OutOfRangeError:
		print("Done")
	finally:
		coord.request_stop()
	coord.join(threads)


n_classes = 4
image_W = 200
image_H = 200
batch_size = 100
capacity = 2000
max_step = 500
learning_rate = 0.0001
#run_trainning()



'''
///////////////////test image/////////////////////
'''
def get_test(path):
	image = []
	for file in os.listdir(path):
		name = file.split('.')
		if(name[0] == 'image'):
			image.append(path + name)
	return image


def get_test_imageName(test):
	n = len(test)
	index = np.random.randint(0, high=n)
	img_Name = test[index]
	return img_Name, index



def get_test_image(test, index):
	ind = index
	img_dir = test[ind]
	image = Image.open(img_dir)
	plt.imshow(image)
	plt.pause(1/30)
	image = image.resize([200, 200])
	image = np.array(image)
	return image

def evaluate_one_image():
	test_dir = '/Users/enzocheng/Desktop/python/data/ok/'
	test = get_test(test_dir)
	img_Name, index = get_test_imageName(test)
	test_image_array = get_test_image(test, index)

	with tf.Graph().as_default():
		BATCH_SIZE = 1
		N_CLASSES = 4
		image = tf.cast(test_image_array, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, 200, 200, 3])
		logit = inference(image, BATCH_SIZE, N_CLASSES)
		logit = tf.nn.softmax(logit)

		x = tf.placeholder(tf.float32, shape = [200, 200, 3])
		logs_of_train_dir = "/Users/enzocheng/Desktop/python/data/log/"
		saver = tf.train.Saver()

		with tf.Session() as sess:
			print("reading checking points")
			ckpt = tf.train.get_checkpoint_state(logs_of_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("loads Successfully global step is %s"%global_step)
			else:
				os.system("clear")
				print("error")

			prediction = sess.run(logit, feed_dict = {x: test_image_array})
			max_index = np.argmax(prediction, axis = 1)
			
			if max_index == 0:
				print("this is a cat with possibility %.6f" %prediction[:,0])
				img = cv2.imread(img_Name, 1)
				path = "/Users/enzocheng/Desktop/python/Cats/"
				cv2.imwrite(str(object= path) + "cat%s.jpg", img)
			else:
				print("this is a dog with possibility %.6f"%prediction[:,1])
				img = cv2.imread(img_Name, 1)
				path = "/Users/enzocheng/Desktop/python/Dogs/"
				cv2.imwrite(str(object= path) + "dog%s.jpg" , img)	

#for i in range(0, 30, +1):
	#evaluate_one_image(i)



#using openCV to test an image or video
def videoShow(mirror = False):
	cap = cv2.VideoCapture('456.mp4')
	while(1):
		ret, frame = cap.read()
		if mirror:
			frame = cv2.flip(frame,1)
		cascPath = sys.argv[1]
		faceCascade = cv2.CascadeClassifier(cascPath)
		
		print(frame.shape)
		#frame = cv2.warpAffine(frame,M,(width,height))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('frame', 1000, 1000)
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        	crop = frame[y : y + h, x : x + w]
        	crop_shaped = cv2.resize(crop, (200, 200))
        	if x+w > 800:
        		cv2.putText(frame, "sad", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        	else:
				cv2.putText(frame, "sad", (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        	cv2.imshow('frame', frame)
        	'''
        	path = "/Users/enzocheng/Desktop/python/test_image/"
        	cv2.imwrite(str(object= path) + "image.jpg", crop_shaped)
        	'''
        	'''
        	if x+w > 800:
				cv2.putText(frame, label, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
			else:
				cv2.putText(frame, label, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
     		'''
     		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
	cap.release()
	cv2.destroyAllWindows()
videoShow(mirror = False)




#to show the image from train directory. this is using for checking the training image's correctness


'''
train_dir = "/Users/enzocheng/Desktop/python/data/train/"
image_list, label_list = get_image(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, image_W = 208, image_H = 208, batch_size = 16, capacity = 2000 )
sess = tf.Session()
#run_trainning
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord )
i = 0
try:
	while not coord.should_stop() and i<1:
		img, lab = sess.run([image_batch, label_batch])
		for j in range(batch_size):
			print("label : %s"%lab[j])
			plt.imshow(img[j])
			plt.pause(1)
		i = i + 1	
except tf.errors.OutOfRangeError:
		print("Done")
finally:
	coord.request_stop()
coord.join(threads)
'''












