#coding=utf-8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name

import sys
sys.path.insert(0, "../../python")
import tensorflow as tf
import numpy as np
import cv2, random
from io import BytesIO
from generateplate import *
from model impot *


class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


def rand_range(lo,hi):
    return lo+r(hi-lo);


def gen_rand():
    name = "";
    label= [];
    label.append(rand_range(0,31));
    label.append(rand_range(41,65));
    for i in range(5):
        label.append(rand_range(31,65))

    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in range(5):
        name+=chars[label[i+2]]
    print("gen_rand name:", name," label:",label)  
    return name,label


def gen_sample(genplate, width, height):
    name,label = gen_rand()
    img = generateplate.generate(name)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return label, img
   

class OCRIter():
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.generateplate = GeneratePlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        print "start"
    def __iter__(self):

        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.generateplate, self.width, self.height)
                data.append(img)
                label.append(num)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            print("__iter__ data_batch:",data_batch)
            yield data_batch

    def reset(self):
        pass

def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 1:],[-1, len(common.CHARS)]),
                                          tf.reshape(y_[:, 1:],[-1, len(common.CHARS)]))
    digits_loss = tf.reshape(digits_loss, [-1, 7])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    digits_loss *= (y_[:, 0] != 0)
    digits_loss = tf.reduce_sum(digits_loss)
    # Calculate the loss from presence indicator being wrong.
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(y[:, :1], y_[:, :1])
    presence_loss = 7 * tf.reduce_sum(presence_loss)
    return digits_loss, presence_loss, digits_loss + presence_loss

output_size = 1+7*len(common.CHARS)]

def weight_variable(shape):
	weight=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	return weight
	
def bias_viribale(shape):
	bias=tf.Variable(tf.constant(0.1,shape=shape))
	return bias
	
def conv2d(x,w,stride,padding='SAME'):
	conv2d=tf.nn.conv2d(x,w,strides=[1,1,1,1],padding=padding)
	return conv2d

def maxpool2d(x,ksize=(2,2),stride=(2,2)):
	maxpool2d=tf.nn.max_pool(x,ksize=[1,ksize[0],ksize[1],1],strides=[1,stride[0],stride[1],1],padding="SAME")

def model_cnn():
	x=tf.placeholder(tf.float32,[None,None,None,3])
	
	#layer 1
	w_conv1=weight_variable([5,5,3,48])
	b_conv1=bias_variable([48])
	conv1=tf.nn.relu(conv2d(x,w_conv1)+b_conv1)
	pool1=maxpool2d(conv1)
	
	#layer 2
	w_conv2=weight_variable([5,5,48,64])
	b_conv2=bias_variable([64])
	conv2=tf.nn.relu(conv2d(pool1,w_conv2)+b_conv2)
	pool2=maxpool2d(conv2)
	
	#layer 3
	w_conv3=weight_variable([5,5,64,128])
	b_conv3=bias_variable([128])
	conv3=tf.nn.relu(conv2d(pool2,w_conv3)+b_conv3)
	pool3=maxpool2d(conv3)
	
	return x,pool3,[w_conv1,b_conv1,w_conv2,b_conv2,w_conv3,b_conv3]
	
def train_model():
	x,y_cnn,paras=model_cnn()
	
	#fc layer 1
	w_fc1=weight_variable([32*8*128,2048])
	b_fc1=bias_variable([2048])
	flatten=tf.reshape(y_cnn,[-1,32*8*128])
	fc1=tf.nn.relu(tf.matmul(flatten,w_fc1)+b_fc1)
	
	#fc layer 2 : output 
	w_fc2=weight_variable([2048,output_size])
	b_fc2=bias_variable(1+7*len(common.CHARS))
	y=tf.matmul(fc1,w_fc2)+b_fc2
	
	return (x,y,paras+[w_fc1,b_fc1,w_fc2,b_fc2])
	
	
def detect_model():
	x,y_cnn,paras=model_cnn()
	
	#conv layer 1
	w_fc1 = weight_variable([32*8*128,2048])
	b_fc1=bias_variable([2048])
	w_conv1=tf.reshape(w_fc1,[8,32,128,2048])
	conv1=tf.nn.relu(conv2d(y_cnn,w_conv1,stride=(1,1),padding="VALID")+b_fc1)
	
	#conv layer 2
	w_fc2 = weight_variable([2048,output_size])
	b_fc2=bias_variable([output_size])
	w_conv2=tf.reshape(,[1,1,2048,output_size])
	conv2=conv2d(conv1,w_conv2)+b_conv2)
	
	return (x,conv2,paras+[w_fc1,b_fc1,w_fc2,b_fc2])
	

def train(learn_rate, report_steps, batch_size, initial_weights=None):
    x, y, params = train_model()
    y_ = tf.placeholder(tf.float32, [None, 7 * len(common.CHARS) + 1])
    digits_loss, presence_loss, loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(common.CHARS)]), 2)
    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()
    
    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      loss],
                     feed_dict={x: test_xs, y_: test_ys})
        num_correct = np.sum(
                        np.logical_or(
                            np.all(r[0] == r[1], axis=1),
                            np.logical_and(r[2] < 0.5,
                                              r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(vec_to_plate(c), pc,
                                           vec_to_plate(b), float(pb))
        num_p_correct = numpy.sum(r[2] == r[3])

        print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
               "(digits: {}, presence: {}) |{}|").format(
            batch_idx,
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            r[4],
            r[5],
            "".join("X "[np.array_equal(b, c) or (not pb and not pc)]
                                           for b, c, pb, pc in zip(*r_short)))

    def do_batch():
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) / (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            np.savez("weights.npz", *last_weights)
            return last_weights


def train():
    batch_size = 8
    data_train = OCRIter(500000, batch_size, 7, 30, 120)
    data_test = OCRIter(1000, batch_size,7, 30, 120)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    #if len(sys.argv) > 1:
    #    f = np.load(sys.argv[1])
    #    initial_weights = [f[n] for n in sorted(f.files,key=lambda s: int(s[4:]))]
    #else:
    #    initial_weights = None

    train(learn_rate=0.001,
          report_steps=20,
          batch_size=50,
          initial_weights=initial_weights)

if __name__ == '__main__':
    train();
