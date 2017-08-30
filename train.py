#encoding:utf-8
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import argparse
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================
tf.flags.DEFINE_string("buckets", "oss://myaitest001/text-cnn/", "input data path")
tf.flags.DEFINE_string("checkpointDir", "oss://myaitest001/text-cnn/", "output model path")
#tf.flags.DEFINE_string("buckets", "D:/ai/cnn-text-classification-tf/", "input data path")
#tf.flags.DEFINE_string("checkpointDir", "D:/ai/cnn-text-classification-tf/", "output model path")
# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("positive_data_file", "rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,5,8", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 15, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("sequence_length", 5012, "Lenth of max article size (default: 5012)")
tf.flags.DEFINE_integer("num_classes", 4, "Number of result class (default: 4)")
tf.flags.DEFINE_integer("vocab_size", 1000000, "Size of vocab (default: 1000000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#
#print("x_text:")
#print(x_text)
#print("y:")
#print(y)

##训练数据集
#train_file_path = os.path.join(FLAGS.buckets, "data/train.tfrecords")
##测试数据集
#test_file_path = os.path.join(FLAGS.buckets, "data/test.tfrecords")

#train_article_filename_queue = tf.train.string_input_producer(
#        tf.train.match_filenames_once(train_file_path))
#train_articles, train_labels = data_helpers.read_article_batch(train_article_filename_queue, FLAGS.batch_size)

#test_article_filename_queue = tf.train.string_input_producer(
#        tf.train.match_filenames_once(test_file_path))
#test_articles, test_labels = data_helpers.read_article_batch(test_article_filename_queue, FLAGS.batch_size)

# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.sequence_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))
#
#print("x:")
#print(x)

# Randomly shuffle data
#np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(y)))
#x_shuffled = x[shuffle_indices]
#y_shuffled = y[shuffle_indices]
#
#print("x_shuffled:")
#print(x_shuffled)
#print("y_shuffled:")
#print(y_shuffled)

# Split train/test set
# TODO: This is very crude, should use cross-validation
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
#
#print("x_train:")
#print(x_train)
#print("y_train:")
#print(y_train)

# Training
# ==================================================
def main(_):
    with tf.Graph().as_default():
    
        def read_article(file_queue):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(file_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                  'text_raw': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64),
                  })
            
        #    article = clean_str(features['text_raw'])

            article = features['text_raw']
            label = tf.cast(features['label'], tf.int32)        
            
            return article, label
        
        def read_article_batch(file_queue, batch_size):
            article, label = read_article(file_queue)
            capacity = 3 * batch_size
            min_after_dequeue = int(capacity / 2)
            article_batch, label_batch = tf.train.shuffle_batch([article, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=1)
            one_hot_labels = tf.to_float(tf.one_hot(label_batch, 4, 1, 0))
            return article_batch, one_hot_labels    
    
        #训练数据集
        train_file_path = os.path.join(FLAGS.buckets, "train.tfrecords/*")
        #测试数据集
        test_file_path = os.path.join(FLAGS.buckets, "test.tfrecords/*")
    
        train_article_filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(train_file_path), num_epochs=FLAGS.num_epochs)
        test_article_filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(test_file_path), num_epochs=FLAGS.num_epochs)
    
        train_articles, train_labels = read_article_batch(train_article_filename_queue, FLAGS.batch_size)
        test_articles, test_labels = read_article_batch(test_article_filename_queue, FLAGS.batch_size)
    
    
#        sess = tf.InteractiveSession()
#        tf.global_variables_initializer().run()
#        tf.local_variables_initializer().run()
#    #    # start queue runner
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#        
#        xb, yb = sess.run([train_articles,train_labels])
#        print('xb len: {:d}'.format(len(xb)))
#        print(bytes.decode(xb[0], 'utf8'))
#        print(bytes.decode(xb[1], 'utf8'))
#        print(yb)
    
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.sequence_length,
                num_classes=FLAGS.num_classes,
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
    
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
    
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.join(FLAGS.checkpointDir, "runs", timestamp)
            print("Writing to {}\n".format(out_dir))
    
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
    
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            if not os.path.exists(train_summary_dir):
                os.makedirs(train_summary_dir)            
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            if not os.path.exists(dev_summary_dir):
                os.makedirs(dev_summary_dir)
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
    
            # Initialize all variables
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
    
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                x_batch = np.array(list(vocab_processor.fit_transform(x_batch)))
                print(x_batch)
                print(y_batch)
                            
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
    
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                x_batch = np.array(list(vocab_processor.fit_transform(x_batch)))
                print(x_batch)
                print(y_batch)
                
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
    
            # start queue runner
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            # Generate batches
#            batches = data_helpers.batch_iter(
#                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for i in range(15000):
    #            x_batch, y_batch = zip(*batch)
    
                x_batch, y_batch = sess.run([train_articles,train_labels])
                
                
                for j in range(len(x_batch)):
                    x_batch[j] = x_batch[j].decode('utf-8')
                
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    x_test_batch, y_test_batch = sess.run([test_articles,test_labels])
                    for j in range(len(x_test_batch)):
                        x_test_batch[j] = x_test_batch[j].decode('utf-8')
                    dev_step(x_test_batch, y_test_batch, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    
                sys.stdout.flush()
            
            # stop queue runner
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    #获得buckets路径
#    parser.add_argument('--buckets', type=str, default='',
#                        help='input data path')
#    #获得checkpoint路径
#    parser.add_argument('--checkpointDir', type=str, default='',
#                        help='output model path')
#    
#    parser.add_argument('--batch_size', type=int, default=2,
#                        help='output model path')
#        
#    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)

