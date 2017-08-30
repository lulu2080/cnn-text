#encoding:utf-8
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.lib.io import file_io
import csv

# Parameters
# ==================================================
tf.flags.DEFINE_string("buckets", "oss://myaitest001/text-cnn/", "input data path")
tf.flags.DEFINE_string("checkpoint_dir", "oss://myaitest001/text-cnn/model/runs/1503020687/checkpoints/", "output model path")

# Data Parameters
#tf.flags.DEFINE_string("positive_data_file", "rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
#if FLAGS.eval_train:
#    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#    y_test = np.argmax(y_test, axis=1)
#else:
#    x_raw = ["a masterpiece four years in the making", "everything is off."]
#    y_test = [1, 0]

# Load data
#print("Loading data...")
#x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================

graph = tf.Graph()
with graph.as_default():

    def read_article(file_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
              'id': tf.FixedLenFeature([], tf.string),
              'title': tf.FixedLenFeature([], tf.string),
              'text_raw': tf.FixedLenFeature([], tf.string)
              })
        
    #    article = clean_str(features['text_raw'])

        id = features['id']
#        title = features['title']
        article = features['text_raw']
        
        return article, id
    
    def read_article_batch(file_queue, batch_size):
        article, id = read_article(file_queue)
        capacity = 3 * batch_size
        article_batch, id_batch = tf.train.batch([article, id], batch_size=batch_size, capacity=capacity, num_threads=1)
        return article_batch, id_batch    
    
    eval_file_path = os.path.join(FLAGS.buckets, "eval.tfrecords/*")
    eval_article_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(eval_file_path), num_epochs=1)
    eval_articles, eval_ids = read_article_batch(eval_article_filename_queue, FLAGS.batch_size)
    
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.buckets, "model/runs/1503020687/vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
#        # Initialize all variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
        
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
#        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_ids = []

#        for x_test_batch in batches:
#            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
#            all_predictions = np.concatenate([all_predictions, batch_predictions])

        try:
            while True:
                articles_batch, id_batch = sess.run([eval_articles,eval_ids])
                
                for j in range(len(articles_batch)):
                    articles_batch[j] = articles_batch[j].decode('utf-8')
                for j in range(len(id_batch)):
                    id_batch[j] = id_batch[j].decode('utf-8')

                x_batch = np.array(list(vocab_processor.fit_transform(articles_batch)))
                
                print(x_batch)
                print(id_batch)
                
                batch_predictions = sess.run(predictions, {input_x: x_batch, dropout_keep_prob: 1.0})
                print(batch_predictions)
                
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_ids = np.concatenate([all_ids, id_batch])
        except tf.errors.OutOfRangeError as e:
            print("data end!")
        finally:
            # stop queue runner
            coord.request_stop()
            coord.join(threads)

# Print accuracy if y_test is defined
#if y_test is not None:
#    correct_predictions = float(sum(all_predictions == y_test))
#    print("Total number of test examples: {}".format(len(y_test)))
#    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((all_ids, all_predictions))
out_path = os.path.join(FLAGS.buckets, "output/", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with file_io.FileIO(out_path, mode="w") as f:
    csv.writer(f).writerows(predictions_human_readable)