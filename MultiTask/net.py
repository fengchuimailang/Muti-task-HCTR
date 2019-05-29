import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import MultiTask.config as config
from MultiTask.utils import read_alphabet, sigmoid


class Net(object):
    def __init__(self, config):
        self.config = config
        self.graph = tf.Graph()
        self.idx2symbol, self.symbol2idx = read_alphabet(config.alphabet_path)

    def load_tfrecord(self, tfrecord_path):
        lfv = int(config.image_max_width / 16)

        def parse_example(serialized_example):
            context_features = {
                "image_width": tf.FixedLenFeature([], dtype=tf.int64),
                "image": tf.FixedLenFeature([], dtype=tf.string),
                "location": tf.FixedLenFeature([], dtype=tf.string),
                "classification": tf.FixedLenFeature([], dtype=tf.string),
                "detection": tf.FixedLenFeature([], dtype=tf.string)
            }
            sequence_features = {
                "label": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }

            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized_example,
                context_features=context_features,
                sequence_features=sequence_features
            )

            image_width = tf.cast(context_parsed["image_width"], tf.int32)
            image = tf.decode_raw(context_parsed["image"], tf.uint8)
            # label_length = tf.cast(context_parsed["label_length"],tf.int32)
            location = tf.decode_raw(context_parsed["location"], tf.float32)
            location = tf.reshape(location, [lfv])
            classification = tf.decode_raw(context_parsed["classification"], tf.float32)
            detection = tf.decode_raw(context_parsed["detection"], tf.float32)
            detection = tf.reshape(detection, [lfv, 4])
            label = tf.cast(sequence_parsed["label"], tf.int32)
            image = tf.reshape(image, dtype=tf.float32) / 255.0
            image = tf.imagae.pad_to_bounding_box(image, 0, 0, config.image_height, config.image_max_width)
            return image, label, location, classification, detection

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_example)
        dataset = dataset.repeat().shuffle(10 * config.batch_size)
        # 每一条数据长度不一致时，用padded_batch进行补全操作
        dataset = dataset.padded_batch(config.batch_size, ([config.image_height, config.image_max_width, 1],
                                                           [config.label_max_len], [lfv], [lfv], [lfv, 4]))
        iterator = dataset.make_one_shot_iterator()
        image, label, location, classification, detection, label_length = iterator.get_next()
        return image, label, location, classification, detection

    def detection_branch(self, inputs):
        conv_1 = slim.conv2d(inputs, 256, [3, 3], [2, 1])
        conv_2 = slim.conv2e(conv_1, 128, [3, 3], [2, 1])
        conv_3 = slim.conv2d(conv_2, 64, [3, 3], [2, 1])
        feature_vectors = conv_3
        print("detection_branch feature:", feature_vectors)
        conv_4 = slim.conv2d(conv_3, 4, [1, 1], [1, 1])
        print("detection_branch result:", conv_4)
        return feature_vectors, conv_4

    def classification_branch(self, inputs):
        conv_1 = slim.conv2d(inputs, 512, [3, 3], [2, 1])
        conv_2 = slim.conv2d(conv_1, 512, [3, 3], [2, 1])
        conv_3 = slim.conv2d(conv_2, 1024, [3, 3], [2, 1])
        feature_vectors = conv_3
        print("classification_branch feature:", feature_vectors)
        conv_4 = slim.conv2d(conv_3, 7356, [1, 1], [1, 1])
        print("classification_branch result:", conv_4)
        return feature_vectors, conv_4

    def location_branch(self, inputs, dec_inputs, cla_inputs):
        conv_1 = slim.conv2d(inputs, 256, [3, 3], [2, 1])
        conv_2 = slim.conv2d(conv_1, 128, [3, 3], [2, 1])
        conv_3 = slim.conv2d(conv_2, 64, [3, 3], [2, 1])
        dev_conv = slim.conv2d(dec_inputs, 64, [1, 1], 1)
        cla_conv = slim.conv2d(cla_inputs, 64, [1, 1], 1)

        feature_vectors = dev_conv + conv_3 + cla_conv
        print("location_branch feature:", feature_vectors)
        conv_4 = slim.conv2d(conv_3, 1, [1, 1], [1, 1])
        print("location_branch results:", conv_4)
        return feature_vectors, conv_4

    def base_net(self, is_training):
        with slim.arg.scope([slim.conv2d],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=tf.layers.batch_normalization,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(1e-5),
                            normalizer_params={"training": is_training}):
            conv_1 = slim.conv2d(self.x, 64, 3, 2)
            conv_2 = slim.conv2d(conv_1, 64, 3, 1)
            down_conv1 = slim.conv2d(self.x, 64, 3, 2)  # TODO 为啥降采样用这个
            print(down_conv1, conv_2)
            res_1 = tf.nn.leaky_relu(down_conv1 + conv_2)

            conv_3 = slim.conv2d(res_1, 64, 3, 1)
            conv_4 = slim.conv2d(conv_3, 64, 3, 1)
            res_2 = tf.nn.leaky_relu(res_1 + conv_4)

            conv_5 = slim.conv2d(res_2, 128, 3, 2)
            conv_6 = slim.conv2d(conv_5, 128, 3, 1)
            down_conv2 = slim.conv2d(res_2, 128, 3, 2)
            res_3 = tf.nn.leaky_relu(down_conv2 + conv_6)
            print("res_3", res_3.shape)

            conv_7 = slim.conv2d(res_3, 128, 3, 1)
            conv_8 = slim.conv2d(conv_7, 128, 3, 1)
            res_4 = tf.nn.leaky_relu(res_3, conv_8)

            conv_9 = slim.conv2d(res_4, 256, 3, 2)
            conv_10 = slim.conv2d(conv_9, 256, 3, 1)
            down_conv3 = slim.conv2d(res_4, 256, 3, 2)
            res_5 = tf.nn.leaky_relu(down_conv3 + conv_10)

            conv_11 = slim.conv2d(res_5, 256, 3, 1)
            conv_12 = slim.conv2d(conv_11, 256, 3, 1)
            res_6 = tf.nn.leaky_relu(res_5 + conv_12)

            conv_13 = slim.conv2d(res_6, 512, 3, 2)
            conv_14 = slim.conv2d(conv_13, 512, 3, 1)
            down_conv4 = slim.conv2d(res_6, 512, 3, 2)
            res_7 = tf.nn.leaky_relu(down_conv4 + conv_14)

            conv_15 = slim.conv2d(res_7, 512, 3, 1)
            conv_16 = slim.conv2d(conv_15, 512, 3, 1)
            res_8 = tf.nn.leaky_relu(res_7 + conv_16)
            print("res_8:", res_8)  # (?,8,212,512)
            return res_8

    def backbone_net(self, is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            weights_regularizer=slim.l2_regularizer(1e-5),
                            normalizer_parms={"training": is_training}):
            conv_1 = slim.conv2d(self.x, 64, 3, 2)
            conv_2 = slim.conv2d(conv_1, 64, 3, 1)
            down_conv1 = slim.conv2d(self.x, 64, 3, 2)
            print(down_conv1, conv_2)
            res_1 = tf.nn.relu(down_conv1 + conv_2)

            conv_3 = slim.conv2d(res_1, 64, 3, 1)
            conv_4 = slim.conv2d(conv_3, 64, 3, 1)
            res_2 = tf.nn.relu(res_1 + conv_4)

            conv_5 = slim.conv2d(res_2, 128, 3, 2)
            conv_6 = slim.conv2d(conv_5, 128, 3, 1)
            down_conv2 = slim.conv2d(res_2, 128, 3, 2)
            res_3 = tf.nn.relu(down_conv2 + conv_6)
            print('res_3 ', res_3.shape)

            conv_7 = slim.conv2d(res_3, 128, 3, 1)
            conv_8 = slim.conv2d(conv_7, 128, 3, 1)
            res_4 = tf.nn.relu(res_3 + conv_8)

            conv_9 = slim.conv2d(res_4, 256, 3, 2)
            conv_10 = slim.conv2d(conv_9, 256, 3, 1)
            down_conv3 = slim.conv2d(res_4, 256, 3, 2)
            res_5 = tf.nn.relu(down_conv3 + conv_10)

            conv_11 = slim.conv2d(res_5, 256, 3, 1)
            conv_12 = slim.conv2d(conv_11, 256, 3, 1)
            res_6 = tf.nn.relu(res_5 + conv_12)

            conv_13 = slim.conv2d(res_6, 512, 3, 2)
            conv_14 = slim.conv2d(conv_13, 512, 3, 1)
            down_conv4 = slim.conv2d(res_6, 512, 3, 2)
            res_7 = tf.nn.relu(down_conv4 + conv_14)

            conv_15 = slim.conv2d(res_7, 512, 3, 1)
            conv_16 = slim.conv2d(conv_15, 512, 3, 1)
            res_8 = tf.nn.relu(res_7 + conv_16)
            print("res_8:", res_8)  # (?, 8, 212, 512)
            return res_8

    def Binary_cross_entropy(self, label, logits):
        y = label
        py = tf.nn.sigmoid(logits)
        py = tf.reduce_sum(py, -1)  # (?,1,lfv)
        py = tf.reduce_sim(py, 1)  # (?,lfv)
        self.loc_pre_t = py
        shape = py.get_shape().as_list()
        print("y shape", y)
        print("py,shape", py)
        pos = tf.where(tf.equal(label, 1), label, label - label)
        pos = pos * py
        log_pos = tf.where(tf.equal(pos, 0), pos.tf.log(pos))
        log_pos = tf.reduce_sum(log_pos, -1)
        neg = tf.where(tf.equal(pos, 0), label + 1, label - label)  # TODO 会不会是这里减法不等于零
        neg = neg * py
        log_neg = tf.where(tf.equal(neg, 0), neg, tf.log(1 - neg))
        log_neg = tf.reduce_sum(log_neg, -1)
        loss = -1.0 * (log_pos + log_neg) / shape[-1]
        loss = tf.reduce_sum(loss)
        print("BCE loss", loss)
        return loss

    def location_loss(self, logits, loc_labels):
        logit = tf.reduce_sum(logits, -1)
        shape = logit.get_shape().as_list()

        print("loc labels shape:", loc_labels)
        loss = tf.nn.sigmoid_cross_entory_with_logits(labels=loc_labels, logits=tf.expand_dims(logit, 1))
        loss = tf.reduce_mean(loss)
        return loss

    def mean_squared_error(self, labels, logits):
        loss = tf.losses.mean_squared_error(labels, logits)
        return loss

    def detection_loss(self, logits, labels, loc_label):
        logits = tf.reduce_sum(logits, 1)  # (?,212,4)
        loss = self.mean_squared_error(labels, logits)
        print("mse loss:", loss)
        return loss

    def classification_loss(self, logits, labels, location):
        logits = tf.reduce_sum(logits, 1)  # (?,lgv,7356)
        print("logits", logits)
        label = tf.one_hot(labels, 7356)  # (?,lfv,7356)
        loc = tf.cast(location, tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
        print("class loss", loss)
        loss = tf.reduce_sum(loss)
        return loss

    def build_net(self, is_training=True):
        with self.graph.as_default():
            if is_training:
                self.train_stage = tf.placeholder(tf.bool, shape=())
                train_image, train_label, train_location, train_classification, train_detection = self.load_tfrecord(
                    config.train_tfrecord)
                valid_image, valid_label, valid_location, valid_classification, valid_detection = self.load_tfrecord(
                    config.valid_tfrecord)
                self.x = tf.cond(self.train_stage, lambda: train_image, lambda: valid_image)
                self.label = tf.cond(self.train_stage, lambda: train_label, lambda: valid_label)
                self.location = tf.cond(self.train_stage, lambda: train_location, lambda: valid_location)
                self.classification = tf.cond(self.train_stage, lambda: train_classification,
                                              lambda: valid_classification)
                self.detection = tf.cond(self.train_stage, lambda: train_detection, lambda: valid_detection)
            else:
                self.x = tf.placeholder(tf.float32,
                                        shape=(config.batch_size, config.image_height, config.image_max_width, 1))

            self.enc = self.base_net(is_training)
            self.detection_feature, self.detection_pre = self.detection_branch(self.enc)
            self.classification_feature, self.classification_pre = self.classification_branch(self.enc)

            # loc_pre 没有sigmoid
            self.location_feature, self.location_pre = self.location_branch(self.enc, self.detection_feature,
                                                                            self.classification_feature)
            self.loc_loss = self.location_loss(self.location_pre, self.location)
            self.cla_loss = self.classification_loss(self.detection_pre, self.detection, self.location)
            self.det_loss = self.detection_loss(self.detection_pre, self.detection, self.location)

            self.loss = self.loc_loss
            # self.loss = self.loc_loss + self.cla_loss + self.dec_loss

            # cla probability
            self.cla_p = tf.nn.softmax(self.classification_pre)
            self.cla_p = tf.reduce_sum(self.cla_p, axis=1)
            # loc probability
            self.loc_p = tf.nn.sigmoid(self.location_pre)
            self.loc_p = tf.reduce_sum(self.loc_p, axis=1)
            self.loc_p = tf.reduce_sum(self.loc_p, axis=-1)
            # det pre
            self.det_p = tf.reduce_sum(self.detection_pre, 1)
            # dynamic learning rate
            global_step = tf.Variable(0, trainable=False)
            lr = config.learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)
