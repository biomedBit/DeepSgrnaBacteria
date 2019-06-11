from __future__ import print_function
import tensorflow as tf
import sonnet as snt
from tensorflow.contrib import slim
from utils import data_list_batch_1_23_4
from tqdm import tqdm
import scipy.stats as stats

def create_init_op():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init_op

'''
def load_ckpt(sess, model_dir, variables_to_restore=None):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_path = ckpt.model_checkpoint_path
    if variables_to_restore is None:
        variables_to_restore = slim.get_variables_to_restore()
    restore_op, restore_fd = slim.assign_from_checkpoint(
        model_path, variables_to_restore)
    sess.run(restore_op, feed_dict=restore_fd)
    print('{0} loaded'.format(model_path))
'''


with tf.variable_scope('ontar'):
    inputs_sg = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 4])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    channel_size = [8, 32, 64, 64, 256, 256]
    betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{0}'.format(i)) for i in
        range(1, len(channel_size))]

    e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
    ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
    e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
    ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
    e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
    ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
    e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
    ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
    e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
    ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')

    encoder = [None, e1, e2, e3, e4, e5]
    encoder_bn_u = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]

    hu0 = inputs_sg
    u_lst = [hu0]
    hu_lst = [hu0]

    for i in range(1, len(channel_size) - 1):
        hu_pre = hu_lst[i - 1]
        pre_u = encoder[i](hu_pre)
        u = encoder_bn_u[i](pre_u, False, test_local_stats=False)
        hu = tf.nn.relu(u + betas[i])
        u_lst.append(u)
        hu_lst.append(hu)

    hu_m1 = hu_lst[-1]
    pre_u_last = encoder[-1](hu_m1)
    u_last = encoder_bn_u[-1](pre_u_last, False, test_local_stats=False)
    u_last = u_last + betas[-1]
    hu_last = tf.nn.relu(u_last)
    u_lst.append(u_last)
    hu_lst.append(hu_last)

    # regressor
    cls_channel_size = [512, 512, 1024, 1]
    e6 = snt.Conv2D(cls_channel_size[0], kernel_shape=[1, 3], stride=2, name='e_6')
    ebn6l = snt.BatchNorm(decay_rate=0.99, name='ebn_6l')
    e7 = snt.Conv2D(cls_channel_size[1], kernel_shape=[1, 3], name='e_7')
    ebn7l = snt.BatchNorm(decay_rate=0.99, name='ebn_7l')
    e8 = snt.Conv2D(cls_channel_size[2], kernel_shape=[1, 3], padding='VALID', name='e_8')
    ebn8l = snt.BatchNorm(decay_rate=0.99, name='ebn_8l')
    e9 = snt.Conv2D(cls_channel_size[3], kernel_shape=[1, 1], name='e_9')

    cls_layers = [None, e6, e7, e8, e9]
    cls_bn_layers = [None, ebn6l, ebn7l, ebn8l]

    hl0 = hu_last
    l_lst = [hl0]
    hl_lst = [hl0]
    for i in range(1, len(cls_channel_size)):
        hl_pre = hl_lst[i - 1]
        pre_l = cls_layers[i](hl_pre)
        l = cls_bn_layers[i](pre_l, False, test_local_stats=False)
        hl = tf.nn.relu(l)
        l_lst.append(l)
        hl_lst.append(hl)

    hl_m1 = hl_lst[-1]
    l_last = cls_layers[-1](hl_m1)
    l_lst.append(l_last)

    y = tf.squeeze(l_last, axis=[1, 2, 3])
    y_1 = tf.squeeze(y_,axis=[1])
    cost = tf.reduce_mean(tf.pow((y_1- y),2))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)
'''
def build_ontar_model(inputs_sg, scope='ontar'):
    with tf.variable_scope(scope):
        channel_size = [8, 32, 64, 64, 256, 256]
        betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{'+ i +'}') for i in
                          range(1, len(channel_size))]

        e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
        ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
        e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
        ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
        e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
        ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
        e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
        ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
        e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
        ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')

        encoder = [None, e1, e2, e3, e4, e5]
        encoder_bn_u = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]

        hu0 = inputs_sg
        u_lst = [hu0]
        hu_lst = [hu0]

        for i in range(1, len(channel_size) - 1):
            hu_pre = hu_lst[i - 1]
            pre_u = encoder[i](hu_pre)
            u = encoder_bn_u[i](pre_u, False, test_local_stats=False)
            hu = tf.nn.relu(u + betas[i])
            u_lst.append(u)
            hu_lst.append(hu)

        hu_m1 = hu_lst[-1]
        pre_u_last = encoder[-1](hu_m1)
        u_last = encoder_bn_u[-1](pre_u_last, False, test_local_stats=False)
        u_last = u_last + betas[-1]
        hu_last = tf.nn.relu(u_last)
        u_lst.append(u_last)
        hu_lst.append(hu_last)

        # classifier
        cls_channel_size = [512, 512, 1024, 2]
        e6 = snt.Conv2D(cls_channel_size[0], kernel_shape=[1, 3], stride=2, name='e_6')
        ebn6l = snt.BatchNorm(decay_rate=0.99, name='ebn_6l')
        e7 = snt.Conv2D(cls_channel_size[1], kernel_shape=[1, 3], name='e_7')
        ebn7l = snt.BatchNorm(decay_rate=0.99, name='ebn_7l')
        e8 = snt.Conv2D(cls_channel_size[2], kernel_shape=[1, 3], padding='VALID', name='e_8')
        ebn8l = snt.BatchNorm(decay_rate=0.99, name='ebn_8l')
        e9 = snt.Conv2D(cls_channel_size[3], kernel_shape=[1, 1], name='e_9')

        cls_layers = [None, e6, e7, e8, e9]
        cls_bn_layers = [None, ebn6l, ebn7l, ebn8l]

        hl0 = hu_last
        l_lst = [hl0]
        hl_lst = [hl0]
        for i in range(1, len(cls_channel_size)):
            hl_pre = hl_lst[i - 1]
            pre_l = cls_layers[i](hl_pre)
            l = cls_bn_layers[i](pre_l, False, test_local_stats=False)
            hl = tf.nn.relu(l)
            l_lst.append(l)
            hl_lst.append(hl)

        hl_m1 = hl_lst[-1]
        l_last = cls_layers[-1](hl_m1)
        hl_last = tf.nn.softmax(l_last)
        l_lst.append(l_last)
        hl_lst.append(hl_last)

        sig_l = tf.squeeze(hl_last, axis=[1, 2])[:, 1]
        return sig_l


def build_offtar_reg_model(inputs_sg, inputs_ot, scope='offtar'):
    with tf.variable_scope(scope):
        channel_size = [8, 32, 64, 64, 256, 256]
        with tf.variable_scope('sg'):
            betas_sg = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{'+ i +'}') for i in
                                 range(1, len(channel_size))]
            e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
            ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
            e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
            ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
            e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
            ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
            e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
            ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
            e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
            ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')
            encoder_sg = [None, e1, e2, e3, e4, e5]
            encoder_bn_u_sg = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]
        with tf.variable_scope('ot'):
            betas_ot = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{'+i+'}') for i in
                                 range(1, len(channel_size))]
            e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
            ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
            e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
            ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
            e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
            ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
            e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
            ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
            e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
            ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')
            encoder_ot = [None, e1, e2, e3, e4, e5]
            encoder_bn_u_ot = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]
        # sg first layer
        hu0_sg = inputs_sg
        u_lst_sg = [hu0_sg]
        hu_lst_sg = [hu0_sg]
        # ot first layer
        hu0_ot = inputs_ot
        u_lst_ot = [hu0_ot]
        hu_lst_ot = [hu0_ot]
        for i in range(1, len(channel_size) - 1):
            # sg layer building
            hu_pre_sg = hu_lst_sg[i - 1]
            pre_u_sg = encoder_sg[i](hu_pre_sg)
            u_sg = encoder_bn_u_sg[i](pre_u_sg, False, test_local_stats=False)
            hu_sg = tf.nn.relu(u_sg + betas_sg[i])
            u_lst_sg.append(u_sg)
            hu_lst_sg.append(hu_sg)
            # ot layer building
            hu_pre_ot = hu_lst_ot[i - 1]
            pre_u_ot = encoder_ot[i](hu_pre_ot)
            u_ot = encoder_bn_u_ot[i](pre_u_ot, False, test_local_stats=False)
            hu_ot = tf.nn.relu(u_ot + betas_ot[i])
            u_lst_ot.append(u_ot)
            hu_lst_ot.append(hu_ot)
        # sg last layer
        hu_m1_sg = hu_lst_sg[-1]
        pre_u_last_sg = encoder_sg[-1](hu_m1_sg)
        u_last_sg = encoder_bn_u_sg[-1](pre_u_last_sg, False, test_local_stats=False)
        u_last_sg = u_last_sg + betas_sg[-1]
        hu_last_sg = tf.nn.relu(u_last_sg)
        u_lst_sg.append(u_last_sg)
        hu_lst_sg.append(hu_last_sg)
        # ot last layer
        hu_m1_ot = hu_lst_ot[-1]
        pre_u_last_ot = encoder_ot[-1](hu_m1_ot)
        u_last_ot = encoder_bn_u_ot[-1](pre_u_last_ot, False, test_local_stats=False)
        u_last_ot = u_last_ot + betas_ot[-1]
        hu_last_ot = tf.nn.relu(u_last_ot)
        u_lst_ot.append(u_last_ot)
        hu_lst_ot.append(hu_last_ot)

        hu_last = tf.concat([hu_last_sg, hu_last_ot], axis=3)
        cls_channel_size = [512, 512, 1024, 1]
        e6 = snt.Conv2D(cls_channel_size[0], kernel_shape=[1, 3], stride=2, name='e_6')
        ebn6l = snt.BatchNorm(decay_rate=0.99, name='ebn_6l')
        e7 = snt.Conv2D(cls_channel_size[1], kernel_shape=[1, 3], name='e_7')
        ebn7l = snt.BatchNorm(decay_rate=0.99, name='ebn_7l')
        e8 = snt.Conv2D(cls_channel_size[2], kernel_shape=[1, 3], padding='VALID', name='e_8')
        ebn8l = snt.BatchNorm(decay_rate=0.99, name='ebn_8l')
        e9 = snt.Conv2D(cls_channel_size[3], kernel_shape=[1, 1], name='e_9')

        cls_layers = [None, e6, e7, e8, e9]
        cls_bn_layers = [None, ebn6l, ebn7l, ebn8l]

        hl0 = hu_last
        l_lst = [hl0]
        hl_lst = [hl0]
        for i in range(1, len(cls_channel_size)):
            hl_pre = hl_lst[i - 1]
            pre_l = cls_layers[i](hl_pre)
            l = cls_bn_layers[i](pre_l, False, test_local_stats=False)
            hl = tf.nn.relu(l)
            l_lst.append(l)
            hl_lst.append(hl)

        hl_m1 = hl_lst[-1]
        l_last = cls_layers[-1](hl_m1)
        l_lst.append(l_last)

        logits_l = tf.squeeze(l_last, axis=[1, 2, 3])
        return logits_l


def build_offtar_model(inputs_sg, inputs_ot, scope='offtar'):
    with tf.variable_scope(scope):
        channel_size = [8, 32, 64, 64, 256, 256]
        with tf.variable_scope('sg'):
            betas_sg = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{'+i+'}') for i in
                                 range(1, len(channel_size))]
            e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
            ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
            e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
            ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
            e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
            ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
            e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
            ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
            e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
            ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')
            encoder_sg = [None, e1, e2, e3, e4, e5]
            encoder_bn_u_sg = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]
        with tf.variable_scope('ot'):
            betas_ot = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name='beta_{'+i+'}') for i in
                                 range(1, len(channel_size))]
            e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
            ebn1u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_1u')
            e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
            ebn2u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_2u')
            e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
            ebn3u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_3u')
            e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
            ebn4u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_4u')
            e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
            ebn5u = snt.BatchNorm(decay_rate=0, offset=False, name='ebn_5u')
            encoder_ot = [None, e1, e2, e3, e4, e5]
            encoder_bn_u_ot = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]
        # sg first layer
        hu0_sg = inputs_sg
        u_lst_sg = [hu0_sg]
        hu_lst_sg = [hu0_sg]
        # ot first layer
        hu0_ot = inputs_ot
        u_lst_ot = [hu0_ot]
        hu_lst_ot = [hu0_ot]
        for i in range(1, len(channel_size) - 1):
            # sg layer building
            hu_pre_sg = hu_lst_sg[i - 1]
            pre_u_sg = encoder_sg[i](hu_pre_sg)
            u_sg = encoder_bn_u_sg[i](pre_u_sg, False, test_local_stats=False)
            hu_sg = tf.nn.relu(u_sg + betas_sg[i])
            u_lst_sg.append(u_sg)
            hu_lst_sg.append(hu_sg)
            # ot layer building
            hu_pre_ot = hu_lst_ot[i - 1]
            pre_u_ot = encoder_ot[i](hu_pre_ot)
            u_ot = encoder_bn_u_ot[i](pre_u_ot, False, test_local_stats=False)
            hu_ot = tf.nn.relu(u_ot + betas_ot[i])
            u_lst_ot.append(u_ot)
            hu_lst_ot.append(hu_ot)
        # sg last layer
        hu_m1_sg = hu_lst_sg[-1]
        pre_u_last_sg = encoder_sg[-1](hu_m1_sg)
        u_last_sg = encoder_bn_u_sg[-1](pre_u_last_sg, False, test_local_stats=False)
        u_last_sg = u_last_sg + betas_sg[-1]
        hu_last_sg = tf.nn.relu(u_last_sg)
        u_lst_sg.append(u_last_sg)
        hu_lst_sg.append(hu_last_sg)
        # ot last layer
        hu_m1_ot = hu_lst_ot[-1]
        pre_u_last_ot = encoder_ot[-1](hu_m1_ot)
        u_last_ot = encoder_bn_u_ot[-1](pre_u_last_ot, False, test_local_stats=False)
        u_last_ot = u_last_ot + betas_ot[-1]
        hu_last_ot = tf.nn.relu(u_last_ot)
        u_lst_ot.append(u_last_ot)
        hu_lst_ot.append(hu_last_ot)

        hu_last = tf.concat([hu_last_sg, hu_last_ot], axis=3)
        cls_channel_size = [512, 512, 1024, 2]
        e6 = snt.Conv2D(cls_channel_size[0], kernel_shape=[1, 3], stride=2, name='e_6')
        ebn6l = snt.BatchNorm(decay_rate=0.99, name='ebn_6l')
        e7 = snt.Conv2D(cls_channel_size[1], kernel_shape=[1, 3], name='e_7')
        ebn7l = snt.BatchNorm(decay_rate=0.99, name='ebn_7l')
        e8 = snt.Conv2D(cls_channel_size[2], kernel_shape=[1, 3], padding='VALID', name='e_8')
        ebn8l = snt.BatchNorm(decay_rate=0.99, name='ebn_8l')
        e9 = snt.Conv2D(cls_channel_size[3], kernel_shape=[1, 1], name='e_9')

        cls_layers = [None, e6, e7, e8, e9]
        cls_bn_layers = [None, ebn6l, ebn7l, ebn8l]

        hl0 = hu_last
        l_lst = [hl0]
        hl_lst = [hl0]
        for i in range(1, len(cls_channel_size)):
            hl_pre = hl_lst[i - 1]
            pre_l = cls_layers[i](hl_pre)
            l = cls_bn_layers[i](pre_l, False, test_local_stats=False)
            hl = tf.nn.relu(l)
            l_lst.append(l)
            hl_lst.append(hl)

        hl_m1 = hl_lst[-1]
        l_last = cls_layers[-1](hl_m1)
        hl_last = tf.nn.softmax(l_last)
        l_lst.append(l_last)
        hl_lst.append(hl_last)

        sig_l = tf.squeeze(hl_last, axis=[1, 2])[:, 1]
        return sig_l


class DCModelOntar:
    def __init__(self, sess, ontar_model_dir, is_reg=False, seq_feature_only=False):
        self.sess = sess
        if seq_feature_only:
            self.inputs_sg = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 4])
        else:
            self.inputs_sg = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 8])
        if is_reg:
            self.pred_ontar = build_ontar_reg_model(self.inputs_sg)
        else:
            self.pred_ontar = build_ontar_model(self.inputs_sg)
        all_vars = slim.get_variables_to_restore()
        on_vars = {v.op.name[6:]: v for v in all_vars if v.name.startswith('ontar')}
        sess.run(create_init_op())
        load_ckpt(sess, ontar_model_dir, variables_to_restore=on_vars)

    def ontar_predict(self, x, channel_first=True):
        if channel_first:
            x = x.transpose([0, 2, 3, 1])
        fd = {self.inputs_sg: x}
        yp = self.sess.run(self.pred_ontar, feed_dict=fd)
        return yp.ravel()


class DCModelOfftar:
    def __init__(self, sess, offtar_model_dir, is_reg=False):
        self.sess = sess
        self.inputs_sg = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 8])
        self.inputs_ot = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 8])
        if is_reg:
            self.pred_offtar = build_offtar_reg_model(self.inputs_sg, self.inputs_ot)
        else:
            self.pred_offtar = build_offtar_model(self.inputs_sg, self.inputs_ot)
        all_vars = slim.get_variables_to_restore()
        off_vars = {v.op.name[7:]: v for v in all_vars if v.name.startswith('offtar')}
        sess.run(create_init_op())
        load_ckpt(sess, offtar_model_dir, variables_to_restore=off_vars)

    def offtar_predict(self, xsg, xot, channel_first=True):
        if channel_first:
            xsg = xsg.transpose([0, 2, 3, 1])
            xot = xot.transpose([0, 2, 3, 1])
        fd = {self.inputs_sg: xsg, self.inputs_ot: xot}
        yp = self.sess.run(self.pred_offtar, feed_dict=fd)
        return yp.ravel()
'''


	
def train_DeepCRISPR(trainData,trainDataAll,testDataAll,ENZ): 
	sess = tf.InteractiveSession()
	sess.run(create_init_op())      
	for e in range(110): # for e in range(20):
		for featureData,targetLabel in data_list_batch_1_23_4(trainData):
			sess.run(train_step, feed_dict={inputs_sg: featureData, y_:targetLabel})
			
		targetLabelList = []
		outputList = []
		trainMSEList = []
		trainMSEloss = 0
		
		for featureData,targetLabel in trainDataAll:
			output = y.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
			trainMSEloss_temp = cost.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
			trainMSEList.append(trainMSEloss_temp)
			targetLabel_temp = targetLabel.squeeze().tolist()
			output_temp = output.squeeze().tolist()
			if not(type(output_temp) == list):
				output_temp = [output_temp]
			if not(type(targetLabel_temp) == list):
				targetLabel_temp = [targetLabel_temp]
			targetLabelList = targetLabelList + targetLabel_temp
			outputList = outputList + output_temp
			
		trainMSEloss = sum(trainMSEList) / len(trainMSEList)
		train_spcc, _ = stats.spearmanr(targetLabelList, outputList) #spearmar
	
		featureData = testDataAll[0]
		targetLabel = testDataAll[1]
		output = y.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
		testMSEloss = cost.eval(feed_dict={inputs_sg: featureData, y_: targetLabel})
		targetLabel_st = targetLabel.squeeze().tolist()
		output_cdnst = output.squeeze().tolist()
		test_spcc, _ = stats.spearmanr(targetLabel_st, output_cdnst)
		
		print("epch {0}: train_loss {1}, train_spcc {2} || test_loss {3}, test_spcc {4}".format(e,trainMSEloss,train_spcc,testMSEloss,test_spcc))
