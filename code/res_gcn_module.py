import tensorflow as tf

from tf_ops.grouping.tf_grouping import group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point


def conv2d(input, n_cout, name, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    return tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)


def batch_norm(input, is_training, name, bn_decay=None, use_bn=False, use_ibn=False):
    if use_bn:
        return tf.layers.batch_normalization(input, name=name, training=is_training, momentum=bn_decay)

    if use_ibn:
        return tf.contrib.layers.instance_norm(input, scope=name)

    return input


def group(xyz, points, k, dilation=1, use_xyz=False):
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation]

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, k, 3)
    grouped_xyz -= tf.expand_dims(xyz, 2)  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def pool(xyz, points, k, npoint):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    _, idx = knn_point(k, xyz, new_xyz)
    new_points = tf.reduce_max(group_point(points, idx), axis=2)

    return new_xyz, new_points


def pointcnn(xyz, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, activation=tf.nn.relu):
    with tf.variable_scope(scope):
        _, grouped_points, _ = group(xyz, None, k)

        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                grouped_points = conv2d(grouped_points, n_cout, name='conv_xyz')
                if idx == n_blocks - 1:
                    return tf.reduce_max(grouped_points, axis=2)
                else:
                    grouped_points = batch_norm(grouped_points, is_training, 'bn_xyz',
                                                bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                    grouped_points = activation(grouped_points)


def res_gcn_up(xyz, points, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None, up_ratio=2):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                points = conv2d(center_points, n_cout, name='conv_center')
                # Neighbor Conv
                grouped_points_nn = conv2d(grouped_points, n_cout, name='conv_neighbor')
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

                if idx == n_blocks - 1:
                    # Center Conv
                    points_xyz = conv2d(center_points, 3*up_ratio, name='conv_center_xyz')
                    # Neighbor Conv
                    grouped_points_xyz = conv2d(grouped_points, 3*up_ratio, name='conv_neighbor_xyz')
                    # CNN
                    new_xyz = tf.reduce_mean(tf.concat([points_xyz, grouped_points_xyz], axis=2), axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value, up_ratio, 3])
                    new_xyz = new_xyz + tf.expand_dims(xyz, axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value*up_ratio, 3])

                    return new_xyz, points


def res_gcn_d(xyz, points, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.leaky_relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                points = conv2d(center_points, n_cout, name='conv_center')
                # Neighbor Conv
                grouped_points_nn = conv2d(grouped_points, n_cout, name='conv_neighbor')
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

    return points


def res_gcn_d_last(points, n_cout, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False):
    with tf.variable_scope(scope):
        points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
        points = tf.nn.leaky_relu(points)
        center_points = tf.expand_dims(points, axis=2)
        points = tf.squeeze(conv2d(center_points, n_cout, name='conv_center'), axis=2)

        return points
