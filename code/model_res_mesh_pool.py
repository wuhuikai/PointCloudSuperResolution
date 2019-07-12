import math
import tensorflow as tf

from res_gcn_module import res_gcn_up, pointcnn, knn_point, group_point, pool, res_gcn_d, res_gcn_d_last


def placeholder_inputs(batch_size, num_point, up_ratio=4):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_normal = tf.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=batch_size)

    return pointclouds_pl, pointclouds_gt, pointclouds_normal, pointclouds_radius


def get_gen_model(point_cloud, is_training, scope, reuse=None, use_bn=False, use_ibn=False,
                  use_normal=False, bn_decay=None, up_ratio=4):
    with tf.variable_scope(scope, reuse=reuse):
        block_num = int(math.log2(up_ratio))

        xyz = point_cloud[:, :, 0:3]
        if use_normal:
            points = point_cloud[:, :, 3:]
        else:
            points = pointcnn(xyz, 8, 128, 3, is_training, 'module_0', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)

        for i in range(block_num):
            new_xyz, points = res_gcn_up(xyz, points, 8, 128, 12, is_training, 'module_{}'.format(i+1),
                                         bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn, up_ratio=2)
            if i < block_num - 1:
                _, idx = knn_point(8, xyz, new_xyz)
                grouped_points = group_point(points, idx)
                points = tf.reduce_mean(grouped_points, axis=2)

            xyz = new_xyz

    return xyz, None


def get_discriminator(point_cloud, is_training, scope, reuse=None, use_bn=False, use_ibn=False,
                      use_normal=False, bn_decay=None):
    with tf.variable_scope(scope, reuse=reuse):
        xyz = point_cloud[:, :, 0:3]
        if use_normal:
            points = point_cloud[:, :, 3:]
        else:
            points = pointcnn(xyz, 8, 64, 2, is_training, 'module_0', bn_decay=bn_decay,
                              use_bn=use_bn, use_ibn=use_ibn, activation=tf.nn.leaky_relu)

        block_num = int(math.log2(point_cloud.get_shape()[1].value/64)/2)
        for i in range(block_num):
            xyz, points = pool(xyz, points, 8, points.get_shape()[1].value//4)

            points = res_gcn_d(xyz, points, 8, 64, 4, is_training, 'module_{}'.format(i+1),
                               bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
        points = res_gcn_d_last(points, 1, is_training, 'module_last', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)

    return points
