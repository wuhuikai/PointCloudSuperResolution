import os
import tensorflow as tf

from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))


def query_ball_point(radius, nsample, xyz1, xyz2):
    """
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    """
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)

ops.NoGradient('QueryBallPoint')


def group_point(points, idx):
    """
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    """
    return grouping_module.group_point(points, idx)


@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]

    return [grouping_module.group_point_grad(points, idx, grad_out), None]


def knn_point(k, xyz1, xyz2):
    """
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    """
    xyz1 = tf.expand_dims(xyz1, axis=1)
    xyz2 = tf.expand_dims(xyz2, axis=2)
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)

    val, idx = tf.nn.top_k(-dist, k=k)

    return val, idx


if __name__ == '__main__':
    import time
    import numpy as np

    knn =True
    np.random.seed(100)
    pts = np.random.random((32, 512, 64)).astype('float32')
    tmp1 = np.random.random((32, 512, 3)).astype('float32')
    tmp2 = np.random.random((32, 128, 3)).astype('float32')
    with tf.device('/gpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        radius = 0.1
        nsample = 64
        if knn:
            _, idx = knn_point(nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
        else:
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
    with tf.Session('') as sess:
        now = time.time()
        for _ in range(100):
            ret = sess.run(grouped_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        print(ret)
