import os

import tensorflow as tf

from tf_ops.CD import tf_nndistance


def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step, ckpt.model_checkpoint_path
    else:
        return 0, None


def get_cd_loss(pred, gt, radius, alpha):
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    # dists_forward is for each element in gt, the cloest distance to this element
    CD_dist = alpha * dists_forward + (1-alpha) * dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist / radius
    cd_loss = tf.reduce_mean(CD_dist_norm)

    return cd_loss, None
