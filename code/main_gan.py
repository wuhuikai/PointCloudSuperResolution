import os
import time
import socket
import argparse
import importlib

import numpy as np
import tensorflow as tf

import model_utils
import data_provider

from glob import glob
from tqdm import tqdm
from utils import pc_visulization_util


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: test]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_res_mesh_pool', help='Model name [default: model_res_mesh_pool]')
parser.add_argument('--log_dir', default='../model/model_res_mesh_pool', help='Log dir')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024] [default: 1024]')
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=80, help='Epoch to run [default: 80]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dataset', default=None)
parser.add_argument('--gan', default=False, action='store_true')
parser.add_argument('--model_path', default=None)
parser.add_argument('--lambd', default=5000, type=float)

USE_DATA_NORM = True
USE_RANDOM_INPUT = True

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
MODEL_GEN = importlib.import_module(FLAGS.model)
MODEL_DIR = FLAGS.log_dir
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
ASSIGN_MODEL_PATH = FLAGS.model_path

print(socket.gethostname())
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX


def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


def train(assign_model_path=None, bn_decay=0.95):
    step = tf.Variable(0, trainable=False)
    learning_rate = BASE_LEARNING_RATE
    # get placeholder
    pointclouds_pl, pointclouds_gt, pointclouds_gt_normal, pointclouds_radius = MODEL_GEN.placeholder_inputs(BATCH_SIZE,
                                                                                                             NUM_POINT,
                                                                                                             UP_RATIO)
    # create discriminator
    if FLAGS.gan:
        d_real = MODEL_GEN.get_discriminator(pointclouds_gt, True, 'd_1', reuse=False, use_bn=False, use_ibn=False,
                                             use_normal=False, bn_decay=bn_decay)
    # create the generator model
    pred, _ = MODEL_GEN.get_gen_model(pointclouds_pl, True, scope='generator', reuse=False, use_normal=False,
                                      use_bn=False, use_ibn=False, bn_decay=bn_decay, up_ratio=UP_RATIO)
    if FLAGS.gan:
        d_fake = MODEL_GEN.get_discriminator(pred, True, 'd_1', reuse=True, use_bn=False, use_ibn=False,
                                             use_normal=False, bn_decay=bn_decay)
    # get cd loss
    gen_loss_cd, _ = model_utils.get_cd_loss(pred, pointclouds_gt, pointclouds_radius, 1.0)
    # get gan loss
    if FLAGS.gan:
        d_loss_real = tf.reduce_mean((d_real - 1)**2)
        d_loss_fake = tf.reduce_mean(d_fake**2)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        # get loss for generator
        g_loss = tf.reduce_mean((d_fake - 1) ** 2)
    # get total loss function
    pre_gen_loss = gen_loss_cd
    if FLAGS.gan:
        pre_gen_loss = g_loss + FLAGS.lambd * pre_gen_loss
    """ Training """
    # divide trainable variables into a group for D and a group for G
    t_vars = tf.trainable_variables()
    if FLAGS.gan:
        d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]
    # optimizers
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if FLAGS.gan:
            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars, colocate_gradients_with_ops=True)
        if assign_model_path:
            learning_rate = learning_rate/10
        pre_gen_train = tf.train.AdamOptimizer(learning_rate, beta1=0.9).minimize(pre_gen_loss, var_list=g_vars,
                                                                                  colocate_gradients_with_ops=True,
                                                                                  global_step=step)
    # weight clipping
    if FLAGS.gan:
        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    # merge summary and add pointclouds summary
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss/gen_cd', gen_loss_cd)
    tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
    tf.summary.scalar('loss/pre_gen_total', pre_gen_loss)
    if FLAGS.gan:
        tf.summary.scalar('loss/d_loss_real', d_loss_real)
        tf.summary.scalar('loss/d_loss_fake', d_loss_fake)
        tf.summary.scalar('loss/d_loss', d_loss)
        tf.summary.scalar('loss/g_loss', g_loss)
    pretrain_merged = tf.summary.merge_all()

    pointclouds_image_input = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_input_summary = tf.summary.image('pointcloud_input', pointclouds_image_input, max_outputs=1)
    pointclouds_image_pred = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_pred_summary = tf.summary.image('pointcloud_pred', pointclouds_image_pred, max_outputs=1)
    pointclouds_image_gt = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_gt_summary = tf.summary.image('pointcloud_gt', pointclouds_image_gt, max_outputs=1)
    image_merged = tf.summary.merge([pointclouds_input_summary, pointclouds_pred_summary, pointclouds_gt_summary])

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, 'train'), sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'pointclouds_gt_normal': pointclouds_gt_normal,
               'pointclouds_radius': pointclouds_radius,
               'pointclouds_image_input': pointclouds_image_input,
               'pointclouds_image_pred': pointclouds_image_pred,
               'pointclouds_image_gt': pointclouds_image_gt,
               'pretrain_merged': pretrain_merged,
               'image_merged': image_merged,
               'gen_loss_cd': gen_loss_cd,
               'pre_gen_train': pre_gen_train,
               'd_optim': d_optim if FLAGS.gan else None,
               'pred': pred,
               'step': step,
               'clip': clip_D if FLAGS.gan else None,
               }
        # restore the model
        saver = tf.train.Saver(max_to_keep=6)
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(MODEL_DIR)
        if restore_epoch == 0:
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')
        else:
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'a')
            saver.restore(sess, checkpoint_path)

        ###assign the generator with another model file
        if assign_model_path is not None:
            print("Load pre-train model from %s" % assign_model_path)
            assign_saver = tf.train.Saver(
                var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
            assign_saver.restore(sess, assign_model_path)

        ##read data
        input_data, gt_data, data_radius, _ = data_provider.load_patch_data(FLAGS.dataset, skip_rate=1, norm=USE_DATA_NORM,
                                                                            use_randominput=USE_RANDOM_INPUT)

        fetchworker = data_provider.Fetcher(input_data, gt_data, data_radius, BATCH_SIZE, NUM_POINT, USE_RANDOM_INPUT,
                                            USE_DATA_NORM)
        fetchworker.start()
        for epoch in tqdm(range(restore_epoch, MAX_EPOCH + 1), ncols=55):
            log_string(LOG_FOUT, '**** EPOCH %03d ****\t' % epoch)
            train_one_epoch(sess, ops, fetchworker, train_writer, LOG_FOUT, FLAGS.gan)
            if epoch % 20 == 0:
                saver.save(sess, os.path.join(MODEL_DIR, "model"), global_step=epoch)
        fetchworker.shutdown()
        LOG_FOUT.close()


def train_one_epoch(sess, ops, fetchworker, train_writer, LOG_FOUT, gan):
    loss_sum, fetch_time = [], 0
    for _ in tqdm(range(fetchworker.num_batches)):
        start = time.time()
        batch_input_data, batch_data_gt, radius = fetchworker.fetch()
        end = time.time()
        fetch_time += end - start
        feed_dict = {ops['pointclouds_pl']: batch_input_data,
                     ops['pointclouds_gt']: batch_data_gt[:, :, 0:3],
                     ops['pointclouds_gt_normal']: batch_data_gt[:, :, 0:3],
                     ops['pointclouds_radius']: radius}
        # update D&G network
        vars = [ops['pretrain_merged'], ops['step'], ops['pre_gen_train'], ops['pred'], ops['gen_loss_cd']]
        if gan:
            vars += [ops['d_optim'], ops['clip']]
        retruns = sess.run(vars, feed_dict=feed_dict)
        summary, step, _, pred_val, gen_loss_cd = retruns[:5]
        train_writer.add_summary(summary, step)
        loss_sum.append(gen_loss_cd)

        if step % 30 == 0:
            pointclouds_image_input = pc_visulization_util.point_cloud_three_views(batch_input_data[0, :, 0:3])
            pointclouds_image_input = np.expand_dims(np.expand_dims(pointclouds_image_input, axis=-1), axis=0)
            pointclouds_image_pred = pc_visulization_util.point_cloud_three_views(pred_val[0, :, :])
            pointclouds_image_pred = np.expand_dims(np.expand_dims(pointclouds_image_pred, axis=-1), axis=0)
            pointclouds_image_gt = pc_visulization_util.point_cloud_three_views(batch_data_gt[0, :, 0:3])
            pointclouds_image_gt = np.expand_dims(np.expand_dims(pointclouds_image_gt, axis=-1), axis=0)
            feed_dict = {ops['pointclouds_image_input']: pointclouds_image_input,
                         ops['pointclouds_image_pred']: pointclouds_image_pred,
                         ops['pointclouds_image_gt']: pointclouds_image_gt,
                         }
            summary = sess.run(ops['image_merged'], feed_dict)
            train_writer.add_summary(summary, step)

    loss_sum = np.asarray(loss_sum)
    log_string(LOG_FOUT, 'step: %d mean gen_loss_cd: %f\n' % (step, round(loss_sum.mean(), 4)))
    print('read data time: %s mean gen_loss_cd: %f' % (round(fetch_time, 4), round(loss_sum.mean(), 4)))


def prediction_whole_model(use_normal=False):
    data_folder = FLAGS.dataset
    phase = data_folder.split('/')[-2] + data_folder.split('/')[-1]
    save_path = os.path.join(MODEL_DIR, 'result/' + phase)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    samples = glob(data_folder + "/*.xyz")
    samples.sort(reverse=True)
    input = np.loadtxt(samples[0])
    if use_normal:
        pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, input.shape[0], 6))
    else:
        pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, input.shape[0], 3))
    pred, _ = MODEL_GEN.get_gen_model(pointclouds_ipt, is_training=False, scope='generator', reuse=False,
                                      use_normal=use_normal, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=UP_RATIO)
    saver = tf.train.Saver()
    _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
    print(restore_model_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, restore_model_path)
        for i, item in enumerate(samples):
            input = np.loadtxt(item)
            input = np.expand_dims(input, axis=0)
            if not use_normal:
                input = input[:, :, 0:3]
            print(item, input.shape)

            pred_pl = sess.run(pred, feed_dict={pointclouds_ipt: input})
            path = os.path.join(save_path, item.split('/')[-1])
            if use_normal:
                norm_pl = np.zeros_like(pred_pl)
                data_provider.save_pl(path, np.hstack((pred_pl[0, ...], norm_pl[0, ...])))
            else:
                data_provider.save_pl(path, pred_pl[0, ...])
            path = path[:-4] + '_input.xyz'
            data_provider.save_pl(path, input[0])


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    if PHASE == 'train':
        # copy the code
        assert not os.path.exists(os.path.join(MODEL_DIR, 'code/'))
        os.makedirs(os.path.join(MODEL_DIR, 'code/'))
        os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model def

        train(assign_model_path=ASSIGN_MODEL_PATH)
    else:
        prediction_whole_model()
