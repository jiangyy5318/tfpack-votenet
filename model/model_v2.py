import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nms_3d'))
from tensorpack import *
import numpy as np
from tensorpack.tfutils import get_current_tower_context, gradproc, optimizer, summary
from dataset.dataset_v2 import class_mean_size
from tf_ops.nms_3d.tf_nms3d import NMS3D
import config
from utils import tf_util
from utils.tf_box_utils import tf_points_in_hull
from utils.pointnet_util import (pointnet_sa_module, pointnet_fp_module)
import tensorflow as tf


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'points'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_xyz'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_lwh'),
                tf.placeholder(tf.float32, [None, None, 8, 3], 'box3d_pts_label'),
                tf.placeholder(tf.int32, (None, None), 'semantic_labels'),
                tf.placeholder(tf.int32, (None, None), 'heading_labels'),
                tf.placeholder(tf.float32, (None, None), 'heading_residuals'),
                tf.placeholder(tf.int32, (None, None), 'size_labels'),
                tf.placeholder(tf.float32, (None, None, 3), 'size_residuals')
                ]

    @staticmethod
    def is_training():
        return get_current_tower_context().is_training

    @staticmethod
    def pointnet_backbone(l0_xyz, l0_points, is_training, bn_decay=None):

        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           scope='sa_layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           scope='sa_layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           scope='sa_layer3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           scope='sa_layer4')
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp_layer2')
        return l2_xyz, l2_points

    @staticmethod
    def hough_voting_mlp(seed, is_training=False, bn_decay=None):
        net = tf.expand_dims(seed, axis=[2])
        mlp_layers = [256, 256, 256 + 3]
        for idx, num_out_channel in enumerate(mlp_layers):
            is_last_layer = (idx == (len(mlp_layers) - 1))
            net = tf_util.conv2d(net, num_out_channel, [1, 1], padding='VALID', stride=[1, 1],
                                 bn=is_last_layer, is_training=is_training,
                                 scope='voting_mlp_%d' % idx, bn_decay=bn_decay,
                                 activation_fn=None if is_last_layer else tf.nn.relu)
        return tf.squeeze(net, axis=[2])

    @staticmethod
    def parse_outputs_to_tensor(proposals_output):
        object_pred = tf.slice(proposals_output, [0, 0, 0], [-1, -1, 2])
        center_pred = tf.slice(proposals_output, [0, 0, 2], [-1, -1, 3])
        heading_scores_pred = tf.slice(proposals_output, [0, 0, 5], [-1, -1, config.NH])
        heading_residuals_normalized_pred = tf.slice(proposals_output, [0, 0, 5 + config.NH], [-1, -1, config.NH])
        size_scores_pred = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2], [-1, -1, config.NS])
        size_residuals_normalized_pred = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2 + config.NS],
                                             [-1, -1, config.NS * 3])
        size_residuals_normalized_pred = tf.reshape(size_residuals_normalized_pred, [-1, config.PROPOSAL_NUM, config.NS, 3])
        sementic_classes_pred = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2 + config.NS * 4], [-1, -1, -1])
        return object_pred, center_pred, heading_scores_pred, heading_residuals_normalized_pred, \
            size_scores_pred, size_residuals_normalized_pred, sementic_classes_pred

    @staticmethod
    def calc_vote_loss(seeds_xyz, votes_xyz, box_center_label, box3d_pts_label):
        in_surface = tf.cast(tf_points_in_hull(seeds_xyz, box3d_pts_label), dtype=tf.float32)
        # every seed belong to one boxes.
        # in_surface = tf.cast(tf.logical_and(in_surface, tf.expand_dims(objects_num_label, axis=1)), dtype=tf.float32)
        # center_label = tf.reduce_mean(box3d_pts_label, axis=2)
        delta_x_ig = tf.norm(tf.expand_dims(votes_xyz, axis=2) -
                             tf.expand_dims(box_center_label, axis=1), axis=-1) ** 2
        loss_vote = tf.reduce_mean(tf.reduce_sum(delta_x_ig * in_surface, axis=[1]) /
                                   tf.reduce_sum(in_surface, axis=[1]))
        return loss_vote

    @staticmethod
    def get_positive_and_negative_sample(proposal_xyz, box_center_label):
        proposal2center_distance = tf.norm(tf.expand_dims(proposal_xyz, axis=[2]) -
                                           tf.expand_dims(box_center_label, axis=[1]), axis=-1)

        min_dist = tf.reduce_min(proposal2center_distance, axis=-1)
        arg_min = tf.argmin(proposal2center_distance, axis=-1)  # (B, N'), e:0~K
        votes_positive, votes_negative = tf.less(min_dist, config.POSITIVE_THRES), \
                                         tf.greater(min_dist, config.NEGATIVE_THRES)
        # (B, N'), (B, N') concat --> (B, N' ,2)
        positive_pro_idx = tf.where(votes_positive)
        positive_gt_idx = tf.concat([tf.expand_dims(positive_pro_idx[:, 0], axis=-1),
                                     tf.expand_dims(tf.gather_nd(arg_min, positive_pro_idx), axis=-1)], axis=1)
        return votes_positive, votes_negative, positive_pro_idx, positive_gt_idx

    def build_graph(self, x, bboxes_xyz, bboxes_lwh, box3d_pts_label, semantic_labels,
                    heading_labels, heading_residuals,
                    size_labels, size_residuals):
        l0_xyz = x
        l0_points = None

        # batch_size = x.get_shape()[0].value

        # point cloud feature learning
        l2_xyz, l2_points = self.pointnet_backbone(l0_xyz, l0_points, self.is_training())
        seeds_xyz = l2_xyz

        votes_xyz_points = self.hough_voting_mlp(tf.concat([seeds_xyz, l2_points], axis=2), is_training=self.is_training())

        votes_xyz, votes_points = tf.slice(votes_xyz_points, (0, 0, 0), (-1, -1, 3)), \
            tf.slice(votes_xyz_points, (0, 0, 3), (-1, -1, -1))

        proposals_xyz, proposals_output, _ = pointnet_sa_module(votes_xyz, votes_points, npoint=config.PROPOSAL_NUM, radius=0.3,
                                                                nsample=64, mlp=[128, 128, 128],
                                                                mlp2=[128, 128, 5+2*config.NH+4*config.NS+config.NC],
                                                                group_all=False,
                                                                # is_training=self.is_training(), bn_decay=None,
                                                                scope='proposal_layer')

        object_pred, center_pred, heading_scores_pred, heading_residuals_normalized_pred, size_scores_pred, \
            size_residuals_normalized_pred, sementic_classes_pred = self.parse_outputs_to_tensor(proposals_output)

        nms_iou = tf.get_variable('nms_iou', shape=[], initializer=tf.constant_initializer(0.25), trainable=False)
        if not self.is_training():
            def get_3d_bbox(box_size, heading_angle, center):
                batch_size = tf.shape(heading_angle)[0]
                c = tf.cos(heading_angle)
                s = tf.sin(heading_angle)
                zeros = tf.zeros_like(c)
                ones = tf.ones_like(c)
                rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1),
                                      tf.stack([batch_size, -1, 3, 3]))
                l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!
                corners = tf.reshape(tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,
                                               h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2,
                                               w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1),
                                     tf.stack([batch_size, -1, 3, 8]))
                return tf.einsum('ijkl,ijlm->ijmk', rotation, corners) + tf.expand_dims(center, 2)  # B * N * 8 * 3

            class_mean_size_tf = tf.constant(class_mean_size)
            size_cls_pred = tf.argmax(size_scores_pred, axis=-1)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # B * N * NS
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)
                                               * tf.reshape(size_residuals_normalized_pred,
                                                            [-1, config.PROPOSAL_NUM, config.NS, 3]), axis=2)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) * tf.maximum(
                1 + size_residual_pred, 1e-6)  # B * N * 3: size
            # with tf.control_dependencies([tf.print(size_pred[0, 0, 2])]):
            center_pred = proposals_xyz + center_pred
            heading_cls_pred = tf.argmax(heading_scores_pred, axis=-1)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot * heading_residuals_normalized_pred, axis=2)
            heading_pred = tf.floormod(
                (tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH, 2 * np.pi)

            bboxes = get_3d_bbox(size_pred, heading_pred, center_pred)  # B * N * 8 * 3,  lhw(xyz) order!!!

            nms_idx = NMS3D(bboxes, tf.reduce_max(proposals_output[..., -config.NC:], axis=-1),
                            proposals_output[..., :2],
                            nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(proposals_output[..., -config.NC:], nms_idx,
                                             name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches
            return

        loss_vote = self.calc_vote_loss(seeds_xyz, votes_xyz, bboxes_xyz, box3d_pts_label)
        loss_vote = tf.identity(loss_vote, 'vote_loss')

        votes_positive, votes_negative, positive_pro_idx, \
            positive_gt_idx = self.get_positive_and_negative_sample(proposals_xyz, bboxes_xyz)

        # object_ness loss
        object_ness_label = tf.cast(
            tf.concat((tf.expand_dims(votes_negative, [-1]),
                       tf.expand_dims(votes_positive, [-1])), axis=2), tf.float32)
        loss_object_ness = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=object_ness_label,
                                                                                    logits=object_pred), name='object_loss')

        center_dist = tf.norm(tf.gather_nd(bboxes_xyz, positive_gt_idx) -
                              tf.gather_nd(center_pred, positive_pro_idx), axis=-1)
        loss_center = huber_loss(center_dist, delta=2.0)
        loss_center = tf.identity(loss_center, name='center_loss')

        #loss_heading_score = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=tf.gather_nd(heading_scores_pred, positive_pro_idx),
        #    labels=tf.gather_nd(heading_labels, positive_gt_idx)), name='heading_loss')

        loss_heading_score = 0.0
        heading_onehot_label = tf.one_hot(tf.gather_nd(heading_labels, positive_gt_idx),
                                          depth=config.NH, on_value=1, off_value=0, axis=-1)
        heading_residual_normalized_label = tf.gather_nd(heading_residuals, positive_gt_idx) / (np.pi / config.NH)

        loss_heading_residual_normalized = huber_loss(
            tf.reduce_sum(tf.gather_nd(heading_residuals_normalized_pred, positive_pro_idx) *
                          tf.cast(heading_onehot_label, tf.float32), axis=1) -
            heading_residual_normalized_label, delta=1.0)
        loss_heading_residual_normalized = tf.identity(loss_heading_residual_normalized, name='heading_residual_loss')

        #loss_size_class = tf.reduce_mean(
        #    tf.nn.sparse_softmax_cross_entropy_with_logits(
        #        logits=tf.gather_nd(size_scores_pred, positive_pro_idx),
        #        labels=tf.gather_nd(size_labels, positive_gt_idx)), name='size_loss')
        loss_size_class = 0.0
        # size residual loss
        size_cls_gt_onehot = tf.one_hot(tf.gather_nd(size_labels, positive_gt_idx),
                                        depth=config.NS, on_value=1, off_value=0, axis=-1)
        size_cls_gt_onehot = tf.tile(tf.expand_dims(tf.cast(size_cls_gt_onehot, dtype=tf.float32), -1),
                                     [1, 1, 3])
        size_residual_gt = tf.gather_nd(size_residuals, positive_gt_idx)  # Np * 3
        size_residual_predicted = tf.reshape(tf.gather_nd(size_residuals_normalized_pred, positive_pro_idx),
                                             [-1, config.NS, 3])
        # print('size_residual_predicted:', size_residual_predicted.get_shape())
        # print('size_cls_gt_onehot:', size_cls_gt_onehot.get_shape())
        # print('size_residual_gt:', size_residual_gt.get_shape())
        size_normalized_dist = tf.norm(
            tf.reduce_sum(size_residual_predicted * tf.cast(size_cls_gt_onehot, dtype=tf.float32), axis=1) -
            tf.cast(size_residual_gt, dtype=tf.float32), axis=-1)
        loss_size_residual_normalized = huber_loss(size_normalized_dist, delta=1.0)
        loss_size_residual_normalized = tf.identity(loss_size_residual_normalized, name='size_residual_loss')

        # semantic loss:
        #loss_sementic_classes = tf.reduce_mean(
        #    tf.nn.sparse_softmax_cross_entropy_with_logits(
        #        logits=tf.gather_nd(sementic_classes_pred, positive_pro_idx),
        #        labels=tf.gather_nd(semantic_labels, positive_gt_idx)), name='semantic_loss')
        loss_sementic_classes = 0.0

        loss_box = loss_center + 0.1 * loss_heading_score + loss_heading_residual_normalized + \
               0.1 * loss_size_class + loss_size_residual_normalized

        total_cost = loss_vote + 0.5 * loss_object_ness + 1.0 * loss_box + 0.1 * loss_sementic_classes

        summary.add_moving_summary(loss_center, loss_heading_residual_normalized,
                                   loss_size_residual_normalized, loss_vote, loss_object_ness)
        total_cost = tf.identity(total_cost, name='total_cost')
        summary.add_moving_summary(total_cost)

        #summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)

        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])


if __name__=='__main__':
   pass
