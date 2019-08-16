import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from tensorpack import *
import numpy as np
from tensorpack.tfutils import get_current_tower_context, gradproc, optimizer, summary
from utils.pointnet_util_new import (pointnet_sa_module, pointnet_fp_module)
from utils.tf_box_utils import tf_points_in_hull
from dataset.dataset_v2 import (class_mean_size)
from tf_ops.nms_3d.tf_nms3d import NMS3D
import config
import tensorflow as tf


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
                tf.placeholder(tf.float32, (None, None, 3), 'size_residuals'),
                ]

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
    def hough_voting_mlp(seed):
        net = tf.expand_dims(seed, axis=[2])
        mlp_layers = [256, 256, 256 + 3]
        for idx, num_out_channel in enumerate(mlp_layers):
            is_last_layer = (idx == (len(mlp_layers) - 1))
            net = Conv2D('voting_mlp_%d' % idx, net, num_out_channel, [1, 1], padding='VALID', stride=[1, 1],
                         activation=None if is_last_layer else BNReLU)
        return tf.squeeze(net, axis=[2])

    @staticmethod
    def vote_reg_loss(seeds_xyz, votes_xyz, bboxes_xyz, box3d_pts_label):
        """
            seeds_xyz: (B, N', 3)
            bboxes_xyz:  (B, BB, 3)
            box3d_pts_label: (B, BB, 8, 3)
            calc:
            dist2center: (B, N', BB)
            votes_assignment: (B, N'), value \in [0,BB)
            in_surface: (B, N', BB)
        """

        dist2center = tf.norm(tf.expand_dims(seeds_xyz, 2) - tf.expand_dims(bboxes_xyz, 1), axis=-1)  # (B, N', BB)
        votes_assignment = tf.argmin(dist2center, axis=-1, output_type=tf.int32)  # B * N, int

        bboxes_xyz_to_votes_idx = tf.stack([tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), -1),
                                                    [1, tf.shape(votes_assignment)[1]]),
                                            votes_assignment], 2)  # B * N * 3

        in_surface = tf_points_in_hull(seeds_xyz, box3d_pts_label)  # (B, N', BB)
        in_surface_to_votes_idx = tf.stack([tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), axis=-1),
                                                    [1, tf.shape(votes_assignment)[1]]),
                                            tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[1]), axis=0),
                                                    [tf.shape(votes_assignment)[0], 1]),
                                            votes_assignment], axis=2)

        vote_reg_loss = tf.reduce_mean(tf.norm(votes_xyz -
                                               tf.gather_nd(bboxes_xyz, bboxes_xyz_to_votes_idx), ord=1, axis=-1) *
                                       tf.cast(tf.gather_nd(in_surface, in_surface_to_votes_idx), tf.float32),
                                       name='vote_reg_loss')
        return vote_reg_loss

    def build_graph(self, x, bboxes_xyz_labels, bboxes_lwh_labels, box3d_pts_labels, semantic_labels,
                    heading_cls_labels, heading_residual_labels,
                    size_cls_labels, size_residual_labels):
    # def build_graph(self, x, bboxes_xyz, bboxes_lwh, semantic_labels, heading_labels, heading_residuals, size_labels, size_residuals):
        l0_xyz = x
        l0_points = None

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False, scope='sa1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa4')
        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp1')
        seeds_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp2')
        seeds_xyz = l2_xyz

        # Voting Module layers
        offset = self.hough_voting_mlp(seeds_points)

        votes_xyz_points = tf.concat([seeds_xyz, seeds_points], 2) + offset
        votes_xyz, votes_points = tf.slice(votes_xyz_points, (0, 0, 0), (-1, -1, 3)), \
            tf.slice(votes_xyz_points, (0, 0, 3), (-1, -1, -1))

        vote_reg_loss = self.vote_reg_loss(seeds_xyz, votes_xyz, bboxes_xyz_labels, box3d_pts_labels)

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(votes_xyz, votes_points, npoint=config.PROPOSAL_NUM,
                                                                radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                mlp2=[128, 128, 5+2 * config.NH+4 * config.NS+config.NC],
                                                                group_all=False, scope='proposal')

        object_scores_pred, center_scores_pred, heading_scores_pred, heading_residuals_normalized_pred, size_scores_pred, \
            size_residuals_normalized_pred, sementic_classes_pred = self.parse_outputs_to_tensor(proposals_output)

        nms_iou = tf.get_variable('nms_iou', shape=[], initializer=tf.constant_initializer(0.25), trainable=False)
        if not get_current_tower_context().is_training:

            # class_mean_size_tf = tf.constant(class_mean_size)
            # def tf_class2angle():
            #     pass
            #
            # def tf_class2size(size_scores_pred, residual_normalized, NS):
            #     """ size_scores_pred: (B, proposal_num, NS)
            #         residual_normalized: (B, proposal_num, NS, 3)
            #         NS:
            #
            #     """
            #     size_cls_pred = tf.argmax(size_scores_pred, axis=-1)  # (B, proposal_num)
            #     size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=NS, axis=-1)  # (B, proposal_num, NS)
            #     size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)  # (B, proposal_num, NS, -1)
            #                                        * tf.reshape(size_residuals_normalized_pred,
            #                                                     [-1, tf.shape(size_residuals_normalized_pred)[1], NS, 3]), axis=2)
            #     #  size_residual_pred : (B, proposal_num, 3)
            #     size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) * tf.maximum(
            #         1 + size_residual_pred, 1e-1)  # B * N * 3: size
            #     return size_pred

            def get_3d_bbox(box_size, heading_angle, center):
                batch_size = tf.shape(heading_angle)[0]
                c = tf.cos(heading_angle)
                s = tf.sin(heading_angle)
                zeros = tf.zeros_like(c)
                ones = tf.ones_like(c)
                rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1), tf.stack([batch_size, -1, 3, 3]))
                l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!
                corners = tf.reshape(tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,
                                               h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2,
                                               w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1),
                                     tf.stack([batch_size, -1, 3, 8]))
                return tf.einsum('ijkl,ijlm->ijmk', rotation, corners) + tf.expand_dims(center, 2)  # B * N * 8 * 3

            class_mean_size_tf = tf.constant(class_mean_size)
            size_cls_pred = tf.argmax(size_scores_pred, axis=-1)  # (B, proposal_num)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # (B, proposal_num, NS)
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)  # (B, proposal_num, NS, -1)
                                               * tf.reshape(size_residuals_normalized_pred, [-1, config.PROPOSAL_NUM, config.NS, 3]), axis=2)
            #  size_residual_pred : (B, proposal_num, 3)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) \
                         * tf.maximum(1 + size_residual_pred, 1e-6)  # B * N * 3: size

            # calc center
            center_pred = proposals_xyz + center_scores_pred  # (B, proposal_num, 3)
            heading_cls_pred = tf.argmax(heading_scores_pred, axis=-1)  # (B, proposal_num)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)  # (B, proposal_num, NH)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot * heading_residuals_normalized_pred, axis=2)
            heading_pred = tf.floormod((tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH,
                                       2 * np.pi)

            # with tf.control_dependencies([tf.print(size_residual_pred[0, :10, :]), tf.print(size_pred[0, :10, :])]):
            bboxes = get_3d_bbox(size_pred, heading_pred, center_pred)  # B * N * 8 * 3,  lhw(xyz) order!!!

            # bbox_corners = tf.concat([bboxes[:, :, 6, :], bboxes[:, :, 0, :]], axis=-1)  # B * N * 6,  lhw(xyz) order!!!
            # with tf.control_dependencies([tf.print(bboxes[0, 0])]):
            nms_idx = NMS3D(bboxes, tf.reduce_max(sementic_classes_pred, axis=-1), object_scores_pred, nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(sementic_classes_pred, nms_idx, name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches

            return

        dist_mat = tf.norm(tf.expand_dims(proposals_xyz, axis=[2]) - tf.expand_dims(bboxes_xyz_labels, axis=[1]), axis=-1)
        min_dist = tf.reduce_min(dist_mat, axis=-1)
        bboxes_assignment = tf.argmin(dist_mat, axis=-1)  # (B, N'), e:0~K

        thres_mid = tf.reduce_mean(min_dist, axis=-1, keepdims=True)
        thres_min = tf.reduce_min(min_dist, axis=-1, keepdims=True)
        thres_max = tf.reduce_max(min_dist, axis=-1, keepdims=True)
        POSITIVE_THRES, NEGATIVE_THRES = (thres_mid + thres_min) / 2.0, (thres_mid + thres_max) / 2.0

        positive_pro_idx = tf.where(min_dist < POSITIVE_THRES)
        negative_pro_idx = tf.where(min_dist > NEGATIVE_THRES)

        positive_gt_idx = tf.stack([positive_pro_idx[:, 0], tf.gather_nd(bboxes_assignment, positive_pro_idx)], axis=1)

        # objectiveness loss
        pos_obj_cls_score = tf.gather_nd(object_scores_pred, positive_pro_idx)
        pos_obj_cls_gt = tf.ones([tf.shape(positive_pro_idx)[0]], dtype=tf.int32)
        neg_obj_cls_score = tf.gather_nd(object_scores_pred, negative_pro_idx)
        neg_obj_cls_gt = tf.zeros([tf.shape(negative_pro_idx)[0]], dtype=tf.int32)
        pos_obj_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pos_obj_cls_score,labels=pos_obj_cls_gt))
        neg_obj_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neg_obj_cls_score,labels=neg_obj_cls_gt))
        obj_cls_loss = tf.identity((pos_obj_cls_loss + neg_obj_cls_loss) / 2.0, name='obj_cls_loss')
        obj_correct = tf.concat([tf.cast(tf.nn.in_top_k(pos_obj_cls_score, pos_obj_cls_gt, 1), tf.float32),
                                 tf.cast(tf.nn.in_top_k(neg_obj_cls_score, neg_obj_cls_gt, 1), tf.float32)], axis=0,
                                name='obj_correct')
        obj_accuracy = tf.reduce_mean(obj_correct, name='obj_accuracy')

        # center regression losses
        # positive_gt_idx: (B,N',2)
        center_gt = tf.gather_nd(bboxes_xyz_labels, positive_gt_idx)
        center_pred = tf.gather_nd(proposals_xyz + center_scores_pred, positive_gt_idx)
        center_loss_left = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=center_gt,
                                                                             predictions=center_pred,
                                                                             reduction=tf.losses.Reduction.NONE),
                                                        axis=-1),
                                          name='center_left_loss')

        bboxes_assignment_dual = tf.argmin(dist_mat, axis=1)
        proposals_to_boxes_idx = tf.stack(
            [tf.tile(tf.expand_dims(tf.range(tf.shape(bboxes_assignment_dual, out_type=tf.int64)[0]), -1),
                     [1, tf.shape(bboxes_assignment_dual)[1]]),
             bboxes_assignment_dual], 2)  # B * BB' * 2

        nearest_proposals_assigned2box = tf.gather_nd(proposals_xyz + center_scores_pred, proposals_to_boxes_idx)
        center_loss_right = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=bboxes_xyz_labels,
                                                                              predictions=nearest_proposals_assigned2box,
                                                                              reduction=tf.losses.Reduction.NONE),
                                                         axis=-1),
                                           name='center_right_loss')
        center_loss = tf.identity(center_loss_left + center_loss_right, name='center_loss')

        # heading classification loss
        heading_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
           logits=tf.gather_nd(heading_scores_pred, positive_pro_idx),
           labels=tf.gather_nd(heading_cls_labels, positive_gt_idx)), name='heading_cls_loss')

        # heading residual loss
        heading_onehot_label = tf.one_hot(tf.gather_nd(heading_cls_labels, positive_gt_idx),
                                          depth=config.NH, on_value=1, off_value=0, axis=-1)
        heading_residual_normalized_label = tf.gather_nd(heading_residual_labels, positive_gt_idx) / (np.pi / config.NH)
        # same name: heading_residuals_normalized_pred
        heading_residuals_normalized_pred = tf.reduce_sum(tf.gather_nd(heading_residuals_normalized_pred, positive_pro_idx) *
                                                          tf.cast(heading_onehot_label, tf.float32), axis=1)

        heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_normalized_label,
                                                     predictions=heading_residuals_normalized_pred,
                                                     reduction=tf.losses.Reduction.MEAN)
        heading_residual_loss = tf.identity(heading_residual_loss, name='heading_residual_loss')

        # size loss
        size_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
               logits=tf.gather_nd(size_scores_pred, positive_pro_idx),
               labels=tf.gather_nd(size_cls_labels, positive_gt_idx)), name='size_cls_loss')

        # size residual loss
        size_cls_gt_onehot = tf.one_hot(tf.gather_nd(size_cls_labels, positive_gt_idx),
                                        depth=config.NS, on_value=1, off_value=0, axis=-1)  # (P, NS)
        size_cls_gt_onehot_tiled = tf.tile(tf.expand_dims(size_cls_gt_onehot, axis=-1), [1, 1, 3])
        # size_cls_gt_onehot_tiled: (P,NS,3), [[...,[0,0,0],[1,1,1]]]

        size_residuals_normalized_pred = tf.reshape(tf.gather_nd(size_residuals_normalized_pred, positive_pro_idx),
                                                    [-1, config.NS, 3])
        size_residuals_normalized_pred = tf.reduce_sum(size_residuals_normalized_pred * tf.cast(size_cls_gt_onehot_tiled, dtype=tf.float32), axis=1)
        mean_size_arr_expand = tf.expand_dims(tf.constant(class_mean_size, dtype=tf.float32), 0)  # (1, NS, 3)
        mean_size_label = tf.reduce_sum(size_cls_gt_onehot_tiled * mean_size_arr_expand, axis=[1])  # (P, 3)
        size_residual_label_normalized = tf.gather_nd(size_residual_labels, positive_gt_idx) / mean_size_label
        size_residual_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=size_residual_label_normalized,
                                                                               predictions=size_residuals_normalized_pred,
                                                                               reduction=tf.losses.Reduction.NONE),
                                                          axis=-1), name='size_residual_loss')
        # box loss
        box_loss = tf.identity(center_loss +
                               0.1 * heading_cls_loss + heading_residual_loss
                               + 0.1 * size_cls_loss + size_residual_loss, name='box_loss')
        # semantic loss
        sem_cls_score = tf.gather_nd(sementic_classes_pred, positive_pro_idx)
        sem_cls_gt = tf.gather_nd(semantic_labels, positive_gt_idx)
        sem_cls_loss = tf.reduce_mean(
           tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_cls_score, labels=sem_cls_gt, name='sem_cls_loss'))

        sem_correct = tf.cast(tf.nn.in_top_k(sem_cls_score, sem_cls_gt, 1), tf.float32, name='sem_correct')
        sem_accuracy = tf.reduce_mean(sem_correct, name='sem_accuracy')

        wd_cost = tf.multiply(1e-5,
                              regularize_cost('.*/W', tf.nn.l2_loss),
                              name='regularize_loss')

        total_cost = vote_reg_loss + 0.5 * obj_cls_loss + 1. * box_loss + 0.1 * sem_cls_loss
        total_cost = tf.identity(total_cost, name='total_loss')
        summary.add_moving_summary(total_cost,
                                   vote_reg_loss,
                                   obj_cls_loss, box_loss,
                                   center_loss, center_loss_left, center_loss_right,
                                   heading_cls_loss, heading_residual_loss,
                                   size_cls_loss, size_residual_loss,
                                   sem_cls_loss,
                                   wd_cost,
                                   obj_accuracy, sem_accuracy,
                                   decay=0)
        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
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
