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
from dataset.sunrgbd_detection_dataset import MAX_NUM_OBJ
from tf_ops.nms_3d.tf_nms3d import NMS3D
import config
import tensorflow as tf
vote_factor = 1
GT_VOTE_FACTOR = 3 # number of GT votes per point
FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8]


class Model(ModelDesc):
    def inputs(self):
        """
        ret_dict['point_clouds'] = point_cloud.astype(np.float32) #(N,3)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3] #(bbox,3)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64) # (bbox)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32) #(bbox)
        ret_dict['size_class_label'] = size_classes.astype(np.int64) #(bbox,)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32) #(bbox,3)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = label_mask.astype(np.int64) # (bbox,)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32) #(bbox,)
        ret_dict['vote_label'] = point_votes.astype(np.float32) #(N,9)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64) #(N)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64) #()
        ret_dict['max_gt_bboxes'] = max_bboxes #(bbox,8)
        """
        return [tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'point_clouds'),
                tf.placeholder(tf.float32, [None, MAX_NUM_OBJ, 3], 'center_label'),
                tf.placeholder(tf.int32, [None, MAX_NUM_OBJ], 'heading_class_label'),
                tf.placeholder(tf.float32, [None, MAX_NUM_OBJ], 'heading_residual_label'),
                tf.placeholder(tf.int32, (None, MAX_NUM_OBJ), 'size_class_label'),
                tf.placeholder(tf.float32, (None, MAX_NUM_OBJ, 3), 'size_residual_label'),
                tf.placeholder(tf.int32, (None, MAX_NUM_OBJ), 'sem_cls_label'),
                tf.placeholder(tf.int32, (None, MAX_NUM_OBJ), 'box_label_mask'),
                tf.placeholder(tf.int32, (None, config.POINT_NUM, 9), 'vote_label'),
                tf.placeholder(tf.int32, (None, config.POINT_NUM, 3), 'vote_label_mask'),
                tf.placeholder(tf.int32, (None, ), 'scan_idx'),
                tf.placeholder(tf.int32, (None, MAX_NUM_OBJ, 3), 'max_gt_bboxes'),
                ]

    @staticmethod
    def parse_outputs_to_tensor(proposals_output, end_points={}):
        object_pred = tf.slice(proposals_output, [0, 0, 0], [-1, -1, 2])
        end_points['objectness_scores'] = object_pred

        base_xyz = end_points['aggregated_vote_xyz']
        center = base_xyz + tf.slice(proposals_output, [0, 0, 2], [-1, -1, 3])
        end_points['center'] = center

        heading_scores = tf.slice(proposals_output, [0, 0, 5], [-1, -1, config.NH])
        heading_residuals_normalized = tf.slice(proposals_output, [0, 0, 5 + config.NH], [-1, -1, config.NH])
        end_points['heading_scores'] = heading_scores
        end_points['heading_residuals_normalized'] = heading_residuals_normalized
        end_points['heading_residuals'] = heading_residuals_normalized * (np.pi / config.NH)

        size_scores = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2], [-1, -1, config.NS])
        size_residuals_normalized = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2 + config.NS],
                                             [-1, -1, config.NS * 3])
        size_residuals_normalized = tf.reshape(size_residuals_normalized, [-1, config.PROPOSAL_NUM, config.NS, 3])

        end_points['size_scores'] = size_scores
        end_points['size_residuals_normalized'] = size_residuals_normalized
        # end_points['size_residuals']
        # end_points['size_residuals'] = None
        sem_cls_scores = tf.slice(proposals_output, [0, 0, 5 + config.NH * 2 + config.NS * 4], [-1, -1, -1])
        end_points['sem_cls_scores'] = sem_cls_scores

        #return object_pred, center, heading_scores, heading_residuals,
        return end_points

    @staticmethod
    def hough_voting_mlp(seed_xyz, seed_points):
        net = tf.expand_dims(seed_points, axis=[2])
        batch_size, num_seed, _ = seed_xyz.get_shape()
        mlp_layers = [256, 256, (256 + 3) * vote_factor]
        vote_num = num_seed * vote_factor
        for idx, num_out_channel in enumerate(mlp_layers):
            is_last_layer = (idx == (len(mlp_layers) - 1))
            net = Conv2D('voting_mlp_%d' % idx, net, num_out_channel, [1, 1], padding='VALID', stride=[1, 1],
                         activation=None if is_last_layer else BNReLU)
        net = tf.reshape(net, [-1, 1024, vote_factor, 3+256])
        offset, residual_features = tf.slice(net, (0,0,0,0), (-1,-1,-1,3)), tf.slice(net, (0,0,0,3), (-1,-1,-1,-1))
        vote_xyz = tf.expand_dims(seed_xyz, axis=2) + offset
        vote_xyz = tf.reshape(vote_xyz, [batch_size, vote_num, 3])
        vote_features = tf.expand_dims(seed_points, axis=2) + residual_features
        vote_features = tf.reshape(vote_features, (batch_size, vote_num, 256))

        return vote_xyz, vote_features

    @staticmethod
    def vote_reg_loss(seed_xyz, vote_xyz, seed_inds, vote_label, vote_label_mask):
        """
        seed_inds (B, 512)
        seed_xyz seed_points (B, 512, 3/C)
        vote_xyz vote_features (B, 512*vote_factor, 3/C)
        vote_num = num_seed * vote_factor
        GT_VOTE_FACTOR so vote_label (B,N,9)
        vote_label_mask: (B,N)
        """

        batch_size, num_seed, _ = seed_xyz.get_shape()
        # tf 1.13
        seed_gt_votes_mask = tf.cast(tf.batch_gather(vote_label_mask, seed_inds), dtype=tf.float32)

        # same with torch.gather with 3 dims
        seed_gt_votes = tf.batch_gather(vote_label, seed_inds) + tf.tile(seed_xyz, [1, 1, 3])

        vote_xyz_reshape = tf.reshape(vote_xyz, [batch_size * num_seed, vote_factor, 3])
        seed_gt_votes_reshape = tf.reshape(seed_gt_votes, [batch_size * num_seed, GT_VOTE_FACTOR, 3])

        diff = tf.expand_dims(vote_xyz_reshape, 2) - tf.expand_dims(seed_gt_votes_reshape, 1)
        dist2center = tf.reduce_sum(tf.losses.huber_loss(labels=tf.zeros_like(diff),
                                                         predictions=diff,
                                                         reduction=tf.losses.Reduction.NONE), axis=-1)  # (B, N', BB)
        dist2 = tf.reduce_min(dist2center, axis=1)
        vote_dist = tf.reduce_min(dist2, axis=1)
        vote_dist = tf.reshape(vote_dist, [batch_size, num_seed])
        vote_loss = tf.reduce_sum(vote_dist * seed_gt_votes_mask) / tf.reduce_sum(seed_gt_votes_mask + 1e-6)
        return vote_loss

    @staticmethod
    def compute_objectness_loss(proposals_xyz, center_label, end_points):
        batch_size, K, _ = proposals_xyz.get_shape()
        dist_mat = tf.reduce_sum(tf.square(tf.expand_dims(proposals_xyz, axis=[2]) -
                                           tf.expand_dims(center_label, axis=[1])),
                                 axis=-1)
        dist1 = tf.reduce_min(dist_mat, axis=2)
        object_assignment = tf.argmin(dist_mat, axis=2)
        euclid_dist1 = tf.sqrt(dist1 + 1e-6)
        objectness_label = tf.where(euclid_dist1 < NEAR_THRESHOLD,
                                    tf.ones_like(euclid_dist1),
                                    tf.zeros_like(euclid_dist1))
        objectness_mask = tf.where(tf.logical_or(euclid_dist1 < NEAR_THRESHOLD, euclid_dist1 > FAR_THRESHOLD),
                                   tf.ones_like(euclid_dist1),
                                   tf.zeros_like(euclid_dist1))
        objectness_weighted = tf.where(euclid_dist1 < NEAR_THRESHOLD,
                                       tf.ones_like(euclid_dist1) * 0.8,
                                       tf.ones_like(euclid_dist1) * 0.2)

        objectness_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['objectness_scores'],
                                                                         labels=objectness_label)
        objectness_loss = tf.reduce_sum(objectness_mask * objectness_weighted * objectness_loss) / \
                          tf.reduce_sum(objectness_mask + 1e-6)
        return objectness_loss, objectness_label, objectness_mask, object_assignment

    @staticmethod
    def compute_box_loss_and_sem_loss(end_points, center_label, heading_class_label, heading_residual_label,
                                      size_class_label, size_residual_label, sem_cls_label,
                                      object_assignment, box_label_mask, objectness_label):

        loss_points = {}

        dist_mat = tf.reduce_sum(tf.square(tf.expand_dims(end_points['center'], axis=[2]) -
                                           tf.expand_dims(center_label, axis=[1])),
                                 axis=-1)
        dist1 = tf.reduce_min(dist_mat, axis=2)
        dist2 = tf.reduce_min(dist_mat, axis=1)

        center_loss_left = tf.reduce_sum(dist1 * objectness_label) / tf.reduce_sum(objectness_label + 1e-6)
        center_loss_right = tf.reduce_sum(dist2 * box_label_mask) / tf.reduce_sum(box_label_mask + 1e-6)
        center_loss_left = tf.identity(center_loss_left, name='center_left_loss')
        center_loss_right = tf.identity(center_loss_right, name='center_right_loss')
        center_loss = tf.identity(center_loss_left + center_loss_right, name='center_loss')

        loss_points['center_loss_left'] = center_loss_left
        loss_points['center_loss_right'] = center_loss_right
        loss_points['center_loss'] = center_loss

        # heading classification loss
        heading_class_label = tf.batch_gather(heading_class_label, object_assignment)
        heading_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['heading_scores'],
            labels=heading_class_label)
        heading_cls_loss = tf.reduce_sum(heading_cls_loss * objectness_label) / tf.reduce_sum(objectness_label + 1e-6)
        heading_cls_loss = tf.identity(heading_cls_loss, 'heading_cls_loss')

        # residual loss
        heading_residual_label = tf.batch_gather(heading_residual_label, object_assignment)
        heading_residual_normalized_label = heading_residual_label / (np.pi / config.NH)
        heading_label_one_hot = tf.one_hot(heading_class_label, depth=config.NH, on_value=1, off_value=0, axis=-1)
        # heading_label_one_hot: (B,bbox,NH), dim 2 is one_hot vector
        heading_residual_normalized_pred = tf.reduce_sum(heading_label_one_hot *
                                                         end_points['size_residuals_normalized'], axis=-1)
        heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_normalized_label,
                                                     predictions=heading_residual_normalized_pred,
                                                     reduction=tf.losses.Reduction.NONE)
        heading_residual_loss = tf.reduce_sum(heading_residual_loss * objectness_label) / tf.reduce_sum(
            objectness_label + 1e-6)
        heading_residual_loss = tf.identity(heading_residual_loss, name='heading_residual_loss')

        loss_points['heading_cls_loss'] = heading_cls_loss
        loss_points['heading_residual_loss'] = heading_residual_loss

        # size classification loss
        size_class_label = tf.batch_gather(size_class_label, object_assignment)
        size_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['size_scores'],
            labels=size_class_label)
        size_cls_loss = tf.reduce_sum(size_cls_loss * objectness_label) / tf.reduce_sum(objectness_label + 1e-6)
        size_cls_loss = tf.identity(size_cls_loss, 'size_cls_loss')

        # size residual loss
        # size_class_label (B,bbox) value = [0,NS)
        # size_residual_label (B,bbox,NH)
        size_label_one_hot = tf.one_hot(size_class_label, depth=config.NS, on_value=1, off_value=0, axis=-1)
        size_label_one_hot_tiled = tf.tile(tf.expand_dims(size_label_one_hot, -1), [1, 1, 1, 3])  #
        predicted_size_residual_normalized = tf.reduce_sum(
            size_label_one_hot_tiled * end_points['size_residuals_normalized'], axis=2)
        # predicted_size_residual_normalized (B,bbox,3)
        mean_size_arr_expanded = tf.expand_dims(tf.expand_dims(class_mean_size, 0), 0)
        mean_size_label = tf.reduce_sum(mean_size_arr_expanded * size_label_one_hot_tiled, axis=2)

        size_residual_label = tf.batch_gather(size_residual_label, object_assignment)
        size_residual_label_normalized = size_residual_label / mean_size_label
        size_residual_loss = tf.reduce_mean(tf.losses.huber_loss(labels=size_residual_label_normalized,
                                                                 predictions=predicted_size_residual_normalized,
                                                                 reduction=tf.losses.Reduction.NONE),
                                            axis=-1)
        size_residual_loss = tf.reduce_sum(size_residual_loss * objectness_label) / tf.reduce_sum(
            objectness_label + 1e-6)
        size_cls_loss = tf.identity(size_cls_loss, name='size_cls_loss')

        loss_points['size_cls_loss'] = size_cls_loss
        loss_points['size_residual_loss'] = size_residual_loss

        # semantic loss
        sem_cls_label = tf.batch_gather(sem_cls_label, object_assignment)  #
        sem_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['sem_cls_scores'],
                                                           labels=sem_cls_label,
                                                           name='sem_cls_loss'))
        sem_cls_loss = tf.reduce_sum(sem_cls_loss * objectness_label) / tf.reduce_sum(objectness_label + 1e-6)
        sem_cls_loss = tf.identity(sem_cls_loss, name='sem_cls_loss')

        loss_points['sem_cls_loss'] = sem_cls_loss

        return loss_points
        # wd_cost = tf.multiply(1e-5,
        #                       regularize_cost('.*/W', tf.nn.l2_loss),
        #                       name='regularize_loss')

        # total_cost = vote_loss + 0.5 * objectness_loss + 1. * box_loss + 0.1 * sem_cls_loss

        # total_cost = tf.add_n([total_cost, wd_cost], name='total_loss')

    # def build_graph(self, x, bboxes_xyz_labels, bboxes_lwh_labels, box3d_pts_labels, semantic_labels,
    #                 heading_cls_labels, heading_residual_labels,
    #                 size_cls_labels, size_residual_labels):
    # def build_graph(self, x, bboxes_xyz, bboxes_lwh, semantic_labels, heading_labels, heading_residuals, size_labels, size_residuals):
    def build_graph(self, x, center_label, heading_class_label, heading_residual_label, size_class_label,
                    size_residual_label,sem_cls_label, box_label_mask, vote_label, vote_label_mask, scan_idx, max_gt_bboxes):

        l0_xyz = x
        l0_points = None if x.shape[2] <=3 else x[:,:,3:]

        end_points = {}

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False, scope='sa1',
                                                           use_xyz=True, normalize_xyz=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=32,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa2',
                                                           use_xyz=True, normalize_xyz=True)
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=16,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa3',
                                                           use_xyz=True, normalize_xyz=True)
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=16,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa4',
                                                           use_xyz=True, normalize_xyz=True)
        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp1')
        seed_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp2')
        seed_xyz = l2_xyz
        # fp2_inds
        fp2_inds = l1_indices[:, 0:tf.shape(seed_xyz)[1]]

        # Voting Module layers
        # seed_xyz seed_points (B, 512, 3/C)
        vote_xyz, vote_features = self.hough_voting_mlp(seed_xyz, seed_points)
        # vote_xyz vote_features (B, 512*vote_factor, 3/C)

        # votes_xyz_points = tf.concat([seeds_xyz, seeds_points], 2) + offset
        # votes_xyz, votes_points = tf.slice(votes_xyz_points, (0, 0, 0), (-1, -1, 3)), \
        #     tf.slice(votes_xyz_points, (0, 0, 3), (-1, -1, -1))

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(vote_xyz, vote_features,
                                                                       npoint=config.PROPOSAL_NUM,
                                                                       radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                       mlp2=[128, 128,5+2 * config.NH+4 * config.NS+config.NC],
                                                                       group_all=False, scope='proposal',
                                                                       use_xyz=True, normalize_xyz=True)

        end_points = self.parse_outputs_to_tensor(proposals_output, end_points)

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
            size_cls_pred = tf.argmax(end_points['size_scores'], axis=-1)  # (B, proposal_num)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # (B, proposal_num, NS)
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)  # (B, proposal_num, NS, -1)
                                               * tf.reshape(end_points['size_residuals_normalized'], [-1, config.PROPOSAL_NUM, config.NS, 3]), axis=2)
            #  size_residual_pred : (B, proposal_num, 3)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) \
                         * tf.maximum(1 + size_residual_pred, 1e-6)  # B * N * 3: size

            # calc center
            heading_cls_pred = tf.argmax(end_points['heading_scores'], axis=-1)  # (B, proposal_num)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)  # (B, proposal_num, NH)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot * end_points['heading_residuals_normalized'], axis=2)
            heading_pred = tf.floormod((tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH,
                                       2 * np.pi)

            # with tf.control_dependencies([tf.print(size_residual_pred[0, :10, :]), tf.print(size_pred[0, :10, :])]):
            bboxes = get_3d_bbox(size_pred, heading_pred, end_points['center'])  # B * N * 8 * 3,  lhw(xyz) order!!!

            # bbox_corners = tf.concat([bboxes[:, :, 6, :], bboxes[:, :, 0, :]], axis=-1)  # B * N * 6,  lhw(xyz) order!!!
            # with tf.control_dependencies([tf.print(bboxes[0, 0])]):
            nms_idx = NMS3D(bboxes, tf.reduce_max(end_points['sem_cls_scores'], axis=-1), end_points['objectness_scores'], nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(end_points['sem_cls_scores'], nms_idx, name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches

        vote_loss = self.vote_reg_loss(seed_xyz, vote_xyz, fp2_inds, vote_label, vote_label_mask)

        objectness_loss, objectness_label, objectness_mask, object_assignment = self.compute_objectness_loss(
            proposals_xyz, center_label, end_points)

        loss_points = self.compute_box_loss_and_sem_loss(end_points, center_label,
                                                         heading_class_label, heading_residual_label,
                                                         size_class_label, size_residual_label, sem_cls_label,
                                                         object_assignment, box_label_mask, objectness_label)
        # dist_mat = tf.reduce_sum(tf.square(tf.expand_dims(end_points['center'], axis=[2]) -
        #                                    tf.expand_dims(center_label, axis=[1])),
        #                          axis=-1)
        # dist1 = tf.reduce_min(dist_mat, axis=2)
        # dist2 = tf.reduce_min(dist_mat, axis=1)
        #
        # center_loss_left = tf.reduce_sum(dist1 * objectness_label) / tf.reduce_sum(objectness_label + 1e-6)
        # center_loss_right = tf.reduce_sum(dist2 * box_label_mask) / tf.reduce_sum(box_label_mask + 1e-6)
        # center_loss_left = tf.identity(center_loss_left, name='center_left_loss')
        # center_loss_right = tf.identity(center_loss_right, name='center_right_loss')
        # center_loss = tf.identity(center_loss_left + center_loss_right, name='center_loss')
        #
        # # heading classification loss
        # heading_class_label = tf.batch_gather(heading_class_label, object_assignment)
        # heading_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=end_points['heading_scores'],
        #    labels=heading_class_label)
        # heading_cls_loss = tf.reduce_sum(heading_cls_loss * objectness_label) / tf.reduce_sum(objectness_label+1e-6)
        #
        # # residual loss
        # heading_residual_label = tf.batch_gather(heading_residual_label, object_assignment)
        # heading_residual_normalized_label = heading_residual_label / np.pi / config.NH
        # heading_label_one_hot = tf.one_hot(heading_class_label, depth=config.NH, on_value=1, off_value=0, axis=-1)
        # # heading_label_one_hot: (B,bbox,NH), dim 2 is one_hot vector
        # heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_normalized_label,
        #                                              predictions=heading_label_one_hot * end_points['size_residuals_normalized'],
        #                                              reduction=tf.losses.Reduction.NONE)
        # heading_residual_loss = tf.reduce_sum(heading_residual_loss * objectness_label) / tf.reduce_sum(objectness_label+1e-6)
        # heading_residual_loss = tf.identity(heading_residual_loss, name='heading_residual_loss')
        #
        # # size classification loss
        # size_class_label = tf.batch_gather(size_class_label, object_assignment)
        # size_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=end_points['size_scores'],
        #     labels=size_class_label)
        # size_cls_loss = tf.reduce_sum(size_cls_loss * objectness_label) / tf.reduce_sum(objectness_label+1e-6)
        # size_cls_loss = tf.identity(size_cls_loss, 'size_cls_loss')
        #
        # # size residual loss
        # # size_class_label (B,bbox) value = [0,NS)
        # # size_residual_label (B,bbox,NH)
        # size_label_one_hot = tf.one_hot(size_class_label, depth=config.NS, on_value=1, off_value=0, axis=-1)
        # size_label_one_hot_tiled = tf.tile(tf.expand_dims(size_label_one_hot, -1), [1, 1, 1, 3]) #
        # predicted_size_residual_normalized = tf.reduce_sum(size_label_one_hot_tiled*end_points['size_residuals_normalized'],axis=2)
        # # predicted_size_residual_normalized (B,bbox,3)
        # mean_size_arr_expanded = tf.expand_dims(tf.expand_dims(class_mean_size, 0), 0)
        # mean_size_label = tf.reduce_sum(mean_size_arr_expanded * size_label_one_hot_tiled, axis=2)
        # size_residual_label_normalized = size_residual_label / mean_size_label
        # size_residual_loss = tf.reduce_mean(tf.losses.huber_loss(labels=size_residual_label_normalized,
        #                                                          predictions=predicted_size_residual_normalized,
        #                                                          reduction=tf.losses.Reduction.NONE),
        #                                     axis=-1)
        # size_residual_loss = tf.reduce_sum(size_residual_loss * objectness_label) / tf.reduce_sum(objectness_label+1e-6)

        # box loss
        box_loss = tf.identity(loss_points['center_loss'] + 0.1 * loss_points['heading_cls_loss'] +
                               loss_points['heading_residual_loss']+ 0.1 * loss_points['size_cls_loss'] +
                               loss_points['size_residual_loss'], name='box_loss')


        wd_cost = tf.multiply(1e-5,
                              regularize_cost('.*/W', tf.nn.l2_loss),
                              name='regularize_loss')

        total_cost = vote_loss + 0.5 * objectness_loss + 1. * box_loss + 0.1 * loss_points['sem_cls_loss']

        total_cost = tf.add_n([total_cost, wd_cost], name='total_loss')

        summary.add_moving_summary(total_cost,
                                   vote_loss,
                                   objectness_loss, box_loss,
                                   loss_points['center_loss'],
                                   loss_points['center_loss_left'], loss_points['center_loss_right'],
                                   loss_points['heading_cls_loss'], loss_points['heading_residual_loss'],
                                   loss_points['size_cls_loss'], loss_points['size_residual_loss'],
                                   loss_points['sem_cls_loss'],
                                   wd_cost,
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
