import tensorflow as tf


def tf_points_in_hull(tf_points, tf_boxes):
    """tf_points (B ,N, 3)
       tf_boxes (B, M, 8, 3)
    """

    tf_centers = tf.reduce_mean(tf_boxes, axis=2)
    # tf_centers: (B, M, 3)

    # boxes shape is [8, 3], The below 3 codes calc vectors of lwh
    vector_1 = tf.slice(tf_boxes, [0, 0, 0, 0], [-1, -1, 1, -1]) - tf.slice(tf_boxes, [0, 0, 1, 0], [-1, -1, 1, -1])
    vector_2 = tf.slice(tf_boxes, [0, 0, 0, 0], [-1, -1, 1, -1]) - tf.slice(tf_boxes, [0, 0, 3, 0], [-1, -1, 1, -1])
    vector_3 = tf.slice(tf_boxes, [0, 0, 0, 0], [-1, -1, 1, -1]) - tf.slice(tf_boxes, [0, 0, 4, 0], [-1, -1, 1, -1])

    # concat 3 vector3
    tf_vectors = tf.concat([vector_1, vector_2, vector_3], axis=2)  # (B, M, 3, 3)
    # euclidean norm (B, M, 3)
    tf_whl = tf.norm(tf_vectors, axis=3, keepdims=True)  # (B, M, 3, 1)
    # Normalization 3 vectors
    tf_vectors_normed = tf_vectors / tf_whl
    # calc points distance from centers.
    tf_dist2centers = tf.abs(tf.expand_dims(tf_points, axis=2) - tf.expand_dims(tf_centers, axis=1))
    # calc the projected components for 3 axis lwh.
    # tf_dist2centers: (B, N, M, 3) --> (B, N, M, 1, 3)
    # tf_vectors_normed: (B, M, 3, 3) --> (B, 1, M, 3, 3)
    #
    tf_dist2centers_proj = tf.expand_dims(tf_dist2centers, axis=3) * tf.expand_dims(tf_vectors_normed, axis=1)
    tf_dist2centers_proj_distance = tf.norm(tf_dist2centers_proj, axis=4)
    # tf_dist2centers_proj_distance: (B, N, M, 3)
    tf_idx_in_hull_3d = tf.cast(tf.less(tf_dist2centers_proj_distance,
                                        tf.expand_dims(tf.squeeze(tf_whl, axis=[3]), axis=1) / 2.0), tf.int32)
    tf_idx_in_hull = tf.equal(tf.reduce_mean(tf_idx_in_hull_3d, axis=-1, keepdims=False), 1)
    # print('tf_vectors:', tf_vectors.get_shape())
    # print('tf_whl:', tf_whl.get_shape())
    # print('tf_vectors_normed:', tf_vectors_normed.get_shape())
    # print('points:', tf_points.get_shape(), 'tf_centers:', tf_centers.get_shape())
    # print('tf_dist2centers:', tf_dist2centers.get_shape())
    # print('tf_dist2centers_proj:', tf_dist2centers_proj.get_shape())
    # print('tf_dist2centers_proj_distance:', tf_dist2centers_proj_distance.get_shape())
    # print('tf_idx_in_hull_3d:', tf_idx_in_hull_3d.get_shape())
    # print('tf_idx_in_hull:', tf_idx_in_hull.get_shape())

    return tf_idx_in_hull




def tf_main_points_in_hull():
    import os
    import numpy as np
    import sys
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR,'../utils'))
    import utils

    def get_2048_points(depth):
        num_point = depth.shape[0]
        pc_in_box_fov = depth[:, :3]
        if num_point > 2048:
            choice = np.random.choice(depth.shape[0], 2048, replace=False)
            pc_in_box_fov = depth[choice, :3]
        return pc_in_box_fov

    pc_in_box_fov = get_2048_points(np.loadtxt('/Users/jiangyy/projects/VoteNet/utils/test_datas/000001.txt'))
    calib = utils.SUNRGBD_Calibration("/Users/jiangyy/projects/VoteNet/utils/test_datas/000001.txt.calib")
    objects = utils.read_sunrgbd_label("/Users/jiangyy/projects/VoteNet/utils/test_datas/000001.txt.label")
    box_list = []
    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib)
        box_list.append(box3d_pts_3d)

    tf_points1 = tf.Variable(pc_in_box_fov, dtype=tf.float32)
    # print(box_list)
    e = np.array(box_list)
    tf_boxes1 = tf.Variable(np.array(box_list), dtype=tf.float32)
    tf_points2 = tf.expand_dims(tf_points1, axis=0)
    tf_boxes2 = tf.expand_dims(tf_boxes1, axis=0)
    tf_idx_in_hull = tf_points_in_hull(tf_points2, tf_boxes2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_idx_in_hull_1 = sess.run(tf_idx_in_hull)
        print(tf_idx_in_hull_1.shape)
    tf_idx_in_hull_2 = tf_idx_in_hull_1.sum(axis=2)

    import mayavi.mlab as mlab
    fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
    mlab.points3d(pc_in_box_fov[:, 0], pc_in_box_fov[:, 1], pc_in_box_fov[:, 2], tf_idx_in_hull_2[0], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    sys.path.append(os.path.join(BASE_DIR, '../mayavi'))
    from viz_util import draw_gt_boxes3d
    draw_gt_boxes3d(box_list, fig, color=(1, 0, 0))
    mlab.orientation_axes()
    mlab.show()


if __name__ == '__main__':
    tf_main_points_in_hull()



