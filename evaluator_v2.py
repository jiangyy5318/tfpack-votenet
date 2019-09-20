from tensorpack.callbacks import Callback
from tensorpack.utils import logger
import os,sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils import sunutils
from dataset.dataset_v2 import *
import numpy as np
import config
from shapely.geometry import Polygon
from model.ap_helper import APCalculator, parse_predictions, parse_groundtruths
type_whitelist = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                               'bookshelf', 'bathtub')
from dataset.model_util_sunrgbd import SunrgbdDatasetConfig
DATASET_CONFIG = SunrgbdDatasetConfig()
CONFIG_DICT = {'remove_empty_box':False,
               'use_3d_nms':True,
               'nms_iou':0.25,
               'use_old_type_nms':False,
               'cls_nms':True,
               'per_class_proposal': True,
               'conf_thresh':0.05,
               'dataset_config':DATASET_CONFIG}

def iou_3d(bbox1, bbox2):
    '''
    return iou of two 3d bbox
    :param bbox1: 8 * 3
    :param bbox2: 8 * 3
    :return: float
    '''
    poly1_xz = Polygon(np.stack([bbox1[:4, 0], bbox1[:4, 2]], -1))
    poly2_xz = Polygon(np.stack([bbox2[:4, 0], bbox2[:4, 2]], -1))
    iou_xz = poly1_xz.intersection(poly2_xz).area / poly1_xz.union(poly2_xz).area
    return max(iou_xz * min(bbox1[0, 1], bbox2[0, 1]) - max(bbox1[4, 1], bbox2[4, 1]), 0)


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


# idea reference: https://github.com/Cartucho/mAP
# def evaluate_one_epoch(dataset, pred_func, ious):


def evaluate_one_epoch(val_df, pred_func, output_names):
    stat_dict = {}  # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=0.25,
                                 class2type_map=DATASET_CONFIG.class2type)
    # net.eval()  # set model to eval mode (for bn and dp)

    val_df.reset_state()  # 初始化
    generator = val_df.get_data()
    for dp in generator:

        point_cloud, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label, \
            sem_cls_label, box_label_mask, vote_label, vote_label_mask, scan_idx, max_gt_bboxes = dp

        values = pred_func(point_cloud)
        end_points = dict(zip(output_names, values))
        end_points['center_label'] = center_label
        end_points['heading_class_label'] = heading_class_label
        end_points['heading_residual_label'] = heading_residual_label
        end_points['size_class_label'] = size_class_label
        end_points['size_residual_label'] = size_residual_label
        end_points['box_label_mask'] = box_label_mask
        end_points['sem_cls_label'] = sem_cls_label

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        metrics_dict = ap_calculator.compute_metrics()
        return metrics_dict

        # mean_loss = stat_dict['loss'] / float(batch_idx + 1)

        # Dump evaluation results for visualization
        # if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT % 10 == 0:
        #     MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    #         # Log statistics
    # TEST_VISUALIZER.log_scalars({key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
    #                             (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * BATCH_SIZE)
    # for key in sorted(stat_dict.keys()):
    #     log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))
    #
    # # Evaluate average precision
    # metrics_dict = ap_calculator.compute_metrics()
    # for key in metrics_dict:
    #     log_string('eval %s: %f' % (key, metrics_dict[key]))

    # mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    # return mean_loss


class Evaluator(Callback):
    def __init__(self, val_df):
        self.data_val = val_df

        self.output_names = ['pred_center', 'pred_heading_class', 'pred_heading_residual', 'pred_size_class',
                             'pred_size_residual', 'pred_sem_cls', 'sem_cls_probs']

    def _setup_graph(self):
        self.pred_func = self.trainer.get_predictor(['point_clouds'], self.output_names)

    def _before_train(self):
        logger.info('Evaluating mAP on validation set...')
        metrics_dict = evaluate_one_epoch(self.data_val, self.pred_func, self.output_names)
        for key in metrics_dict:
            print('eval %s: %f' % (key, metrics_dict[key]))

        # for iou in mAPs:
        #     logger.info("mAP{:.2f}:{:.4f}".format(iou,  mAPs[iou]))

    def _trigger(self):
        logger.info('Evaluating mAP on validation set...')
        metrics_dict = evaluate_one_epoch(self.data_val, self.pred_func, self.output_names)
        for key in metrics_dict:
            # print('eval %s: %f' % (key, metrics_dict[key]))
            self.trainer.monitors.put_scalar(key, metrics_dict[key])


if __name__ == '__main__':
    import itertools
    # from model.model_bak import Model
    #
    #
    # mAPs = eval_mAP(sunrgbd_object('/home/neil/mysunrgbd', 'training', idx_list=list(range(11, 21))), OfflinePredictor(PredictConfig(
    #         model=Model(),
    #         session_init=SaverRestore('./train_log/run/checkpoint'),
    #         input_names=['points'],
    #         output_names=['bboxes_pred', 'class_scores_pred', 'batch_idx'])), [0.25, 0.5])
    # for iou in mAPs:
    #     print("mAP{:.2f}:{:.4f}".format(iou, mAPs[iou]))
