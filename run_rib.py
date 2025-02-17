from tensorpack import *
import tensorpack.utils
import os, sys
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
import multiprocessing
from evaluator import Evaluator
import six
import numpy as np
from tensorpack.train import launch_train_with_config
from tensorpack.utils.gpu import get_num_gpu
from model.model_v1_2 import Model
from dataset.dataset_v2 import MyDataFlow

BATCH_SIZE = 8


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for _ in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='edge')
    return b


class BatchData2Biggest(BatchData):
    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData2Biggest._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData2Biggest._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _batch_numpy(data_list):
        data = data_list[0]
        if isinstance(data, six.integer_types):
            dtype = 'int32'
        elif type(data) == bool:
            dtype = 'bool'
        elif type(data) == float:
            dtype = 'float32'
        elif isinstance(data, (six.binary_type, six.text_type)):
            dtype = 'str'
        else:
            try:
                dtype = data.dtype
            except AttributeError:
                raise TypeError("Unsupported type to batch: {}".format(type(data)))
        try:
            return np.asarray(data_list, dtype=dtype)
        except Exception:  # noqa)
            try:
                largest_dim = max([d.shape[0] for d in data_list])
                data_list = [pad_along_axis(d, largest_dim) for d in data_list]
                return np.asarray(data_list, dtype=dtype)
            except Exception:
                try:
                    # open an ipython shell if possible
                    import IPython as IP; IP.embed()    # noqa
                except ImportError:
                    pass

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        first_dp = data_holder[0]
        if isinstance(first_dp, (list, tuple)):
            result = []
            for k in range(len(first_dp)):
                data_list = [x[k] for x in data_holder]
                if use_list:
                    result.append(data_list)
                else:
                    result.append(BatchData2Biggest._batch_numpy(data_list))
        elif isinstance(first_dp, dict):
            result = {}
            for key in first_dp.keys():
                data_list = [x[key] for x in data_holder]
                if use_list:
                    result[key] = data_list
                else:
                    result[key] = BatchData2Biggest._batch_numpy(data_list)
        return result


if __name__ == '__main__':
    tensorpack.utils.logger.auto_set_dir(action='k')
    # this is the official train/val split
    train_set = MyDataFlow('/data/jiangyy/sun_rgbd', 'train',
                           idx_list=[int(e.strip()) for e in open('/data/jiangyy/sun_rgbd/train/train_data_idx.txt').readlines()])
    lr_schedule = [(0, 1e-3), (80, 1e-4), (120, 1e-5)]
    # lr_schedule = [(i, 5e-5) for i in range(260)]
    # get the config which contains everything necessary in a training
    config = AutoResumeTrainConfig(
        always_resume=False,
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=QueueInput(BatchData2Biggest(PrefetchData(train_set, multiprocessing.cpu_count() // 2, multiprocessing.cpu_count() // 2), BATCH_SIZE)),
        # starting_epoch=60,
        callbacks=[
            ModelSaver(),  # save the model after every epoch
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            # compute mAP on val set
            PeriodicTrigger(Evaluator('/data/jiangyy/sun_rgbd', 'train', 1,
                                     idx_list=[int(e.strip()) for e in open('/data/jiangyy/sun_rgbd/train/val_data_idx.txt').readlines()])
                           , every_k_epochs=20, before_train=False),
            # MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        max_epoch=250,
    )

    trainer = SyncMultiGPUTrainerParameterServer(max(get_num_gpu(), 1))
    if BATCH_SIZE == 1:
        trainer = SimpleTrainer()
    launch_train_with_config(config, trainer)

