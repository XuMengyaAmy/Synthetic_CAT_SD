import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('Agg')

# Metrics https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
class _StreamMetrics(object):

    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds): # 有很多个样本，一个样本一个样本地处理
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    # val_metrics = StreamSegMetrics(n_classes)
    # metrics=val_metrics

    # _, prediction = outputs.max(dim=1) # prediction class
    # labels = labels.cpu().numpy()
    # prediction = prediction.cpu().numpy()
    # metrics.update(labels, prediction)

    # score = metrics.get_results()
    # val_score = score
    # logger.info(val_metrics.to_str(val_score))

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != "Confusion Matrix":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def _fast_hist(self, label_true, label_pred):
        # print('self.n_classes', self.n_classes) # self.n_classes 8
        # print('label_true', label_true) # label_true [0 0 0 ... 0 0 0]
        mask = (label_true >= 0) & (label_true < self.n_classes)
        # print('mask', mask) # mask [ True  True  True ...  True  True  True]
        # mask对应的是flatten后 gt label不为0的像素位置，也就是foreground区域的位置
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        # print('hist', hist.shape) # hist (8, 8)
# hist [[1.01469301e+08 7.57774000e+05 2.70099000e+05 1.86220000e+05
#   2.78000000e+02 0.00000000e+00 0.00000000e+00 8.07170000e+04]
#  [1.27827000e+06 1.07568500e+06 6.24860000e+05 2.32379700e+06
#   9.07000000e+02 0.00000000e+00 0.00000000e+00 1.31172000e+05]
#  [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
#  [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
#  [9.67263000e+05 1.78802000e+05 8.54110000e+05 4.61790000e+04
#   4.14400000e+03 0.00000000e+00 1.03600000e+03 2.49072000e+05]
#  [8.30143000e+05 1.39395400e+06 1.65249000e+05 3.90790000e+04
#   1.10340000e+04 0.00000000e+00 0.00000000e+00 6.35600000e+04]
#  [1.57206200e+06 2.22572400e+06 4.26563000e+05 1.89920000e+04
#   1.68400000e+03 0.00000000e+00 6.66100000e+03 7.10409000e+05]
#  [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]

# >>> a = np.ones((3,4))
# >>> print(a)
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
# >>> b = a.sum(axis=1)
# >>> print(b)
# [4. 4. 4.]
# >>> c = a.sum(axis=0)
# >>> print(c)
# [3. 3. 3. 3.]
# >>>
        gt_sum = hist.sum(axis=1) # 横着加
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])  ########### MY: mean_iu = np.mean(iu[mask]) or mean_iu = np.mean(iu)

        # >>> import numpy as np
        # >>> mask = np.array([ True, True, False, False,  True,  True,  True, False])
        # >>> iu = np.array([0.94358682, 0.09820185, 0,         0,         0.00103956, 0, 0.00225528, 0])
        # >>> mean_iu = np.mean(iu[mask])
        # >>> print(iu[mask])
        # [0.94358682 0.09820185 0.00103956 0.         0.00225528]
        # >>> print(mean_iu)
        # 0.209016702
        # >>> mean_iou_2 = np.mean(iu)
        # >>> print(mean_iou_2)
        # 0.13063543875
        # >>> 
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # print('mask', mask)
        # print('iu', iu)
        # mask [ True  True False False  True  True  True False] # 8 elements
        # iu [0.94358682 0.09820185 0.         0.         0.00103956 0.
        #  0.00225528 0.        ]

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(
            zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)])
        )

        # >>> seq = ['one', 'two', 'three']
        # >>> for i, element in enumerate(seq):
        # ...     print i, element
        # ...
        # 0 one
        # 1 two
        # 2 three

    # >>> mask = np.array([ True, True, False, False,  True,  True,  True, False])
    # >>> print(mask)
    # [ True  True False False  True  True  True False]
    # >>> iu = np.array([0.94358682, 0.09820185, 0,         0,         0.00103956, 0, 0.00225528, 0])
    # >>> cls_iu = dict(zip(range(8), [iu[i] if m else "X" for i, m in enumerate(mask)]))
    # >>> print(cls_iu)
    # {0: 0.94358682, 1: 0.09820185, 2: 'X', 3: 'X', 4: 0.00103956, 5: 0.0, 6: 0.00225528, 7: 'X'}
    # >>> 

    # X may because in validation set, there is no such sample with this class (just a guess).
    # X means that class do not appear in test set

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Confusion Matrix": self.confusion_matrix_to_fig()
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1) +
                                                      0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix', ylabel='True label', xlabel='Predicted label')

        fig.tight_layout()
        return fig


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
