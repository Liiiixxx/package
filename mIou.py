import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

class IOUMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
# 计算混淆矩阵
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(self.num_classes * label_true[mask].astype(int) + label_pred[mask], minlength = self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # self.hist += self._fast_hist(predictions, gts,3)
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0)- np.diag(self.hist))
        mean_iou = np.nanmean(iou)
        return  mean_iou


def get_iou(mask_name, predict):
    image_mask = cv2.imread(mask_name, 3)
    predict = predict.astype(np.int16)
    Iou = IOUMetric(3)
    Iou.add_batch(predict, image_mask)
    m = Iou.evaluate()
    print('%s:iou=%f' % (mask_name, m))
    return m