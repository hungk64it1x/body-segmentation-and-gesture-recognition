import numpy as np

mean_precision = 0
mean_recall = 0
mean_iou = 0
mean_dice = 0


def recall_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-07)
    return recall


def precision_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-07)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-07))


def jaccard_m(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-07)


def jaccard_(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    print("fg", intersection / (union + 1e-07))
    intersection_ = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
    union_ = np.sum(1-y_true) + np.sum(1-y_pred) - intersection_
    print("bg",intersection_ / (union_ + 1e-07))
    return intersection_ / (union_ + 1e-07)


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps  #
    den = o.sum() + t.sum() + eps  # eps
    # print(o.sum(),t.sum(),num,den)
    # print('All_voxels:240*240*155 | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num),
    #                                                                                                       int(den),
    #                                                                                                       o.sum(),
    #                                                                                                       int(t.sum())))
    return num / den


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0
    t = target > 0  # ce
    ret += (dice_score(o, t),)

    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += (dice_score(o, t),)

    # active
    o = output == 3
    t = target == 4
    ret += (dice_score(o, t),)

    return ret


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve, auc


def get_scores_v1(gts, prs, log):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    mean_acc = 0
    mean_F2 = 0
    mean_spe = 0
    mean_se = 0

    mean_iou1 = 0
    mean_dice1 = 0
    mean_mcc = 0
    mean_auc = 0

    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    for gt, pr in zip(gts, prs):
        gt = gt.round()
        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += jaccard_m(gt, pr)
        mean_dice += dice_m(gt, pr)

        tp = np.sum(gt * pr)
        fp = np.sum(pr) - tp
        fn = np.sum(gt) - tp
        tn = np.sum((1 - pr) * (1 - gt))

        mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
            4 * precision_m(gt, pr) + recall_m(gt, pr)
        )
        mean_acc += (tp + tn) / (tp + tn + fp + fn)

        mean_se += tp / (tp + fn)
        mean_spe += tn / (tn + fp)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        tn_all += tn
        pr = np.where(pr>0.5, 1, 0)
        gt   = np.where(gt>0.5, 1, 0)

        mean_iou1 += jaccard_score(gt.reshape(-1,), pr.reshape(-1,), average="binary")
        mean_dice1 += f1_score(
            gt.reshape(-1,), pr.reshape(-1,), average="binary",
        )

        mean_mcc += matthews_corrcoef(gt.reshape(-1,), pr.reshape(-1,))
        fpr, tpr, thresholds = roc_curve(gt.reshape(-1,), pr.reshape(-1,))
        mean_auc += auc(fpr, tpr)


    mean_se /= len(gts)
    mean_spe /= len(gts)
    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)
    mean_F2 /= len(gts)
    mean_acc /= len(gts)

    mean_iou1 /= len(gts)
    mean_dice1 /= len(gts)
    mean_mcc /= len(gts)
    mean_auc /= len(gts)


    log.info(
        "scores ver1: miou={:.5f} dice={:.5f} precision={:.5f} recall={:.5f} Sensitivity={:.5f} Specificity={:.5f} ACC={:.5f} F2={:.5f} miou1={:.5f} dice1={:.5f} mean_mcc={:.5f} mean_auc={:.5f}".format(
            mean_iou,
            mean_dice,
            mean_precision,
            mean_recall,
            mean_se,
            mean_spe,
            mean_acc,
            mean_F2,
            mean_iou1,
            mean_dice1,
            mean_mcc,
            mean_auc,
        )
    )

    return (mean_iou, mean_dice, mean_precision, mean_recall)


def get_scores_v2(gts, prs, log):
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for gt, pr in zip(gts, prs):
        tp = np.sum(gt * pr)
        fp = np.sum(pr) - tp
        fn = np.sum(gt) - tp
        tp_all += tp
        fp_all += fp
        fn_all += fn

    precision_all = tp_all / (tp_all + fp_all + 1e-07)
    recall_all = tp_all / (tp_all + fn_all + 1e-07)
    dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
    iou_all = (
        recall_all
        * precision_all
        / (recall_all + precision_all - recall_all * precision_all)
    )

    log.info(
        f"scores ver2: miou={iou_all}, dice={dice_all}, precision={precision_all}, recall={recall_all}"
    )

    return (iou_all, dice_all, precision_all, recall_all)
