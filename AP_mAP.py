import numpy as np

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection
    return intersection / union

def compute_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    num_gt = len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(num_gt, dtype=bool)
    
    for i, pred in enumerate(pred_boxes):
        box = pred[:4]
        ious = compute_iou(box, np.array(gt_boxes))
        max_iou = np.max(ious)
        max_idx = np.argmax(ious)
        
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 合并相同召回率点并插值
    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    
    # 合并重复的召回率点，保留最大精确率
    unique_recalls, unique_indices = np.unique(sorted_recalls, return_index=True)
    sorted_precisions = np.maximum.accumulate(sorted_precisions[unique_indices[::-1]])[::-1]
    
    # 插值并计算AP
    mrec = np.concatenate(([0], unique_recalls, [1]))
    mpre = np.concatenate(([0], sorted_precisions, [0]))
    ap = np.sum(np.diff(mrec) * mpre[1:])
    
    return ap

def compute_mAP(all_gt, all_pred, iou_threshold=0.5):
    aps = []
    for class_id in all_gt:
        ap = compute_ap(all_gt[class_id], all_pred.get(class_id, []), iou_threshold)
        aps.append(ap)
    return np.mean(aps)

gt_boxes = [[10, 10, 20, 20], [30, 30, 40, 40]]
pred_boxes = [
    [12, 12, 18, 18, 0.9],  # TP
    [32, 32, 42, 42, 0.8],  # TP
    [15, 15, 25, 25, 0.7],  # FP
    [35, 35, 45, 45, 0.6]   # FP
]

ap = compute_ap(gt_boxes, pred_boxes)
print("AP:", ap)  # 输出AP值