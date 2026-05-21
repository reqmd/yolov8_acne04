import torch

def make_anchors(img_size=640, strides=[8, 16, 32]):
    """
    Возвращает anchor points в пикселях для всех масштабов.
    img_size: размер входного изображения
    """
    anchor_points = []
    stride_tensor = []

    for stride in strides:
        h = w = img_size // stride  # размер сетки
        # Центры ячеек: 0.5, 1.5, 2.5, ...
        sx = torch.arange(w, dtype=torch.float32) + 0.5
        sy = torch.arange(h, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')

        anchors = torch.stack([
            grid_x.flatten(),
            grid_y.flatten()
        ], dim=1)  # (H*W, 2)

        anchor_points.append(anchors)
        stride_tensor.append(torch.full((h * w, 1), stride))

    anchor_points = torch.cat(anchor_points, dim=0)  # (33600, 2)
    stride_tensor = torch.cat(stride_tensor, dim=0)  # (33600, 1)

    return anchor_points, stride_tensor

def bbox2dist(matched_boxes, anchor_points, stride_tensor, reg_max=16):
    """
    Конвертирует GT боксы x1y1x2y2 в расстояния ltrb для DFL.
    matched_boxes:  (B, N, 4) в пикселях
    anchor_points:  (N, 2)    центры ячеек
    stride_tensor:  (N, 1)    stride каждого anchor
    """
    device = matched_boxes.device
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)
    
    anc = (anchor_points * stride_tensor).unsqueeze(0)  # (1, N, 2)

    lt = anc - matched_boxes[..., :2]   # расстояние до левого/верхнего края
    rb = matched_boxes[..., 2:] - anc   # расстояние до правого/нижнего края

    # Переводим в единицы бинов (делим на stride)
    str = stride_tensor.unsqueeze(0)    # (1, N, 1)
    lt  = lt / str
    rb  = rb / str

    dist = torch.cat([lt, rb], dim=-1)  # (B, N, 4)

    # Клампируем в диапазон [0, reg_max - 1]
    return dist.clamp(0, reg_max - 1)

def dist2bbox(pred_dist, anchor_points, stride_tensor):
    """
    pred_dist:     (B, 64, N)
    anchor_points: (N, 2)
    stride_tensor: (N, 1)
    """
    B, _, N = pred_dist.shape
    reg_max = 16

    pred_dist = pred_dist.permute(0, 2, 1)            # (B, N, 64)
    pred_dist = pred_dist.view(B, N, 4, reg_max)      # (B, N, 4, 16)
    pred_dist = pred_dist.softmax(dim=3)

    bins = torch.arange(reg_max, dtype=torch.float32, device=pred_dist.device)
    pred_dist = (pred_dist * bins).sum(dim=3)          # (B, N, 4) — [l, t, r, b]

    # Добавляем batch dimension для broadcast
    anc = anchor_points.unsqueeze(0).to(pred_dist.device)   # (1, N, 2)
    str = stride_tensor.unsqueeze(0).to(pred_dist.device)   # (1, N, 1)

    lt = pred_dist[..., :2]   # (B, N, 2)
    rb = pred_dist[..., 2:]   # (B, N, 2)

    x1y1 = anc * str - lt * str
    x2y2 = anc * str + rb * str

    return torch.cat([x1y1, x2y2], dim=-1)  # (B, N, 4)


def decode_predictions(outputs, anchor_points, stride_tensor):
    """
    Собирает выходы всех трёх масштабов и декодирует их.
    outputs: [(box_p3, cls_p3), (box_p4, cls_p4), (box_p5, cls_p5)]
    """
    # Собираем все предсказания
    all_box = torch.cat([box for box, _ in outputs], dim=2)  # (B, 64, N)
    all_cls = torch.cat([cls for _, cls in outputs], dim=2)  # (B,  1, N)

    # Декодируем боксы
    pred_boxes = dist2bbox(all_box, anchor_points, stride_tensor)  # (B, N, 4)

    return pred_boxes, all_cls, all_box  # all_box нужен для DFL loss

def xywh2xyxy(boxes, img_size):
    """
    Конвертирует нормализованный xywh → пиксельный x1y1x2y2
    boxes: (n, 4)
    """
    cx = boxes[:, 0] * img_size
    cy = boxes[:, 1] * img_size
    w  = boxes[:, 2] * img_size
    h  = boxes[:, 3] * img_size
    return torch.stack([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ], dim=1)


def tal_matcher(pred_boxes, pred_cls, targets, anchor_points,
                stride_tensor, img_size, topk=10, alpha=0.5, beta=6.0):
    """
    pred_boxes:    (B, N, 4)  — декодированные боксы в пикселях
    pred_cls:      (B, 1, N)  — сырые логиты cls
    targets:       (M, 6)     — [batch_idx, cls, x, y, w, h]
    anchor_points: (N, 2)     — центры ячеек (в ячейках)
    stride_tensor: (N, 1)     — stride каждого anchor
    topk:          сколько лучших anchors брать на объект
    """
    B = pred_boxes.shape[0]
    N = anchor_points.shape[0]
    device = pred_boxes.device

    positive_mask  = torch.zeros(B, N, dtype=torch.bool,    device=device)
    matched_boxes  = torch.zeros(B, N, 4,                   device=device)
    matched_scores = torch.zeros(B, N,                      device=device)
    anc_px = (anchor_points * stride_tensor).to(device)  # (N, 2)

    for b in range(B):
        obj_mask  = targets[:, 0] == b
        obj_boxes = targets[obj_mask, 2:]

        if len(obj_boxes) == 0:
            continue

        gt_xyxy = xywh2xyxy(obj_boxes, img_size).to(device)  # (n, 4)
        n_obj   = len(gt_xyxy)

        ax = anc_px[:, 0].unsqueeze(0)  # (1, N)
        ay = anc_px[:, 1].unsqueeze(0)  # (1, N)

        in_box = (
            (ax > gt_xyxy[:, 0:1]) &
            (ax < gt_xyxy[:, 2:3]) &
            (ay > gt_xyxy[:, 1:2]) &
            (ay < gt_xyxy[:, 3:4])
        )  # (n_obj, N)

        iou = compute_iou(
            pred_boxes[b].unsqueeze(0).expand(n_obj, -1, -1),
            gt_xyxy.unsqueeze(1).expand(-1, N, -1)
        )  # (n_obj, N)

        cls_prob = pred_cls[b, 0].unsqueeze(0).sigmoid()
        scores   = (iou ** alpha) * (cls_prob ** beta)
        scores   = scores * in_box.float()

        topk_scores, topk_idx = scores.topk(min(topk, N), dim=1, largest=True)
        topk_mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
        topk_mask.scatter_(1, topk_idx, True)
        topk_mask = topk_mask & in_box

        if topk_mask.sum() > 0:
            obj_iou  = iou * topk_mask.float()
            best_obj = obj_iou.argmax(dim=0)
            any_positive = topk_mask.any(dim=0)

            positive_mask[b] = any_positive
            matched_boxes[b] = gt_xyxy[best_obj]
            matched_scores[b] = iou[best_obj,
                                    torch.arange(N, device=device)] * any_positive

    return positive_mask, matched_boxes, matched_scores


def compute_iou(boxes1, boxes2, eps=1e-7):
    """
    boxes1, boxes2: (..., 4) x1y1x2y2
    """
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    return inter / (area1 + area2 - inter + eps)