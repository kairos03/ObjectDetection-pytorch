import torch


def toPoint(coords):
    """
    Convert center coordinate(cx, cy, w, h) to point coordinate (TopLeft, TopRight, BottomLeft, BottomRight).
    Args:
        coords: (tensor) coordinate of center from 
    Return:
        (tensor) Converted coordinate
    """
    return torch.cat((coords[:, :2] - coords[:, 2:]/2,     # Top
                      coords[:, :2] + coords[:, 2:]/2), 1)  # Bottom


def toCenter(coords):
    """
    Convert point coordinate(TopLeft, TopRight, BottomLeft, BottomRight) to center coordinate(cx, cy, w, h).
    Args:
        coords: (tensor) coordinate of point from 
    Return:
        (tensor) Converted coordinate
    """
    return torch.cat(((coords[:, 2:] + coords[:, :2])/2,  # cx, cy
                     coords[:, 2:] - coords[:, :2]), 1)  # w, h


def intersection(box_a, box_b):
    """
    Calculate intersection of two set of boxes.
    The inputs are collection of coordinate (TopLeft, TopRight, BottomLeft, BottomRight)
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4]
        box_b: (tensor) bounding boxes, Shape: [B, 4]
    Return:
        (tensor) intersection area, Shape: [A, B]
    """
    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    Calculate jaccard overlap of two set of boxes.
    Calc:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4]
        box_b: (tensor) bounding boxes, Shape: [B, 4]
    Return:
        (tensor) jaccard overlap, Shape: [A, B]
    """
    inter = intersection(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(default_boxes, ground_truths, threshold=0.5):
    """
    """
