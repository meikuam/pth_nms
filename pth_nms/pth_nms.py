import torch
import numpy as np
from pth_nms.nms_cpu import nms as nms_cpu
if torch.cuda.is_available():
    from pth_nms.nms_gpu import nms as nms_gpu

def pth_nms(dets, thresh):
    """
    dets has to be a tensor
    """
    if not dets.is_cuda:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms_cpu(keep, num_out, dets, order, areas, thresh)

        return keep[:num_out[0]]
    else:
        # x1 = dets[:, 0]
        # y1 = dets[:, 1]
        # x2 = dets[:, 2]
        # y2 = dets[:, 3]
        scores = dets[:, 4]

        # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]

        dets = dets[order].contiguous()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)

        nms_gpu(keep, num_out, dets, thresh)

        return order[keep[:num_out[0]].to(dets.device)].contiguous()
