import torch
from torch._C import dtype
import torch.nn as nn
from bbox_transform import bbox_transform_inv, clip_boxes
from generate_anchors import generate_anchors
from torchvision.ops.boxes import nms
import model.utils.config as cfg


class _ProposalLayer(nn.Module):
    def __init__(self, feat_stride, scales, ratios) -> None:
        super(_ProposalLayer, self).__init__()

        self.feature_stride_ = feat_stride
        self.anchors_ = torch.from_numpy(generate_anchors(ratios=ratios, scales=scales))
        self.num_anchors_ = self.anchors_.shape[0]

    def forward(self, input):
        score = input[0][:, self.num_anchors_:, :, :]   # [batch, anchor, h, w]
        bbox_delta = input[1]
        img_info = input[2]
        cfg_key = input[3] #TRAIN or TEST

        batch_size = score.shape[0]

        score = score.permute(0,2,3,1) # [batch, h, w, anchor]
        bbox_delta = bbox_delta.permute(0,2,3,1) # [batch, h, w, anchor * 4]

        feat_h = bbox_delta.shape[1]
        feat_w = bbox_delta.shape[2]

        h = (torch.arange(1,feat_h+1) + torch.arange(0,feat_h)) / 2
        w = (torch.arange(1,feat_w+1) + torch.arange(0,feat_w)) / 2
        
        x, y = torch.meshgrid(h, w)
        anchor_center = torch.concat([y.unsqueeze(-1), x.unsqueeze(-1)], dim=-1).repeat(1,1,2 * self.num_anchors_)
        anchor_center = torch.empty_like(bbox_delta).copy_(anchor_center * self.feature_stride_)
  

        score = score.reshape((score.shape[0], -1, 1))
        bbox_delta = bbox_delta.reshape((bbox_delta.shape[0], -1, 4))

        anchor_center = anchor_center.reshape((anchor_center.shape[0], -1, 4))
        anchors = torch.empty_like(bbox_delta).copy_(self.anchors_.unsqueeze(0).repeat(2, feat_h * feat_w,1))
        anchors = anchors + anchor_center
        proposals = clip_boxes(bbox_transform_inv(anchors, bbox_delta, batch_size), img_info, batch_size)
        # print(proposals.shape)

        outputs = proposals.new_zeros((batch_size, cfg.TRAIN.RPN_POST_NMS_TOP_N, 5))
        for i in range(batch_size):
            proposal_single = proposals[i]
            score_single = score[i]
            keep_idx = nms(proposal_single, score[i].squeeze(-1), cfg.TRAIN.RPN_POST_NMS_TOP_N)

            if cfg.TRAIN.RPN_POST_NMS_TOP_N > 0 and len(keep_idx) > cfg.TRAIN.RPN_POST_NMS_TOP_N:
                keep_idx = keep_idx[:cfg.TRAIN.RPN_POST_NMS_TOP_N]

            outputs[i,:,0] = torch.index_select(score_single, 0, keep_idx).squeeze(-1)
            outputs[i,:,1:] = torch.index_select(proposal_single, 0, keep_idx)
        
        return outputs

    def backward(self):
        pass

    def reshape(self):
        pass


if __name__ == "__main__":

    import numpy as np
    score = torch.tensor([np.array([np.array([[11,12,13,14,15,16],
                            [21,22,23,24,25,26],
                            [31,32,33,34,35,36],
                            [41,42,43,44,45,46],
                            [51,52,53,54,55,56]])+ np.array([100 * j]) for j in range(1,19)])+ 10000 * i for i in range(1,3)],
                             dtype=torch.float64) * 0.00001
    bb = torch.tensor([np.array([np.array([[11,12,13,14,15,16],
                            [21,22,23,24,25,26],
                            [31,32,33,34,35,36],
                            [41,42,43,44,45,46],
                            [51,52,53,54,55,56]])+ np.array([100 * j]) for j in range(1,37)])+ 10000*i for i in range(1,3)],
                            dtype=torch.float64) * 0.0001

    model = _ProposalLayer(16, 2**np.arange(3, 6),[0.5, 1, 2])
    model.forward([score.cuda(), bb.cuda(), torch.tensor([[600.0000, 800.0000, 1.6000], [600.0000, 800.0000, 1.6000]]), 'TRAIN'])
