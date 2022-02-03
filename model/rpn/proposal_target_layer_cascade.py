import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__))))))
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

FG_THRESHOLD = 0.5
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.1

class _ProposalTargetLayer(nn.Module):
    def __init__(self, nclasses) -> None:
        super().__init__()
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(1)
        self.num_classes = nclasses

    def forward(self, rois, gt_boxes):
        pass

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        batch_size = all_rois.shape[0]
        
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_iou, matched_gt_idx = torch.max(overlaps, 2)

        labels = gt_boxes[:, :, 4].view(-1).contiguous()

        label_batch = labels.new_zeros((batch_size, rois_per_image))
        rois_batch = all_rois.new_zeros((batch_size, rois_per_image, 5))
        gt_rois_batch = all_rois.new_zeros((batch_size, rois_per_image, 5))

        for b in range(batch_size):
            fg_inds = torch.nonzero(max_iou[b] > FG_THRESHOLD).sqeeze(-1)
            fg_num_rois = fg_inds.numel()

            bg_inds = torch.nonzero((max_iou[b] < BG_THRESH_HI) & (max_iou[b] >= BG_THRESH_LO)).squeeze(-1)
            bg_num_rois = bg_inds.numel()
            
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_limit = min(fg_num_rois, fg_rois_per_image)
                rand_idx = torch.randint(0, fg_num_rois, (fg_rois_limit,))
                fg_inds = fg_inds[rand_idx]

                bg_rois_limit = min(bg_num_rois, rois_per_image - fg_rois_limit)
                rand_idx = torch.randint(0, bg_num_rois, (bg_rois_limit,))
                bg_inds = bg_inds[rand_idx]
            elif fg_num_rois > 0:
                fg_rois_limit = min(fg_num_rois, rois_per_image)
                rand_idx = torch.randint(0, fg_num_rois, (fg_rois_limit,))
                fg_inds = fg_inds[rand_idx]
            elif bg_num_rois > 0:
                bg_rois_limit = min(bg_num_rois, rois_per_image)
                rand_idx = torch.randint(0, bg_num_rois, (bg_rois_limit,))
                bg_inds = bg_inds[rand_idx]
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            keep_inds = torch.concat([fg_inds, bg_inds])

            label_batch[b] = labels[b][keep_inds]
            if bg_inds.numel() > 0:
                label_batch[b, fg_inds.numel():] = 0

            rois_batch[b] = all_rois[b][keep_inds]
            rois_batch[b,:,0] = b

            gt_rois_batch[b] = gt_boxes[b][matched_gt_idx[b][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(rois_batch[:,:,1:], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weight = self._get_bbox_regression_labels_pytorch(
                                                        bbox_target_data, label_batch)

        return label_batch, rois_batch, bbox_targets, bbox_inside_weight

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        targets = bbox_transform_batch(ex_rois, gt_rois)

        return targets

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, label_batch):
        batch_size = bbox_target_data.shape[0]
        roi_per_image = label_batch.shape[1]

        bbox_targets = bbox_target_data.new_zeros((batch_size, roi_per_image, 4))
        bbox_inside_weight = bbox_target_data.new_zeros((batch_size, roi_per_image))

        for b in range(batch_size):
            inds = torch.nonzero(label_batch[b] > 0).squeeze(1)
            bbox_targets[b, inds, :] = bbox_target_data[b, inds, :]
            bbox_inside_weight[b, inds] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weight

    def backward(self):
        pass

if __name__ == "__main__":
    # m = _ProposalTargetLayer(5)
    # m._get_bbox_regression_labels_pytorch(torch.ones((2,10,4), device='cuda'), torch.ones((2,3), device='cuda'))

    x = torch.tensor([1,2,3,4,5,6,7,8,9,10,11])
    inds = torch.randint(10, (6,))

    print(inds)
    print(x[inds])