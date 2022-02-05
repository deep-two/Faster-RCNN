import torch
import torch.nn as nn

from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch
from model.utils.config import cfg

class _ProposalTargetLayer(nn.Module):
    def __init__(self, nclasses) -> None:
        super().__init__()
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(1)
        self.num_classes = nclasses

    def forward(self, rois_in, gt_boxes):
        batch_size = gt_boxes.shape[0]

        gt_rois = gt_boxes.new_zeros((gt_boxes.shape))
        gt_rois[:,:,0] = gt_boxes[:,:,4]
        gt_rois[:,:,1:] = gt_boxes[:,:,:4]
        all_rois = torch.cat([rois_in, gt_rois], 1)

        rois_per_image = cfg.TRAIN.BATCH_SIZE
        fg_rois_per_image = int(rois_per_image * cfg.TRAIN.FG_FRACTION + 0.5)
        fg_rois_per_image = 1 if fg_rois_per_image <= 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights = \
            self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image)

        bbox_outside_weights = bbox_inside_weights.new_zeros(bbox_inside_weights.shape)
        outside_inds = torch.nonzero(bbox_inside_weights > 0)
        bbox_outside_weights[outside_inds] = 1

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
        

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        batch_size = all_rois.shape[0]
        
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_iou, matched_gt_idx = torch.max(overlaps, 2)

        labels = gt_boxes[:, matched_gt_idx, 4].view(batch_size, -1).contiguous()

        label_batch = labels.new_zeros((batch_size, rois_per_image))
        rois_batch = all_rois.new_zeros((batch_size, rois_per_image, 5))
        gt_rois_batch = all_rois.new_zeros((batch_size, rois_per_image, 5))

        for b in range(batch_size):
            fg_inds = torch.nonzero(max_iou[b] > cfg.TRAIN.FG_THRESH).squeeze(-1)
            fg_num_rois = fg_inds.numel()

            bg_inds = torch.nonzero((max_iou[b] < cfg.TRAIN.BG_THRESH_HI) & 
                                (max_iou[b] >= cfg.TRAIN.BG_THRESH_LO)).squeeze(-1)
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

            keep_inds = torch.cat([fg_inds, bg_inds])

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
        bbox_inside_weight = bbox_target_data.new_zeros(bbox_targets.shape)

        for b in range(batch_size):
            inds = torch.nonzero(label_batch[b] > 0).squeeze(1)
            bbox_targets[b, inds, :] = bbox_target_data[b, inds, :]
            bbox_inside_weight[b, inds, :] = self.BBOX_INSIDE_WEIGHTS

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