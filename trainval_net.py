import torch
import torch.utils.data
import os

from dataset.data_sampler import sampler
from dataset.roi_data_layer.roidb import combined_roidb
from dataset.roi_data_layer.roibatchLoader import roibatchLoader

from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.trainval_args import parse_args

## args or cfg?
CUDA = True
START_EPOCH = 0
MAX_EPOCH = 10

def train(args):
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_2012":
        args.imdb_name = "voc_2012_trainval"
        args.imdbval_name = "voc_2012_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic) 
    model.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    
    optimizer = torch.optim.Adam(params)
    
    
    if CUDA:
        model.cuda()

    for epoch in range(START_EPOCH, MAX_EPOCH + 1):
        for i, data in enumerate(dataloader):
            im_data = data[0]
            im_info = data[1]
            gt_boxes = data[2]
            num_boxes = data[3]

            if CUDA:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                gt_boxes = gt_boxes.cuda()
                num_boxes = num_boxes.cuda()

            model.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    train(args)