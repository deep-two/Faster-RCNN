import torch
import torch.utils.data
import os

from dataset.data_sampler import sampler
from dataset.roi_data_layer.roidb import combined_roidb
from dataset.roi_data_layer.roibatchLoader import roibatchLoader

from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.trainval_args import parse_args

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

    if args.resume:
        load_name = os.path.join(output_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
    
    
    if args.cuda:
        model.cuda()

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        model.train()
        loss_temp = 0
        iters_per_epoch = len(dataloader)

        if epoch % (args.lr_decay_step) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_decay_gamma * param_group['lr']
            lr *= args.lr_decay_gamma

        for step, data in enumerate(dataloader):
            im_data = data[0]
            im_info = data[1]
            gt_boxes = data[2]
            num_boxes = data[3]

            if args.cuda:
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
            loss_temp += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (args.disp_interval * 10) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (args.lr_decay_gamma * 7) * param_group['lr']
                lr *= (args.lr_decay_gamma * 7)

            if step % args.disp_interval == 0:
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt, bg_cnt))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                loss_temp = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    torch.save({
      'session': args.session,
      'epoch': epoch + 1,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    train(args)