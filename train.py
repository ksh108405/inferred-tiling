import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import random
import time
import argparse
import numpy as np
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_dataset_config, build_model_config
from models.detector import build_model

GLOBAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Visualization
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='./weights/', type=str,
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # Mix precision training
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False,
                        help='do evaluation during training.')
    parser.add_argument('--eval_epoch', default=1, type=int,
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='save inference results.')

    # Model
    parser.add_argument('-v', '--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'],
                        help='build YOWO')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', dest="ema", action="store_true", default=False,
                        help="use model EMA.")

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')

    # Inferred Tiling
    parser.add_argument('-it', '--inferred_tiling', action='store_true', default=False,
                        help='use inferred tiling technic')
    parser.add_argument('--it_weight_share', action='store_true', default=False,
                        help='Share weights between whole-image net and object-tile net')
    parser.add_argument('--it_feature_agg', default='sum', type=str, choices=['sum', 'self-att'],
                        help='Share weights between whole-image net and object-tile net')
    parser.add_argument('--it_drop', default=0., type=float,
                        help='probability of dropping a object tile')
    parser.add_argument('--it_wrong', default=0., type=float,
                        help='probability of cropping an wrong object tile')
    parser.add_argument('--it_wrong_surplus', default=0., type=float,
                        help='probability of cropping an surplus wrong object tile')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False,
                        help='use sybn.')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # amp
    scaler = amp.GradScaler(enabled=args.fp16)

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True, inferred_tiling=args.inferred_tiling)

    # dataloader
    if args.inferred_tiling:
        batch_size = 1 * distributed_utils.get_world_size()
    else:
        batch_size = d_cfg['batch_size'] * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc(), is_train=True)

    # build model
    net = build_model(args=args,
                      d_cfg=d_cfg,
                      m_cfg=m_cfg,
                      device=device,
                      num_classes=num_classes,
                      trainable=True,
                      resume=args.resume,
                      inferred_tiling=args.inferred_tiling,
                      it_weight_share=args.it_weight_share,
                      it_feature_agg=args.it_feature_agg)
    model = net
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Compute FLOPs and Params
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg['test_size'],
            len_clip=d_cfg['len_clip'],
            device=device,
            inferred_tiling=args.inferred_tiling)
        del model_copy

    # optimizer
    if args.inferred_tiling:
        base_lr = d_cfg['base_lr'] / d_cfg['batch_size']
    else:
        base_lr = d_cfg['base_lr']
    optimizer, start_epoch = build_optimizer(
        model=model_without_ddp,
        base_lr=base_lr,
        name=d_cfg['optimizer'],
        momentum=d_cfg['momentum'],
        weight_decay=d_cfg['weight_decay'],
        resume=args.resume
    )

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=d_cfg['lr_epoch'],
        gamma=d_cfg['lr_decay_ratio']
    )

    # warmup scheduler
    if args.inferred_tiling:
        wp_iter = d_cfg['wp_iter'] * d_cfg['batch_size']
    else:
        wp_iter = d_cfg['wp_iter']

    warmup_scheduler = build_warmup(
        name=d_cfg['warmup'],
        base_lr=base_lr,
        wp_iter=wp_iter,
        warmup_factor=d_cfg['warmup_factor']
    )

    # training configuration
    max_epoch = d_cfg['max_epoch']
    epoch_size = len(dataloader)
    warmup = True

    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

            # train one epoch
        for iter_i, (frame_ids, video_clips, targets, object_tiles) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            """
            # visualization
            for batch_idx, video_clip in enumerate(video_clips):
                video_clip = video_clip.permute(1, 2, 3, 0)
                for image_idx, image in enumerate(video_clip):
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if image_idx == len(video_clip) - 1:
                        bboxes = targets[batch_idx]['boxes']
                        bboxes_it = targets[batch_idx]['boxes_it']
                        for bbox in bboxes:
                            bbox_left = bbox[0] * 224
                            bbox_top = bbox[1] * 224
                            bbox_right = bbox[2] * 224
                            bbox_bottom = bbox[3] * 224
                            print(f'{frame_ids[batch_idx]}: {bbox_left}, {bbox_top}, {bbox_right - bbox_left}, {bbox_bottom - bbox_top}')
                            image = cv2.rectangle(image, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)), (0, 0, 255), 2)
                        if bboxes_it is not None:
                            for bbox in bboxes_it:
                                bbox_left = bbox[0] * 224
                                bbox_top = bbox[1] * 224
                                bbox_right = bbox[2] * 224
                                bbox_bottom = bbox[3] * 224
                                print(f'{frame_ids[batch_idx]}: {bbox_left}, {bbox_top}, {bbox_right - bbox_left}, {bbox_bottom - bbox_top}')
                                image = cv2.rectangle(image, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)), (128, 128, 255), 2)
                    cv2.imshow('image', image)
                    cv2.waitKey(0)
                    # exit(0)
                if object_tiles is not None:
                    for image_idx, tile in enumerate(object_tiles[batch_idx]):
                        tile = tile.permute(1, 2, 0)
                        tile = np.array(tile)
                        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                        cv2.imshow('image', tile)
                        cv2.waitKey(0)
            # exit(0)
            """
            # warmup
            if ni < wp_iter and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == wp_iter and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            # to device
            video_clips = video_clips.to(device)
            if object_tiles != [None]:
                object_tiles = object_tiles.to(device)
            else:
                object_tiles = None

            # inference
            if args.fp16:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss_dict, _ = model(video_clips, object_tiles, targets=targets)
            else:
                loss_dict, _ = model(video_clips, object_tiles, targets=targets)

            losses = loss_dict['losses']
            losses = losses / d_cfg['accumulate']

            # reduce
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN!!')
                continue

            # Backward and Optimize
            if args.fp16:
                scaler.scale(losses).backward()

                # Optimize
                if ni % d_cfg['accumulate'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                # Backward
                losses.backward()

                # Optimize
                if ni % d_cfg['accumulate'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                # basic infor
                log = '[Epoch: {}/{}]'.format(epoch + 1, max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.8f}]'.format(cur_lr[0])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[size: {}]'.format(d_cfg['train_size'])

                # print log infor
                print(log, flush=True)

                t0 = time.time()

        lr_scheduler.step()

        # evaluation
        if epoch % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            model_eval = model_without_ddp
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')

                else:
                    print('eval ...')
                    # set eval mode
                    model_eval.trainable = False
                    model_eval.eval()

                    # evaluate
                    evaluator.evaluate_frame_map(model_eval, epoch + 1)

                    # set train mode.
                    model_eval.trainable = True
                    model_eval.train()

                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}'.format(args.version, epoch + 1)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                if os.path.isfile(checkpoint_path + '.pth'):
                    checkpoint_path += '_'
                    file_dup = 1
                    while os.path.isfile(f'{checkpoint_path}{file_dup}.pth'):
                        file_dup += 1
                    checkpoint_path += f'{file_dup}'
                torch.save({'model': model_eval.state_dict(),
                            'epoch': epoch,
                            'args': args},
                           checkpoint_path + '.pth')

            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()


if __name__ == '__main__':
    train()
