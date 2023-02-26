#!/usr/bin/python
# encoding: utf-8

from locale import normalize
import os
import random
import numpy as np
import glob
import time

import torch
from torch.utils.data import Dataset
from PIL import Image
import pyvips
from utils.frame_matcher import frame_match_bboxes


# Dataset for UCF24 & JHMDB
class UCF_JHMDB_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 is_train=False,
                 len_clip=16,
                 sampling_rate=1,
                 img_processing='PIL',
                 inferred_tiling=False):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train

        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.img_processing = img_processing
        self.inferred_tiling = inferred_tiling
        if self.inferred_tiling and self.img_processing == 'pyvips':
            raise Exception('Inferred Tiling implementation only supports PIL, now.')

        if self.is_train:
            self.split_list = 'trainlist.txt'
        else:
            self.split_list = 'testlist.txt'

        # load data
        if self.inferred_tiling and self.is_train:
            self.file_names = []
            with open(os.path.join(data_root, self.split_list), 'r') as file:
                lines = file.readlines()
                for idx in range(len(lines) - 1):
                    video_name = lines[idx].split('/')[-2]
                    next_video_name = lines[idx + 1].split('/')[-2]
                    frame_id = int(lines[idx].split('/')[-1][:5])
                    next_frame_id = int(lines[idx + 1].split('/')[-1][:5])
                    if video_name == next_video_name and next_frame_id - frame_id == 1:
                        self.file_names.append(lines[idx])
        else:
            with open(os.path.join(data_root, self.split_list), 'r') as file:
                self.file_names = file.readlines()
        self.num_samples = len(self.file_names)

        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21
        elif dataset == 'aihub_park':
            self.num_classes = 4
        elif dataset == 'aihub_park_subset':
            self.num_classes = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # load a data
        frame_idx, video_clip, inferred_tiles, target = self.pull_item(index)

        return frame_idx, video_clip, inferred_tiles, target

    def pull_item(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # image name
        img_id = int(img_split[-1][:5])

        # path to label
        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))
        if self.inferred_tiling and self.is_train:
            label_nf_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id + 1))
        else:
            label_nf_path = None

        # image folder
        img_folder = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2])

        # frame numbers
        if self.dataset == 'ucf24':
            max_num = len(os.listdir(img_folder))
        elif self.dataset == 'jhmdb21':
            max_num = len(os.listdir(img_folder)) - 1
        elif self.dataset in ['aihub_park', 'aihub_park_subset']:
            max_num = len(os.listdir(img_folder))

        # sampling rate
        if self.is_train and not self.inferred_tiling:
            d = random.randint(1, 2)
        else:
            d = self.sampling_rate

        # load images
        video_clip = []
        t0 = time.time()
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i * d
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            if self.dataset == 'ucf24':
                path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2],
                                        '{:05d}.jpg'.format(img_id_temp))
            elif self.dataset == 'jhmdb21':
                path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2],
                                        '{:05d}.png'.format(img_id_temp))
            elif self.dataset in ['aihub_park', 'aihub_park_subset']:
                path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2],
                                        '{:05d}.jpg'.format(img_id_temp))

            if self.img_processing == 'PIL':
                frame = Image.open(path_tmp).convert('RGB')  # PIL
            elif self.img_processing == 'pyvips':
                frame = pyvips.Image.new_from_file(path_tmp, access='sequential')  # pyvips
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

            frame_id = img_split[1] + '_' + img_split[2] + '_' + img_split[3]
        # print(f'{time.time() - t0:0.3f}', end='\t')

        # load an annotation
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None

        # load next frame annotation
        target_nf = None
        if label_nf_path is not None:
            if os.path.getsize(label_nf_path):
                target_nf = np.loadtxt(label_nf_path)

        # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]
        label = target[..., :1]
        boxes = target[..., 1:]
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)
        if target_nf is not None:
            label_nf = target_nf[..., :1]
            boxes_nf = target_nf[..., 1:]
            target_nf = np.concatenate([label_nf, boxes_nf], axis=-1).reshape(-1, 5)

            nf_bbox_order_list = \
                frame_match_bboxes(target[..., :4], target[..., 4:], target_nf[..., 1:], target_nf[..., :1])

            label_nf = target_nf[..., :1]
            boxes_nf = target_nf[..., 1:]
            boxes_nf_ = []
            for box_nf in boxes_nf:
                x1 = box_nf[0]
                y1 = box_nf[1]
                x2 = box_nf[2]
                y2 = box_nf[3]
                w = x2 - x1
                h = y2 - y1
                x1 = (x1 - w * 0.1)
                y1 = (y1 - h * 0.1)
                x2 = (x2 + w * 0.1)
                y2 = (y2 + h * 0.1)
                boxes_nf_.append([x1, y1, x2, y2])
            boxes_nf = np.array(boxes_nf_)
            target_nf = np.concatenate([label_nf, boxes_nf], axis=-1).reshape(-1, 5)

            label_nf = target_nf[..., :1]
            boxes_nf = target_nf[..., 1:]
            boxes_nf_ = []
            label_nf_ = []
            for idx in range(label.size):
                if idx in nf_bbox_order_list:
                    boxes_nf_.append(boxes_nf[np.where(nf_bbox_order_list == idx)[0]])
                    label_nf_.append(label_nf[np.where(nf_bbox_order_list == idx)[0]])
                else:
                    boxes_nf_.append(np.zeros((1, 4)))
                    label_nf_.append(np.ones((1, 1)) * -1)
            boxes_nf = np.concatenate(boxes_nf_, axis=0)
            label_nf = np.concatenate(label_nf_, axis=0)
            target_nf = np.concatenate([boxes_nf, label_nf], axis=-1).reshape(-1, 5)
            assert target_nf.shape == target.shape, \
                f"Frame matching error.\nprev_frame: {target}\ncur_frame: {target_nf}"

        # transform
        video_clip, target, target_nf, object_tiles, ot_bboxes = self.transform(video_clip, target, target_nf=target_nf)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)
        if self.inferred_tiling and self.is_train and object_tiles is not None:
            object_tiles = torch.stack(object_tiles, dim=0)
        else:
            object_tiles = None

        # reformat target
        if self.inferred_tiling and self.is_train:
            target = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'boxes_it': target_nf[:, :4].float(),  # [N, 4]
                'boxes_ot': ot_bboxes.float(),  # [N, 4]
                'labels': target[:, -1].long() - 1,  # [N,]
                'orig_size': [ow, oh]
            }
        else:
            target = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, -1].long() - 1,  # [N,]
                'orig_size': [ow, oh]
            }

        return frame_id, video_clip, target, object_tiles

    def pull_anno(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # image name
        img_id = int(img_split[-1][:5])

        # path to label
        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))

        # load an annotation
        target = np.loadtxt(label_path)
        target = target.reshape(-1, 5)

        return target


# Video Dataset for UCF24 & JHMDB
class UCF_JHMDB_VIDEO_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 sampling_rate=1):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform

        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate

        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21
        elif dataset == 'aihub_park':
            self.num_classes = 4
        elif dataset == 'aihub_park_subset':
            self.num_classes = 1

    def set_video_data(self, line):
        self.line = line

        # load a video
        self.img_folder = os.path.join(self.data_root, 'rgb-images', self.line)

        if self.dataset == 'ucf24':
            self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))
        elif self.dataset == 'jhmdb21':
            self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.png')))
        elif self.dataset in ['aihub_park', 'aihub_park_subset']:
            self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        image_path = self.label_paths[index]

        video_split = self.line.split('/')
        video_class = video_split[0]
        video_file = video_split[1]
        # for windows:
        # img_split = image_path.split('\\')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # for linux
        img_split = image_path.split('/')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        # image name
        img_id = int(img_split[-1][:5])
        max_num = len(os.listdir(self.img_folder))
        if self.dataset == 'ucf24':
            img_name = os.path.join(video_class, video_file, '{:05d}.jpg'.format(img_id))
        elif self.dataset == 'jhmdb21':
            img_name = os.path.join(video_class, video_file, '{:05d}.png'.format(img_id))
        elif self.dataset in ['aihub_park', 'aihub_park_subset']:
            img_name = os.path.join(video_class, video_file, '{:05d}.jpg'.format(img_id))

        # load video clip
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            if self.dataset == 'ucf24':
                path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file,
                                        '{:05d}.jpg'.format(img_id_temp))
            elif self.dataset == 'jhmdb21':
                path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file,
                                        '{:05d}.png'.format(img_id_temp))
            elif self.dataset in ['aihub_park', 'aihub_park_subset']:
                path_tmp = os.path.join(self.data_root, 'rgb-images', video_class, video_file,
                                        '{:05d}.jpg'.format(img_id_temp))
            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

        # transform
        video_clip, _ = self.transform(video_clip, normalize=False)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)
        orig_size = [ow, oh]  # width, height

        target = {'orig_size': [ow, oh]}

        return img_name, video_clip, target


if __name__ == '__main__':
    import cv2
    from transforms import Augmentation, BaseTransform

    data_root = 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24'
    dataset = 'ucf24'
    is_train = True
    img_size = 224
    len_clip = 16
    trans_config = {
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    transform = Augmentation(
        img_size=img_size,
        pixel_mean=trans_config['pixel_mean'],
        pixel_std=trans_config['pixel_std'],
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure']
    )
    # transform = BaseTransform(
    #     img_size=img_size,
    #     pixel_mean=trans_config['pixel_mean'],
    #     pixel_std=trans_config['pixel_std']
    #     )

    train_dataset = UCF_JHMDB_Dataset(
        data_root=data_root,
        dataset=dataset,
        img_size=img_size,
        transform=transform,
        is_train=is_train,
        len_clip=len_clip,
        sampling_rate=1
    )

    print(len(train_dataset))
    std = trans_config['pixel_std']
    mean = trans_config['pixel_mean']
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = ((key_frame * std + mean) * 255).astype(np.uint8)
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # # PIL show
        # image = Image.fromarray(image.astype(np.uint8))
        # image.show()

        # cv2 show
        cv2.imshow('key frame', key_frame[..., (2, 1, 0)])
        cv2.waitKey(0)
