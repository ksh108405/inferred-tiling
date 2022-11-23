import argparse
import cv2
import os
import time
import numpy as np
import torch
import json
from PIL import Image

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes, get_ious
from utils.vis_tools import vis_detection
from config import build_dataset_config, build_model_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO Action Tubing')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--save_vid', action='store_true', default=False,
                        help='save visualization results.')
    parser.add_argument('--save_folder', default='tubing_results/', type=str,
                        help='Dir to save visualization results')
    parser.add_argument('--save_file_name', default='detection.mp4', type=str,
                        help='Dir to save visualization results')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tube_start_thresh', default=0.6, type=float,
                        help='threshold for starting action tube')
    parser.add_argument('--tube_end_thresh', default=0.4, type=float,
                        help='threshold for ending action tube')
    parser.add_argument('--tube_trans_thresh', default=0.1, type=float,
                        help='threshold for translation per frame in action tube')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')

    # class label config
    parser.add_argument('-d', '--dataset', default='aihub_park',
                        help='aihub_park')

    # model
    parser.add_argument('-v', '--version', default='yowo_nano', type=str, choices=['yowo', 'yowo_nano'],
                        help='build YOWO')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    return parser.parse_args()


@torch.no_grad()
def detect(args, d_cfg, model, device, transform, class_names, class_colors):
    # path to save
    save_path = os.path.join(args.save_folder)
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_skip_frame = int(video_fps / 3) if int(video_fps / 3) > 0 else 1
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_size = (video_w, video_h)
        save_name = os.path.join(save_path, args.save_file_name)
        out = cv2.VideoWriter(save_name, fourcc, video_fps, save_size)

    # run
    video_clip = []
    frames = []
    results_dict = []
    image_list = []
    frame_idx = 0
    t0 = time.time()
    while True:
        frame_idx += 1
        ret, frame = video.read()

        if args.save_vid:
            if ret:
                frames.append(frame)

        if frame_idx % video_skip_frame != 0:
            continue

        if ret:
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]

            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg['len_clip']):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _, _, _, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device)  # [B, 3, T, H, W], B=1

            # inference
            outputs = model(x, None)

            batch_scores, batch_labels, batch_bboxes, _ = outputs
            # batch size = 1
            scores = batch_scores[0]
            labels = batch_labels[0]
            bboxes = batch_bboxes[0]
            # rescale
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h])
            results_dict.append({'frame_idx': frame_idx, 'scores': scores, 'labels': labels, 'bboxes': bboxes})

            progress = float(frame_idx) / float(video_length)
            print(
                f"{progress * 100.0:.2f}%\t{(time.time() - t0) / float(frame_idx) * float(video_length - frame_idx):.2f}s")

        else:
            break

    video.release()

    """
    Input:
        results_dict: List(Dict) -> [{'frame_idx': frame_idx,
                                      'scores': [scores_list],
                                      'labels': [labels_list],
                                      'bboxes': [bboxes_list]},
                                       ... ,
                                     {'frame_idx': frame_idx,
                                      'scores': [scores_list],
                                      'labels': [labels_list],
                                      'bboxes': [bboxes_list]}]
        action_tubes: List(Dict) -> [{'start_frame': start_frame_idx,
                                      'end_frame': end_frame_idx,
                                      'label': label,
                                      'bboxes': [bboxes_list]},
                                     ... ,
                                     {'start_frame': start_frame_idx,
                                      'end_frame': end_frame_idx,
                                      'label': label,
                                      'bboxes': [bboxes_list]}]
    """

    action_tubes = []
    progressing_tube_idx_list = []
    for cur_frame_idx, cur_frame_results in enumerate(results_dict):
        tubed_bbox_idx = []  # saves used bbox index in 'result'
        finishing_tube_idx_list = []
        # maintain or finish existing action tubes
        for prog_tube_list_idx, prog_tube_idx in enumerate(progressing_tube_idx_list):
            tube = action_tubes[prog_tube_idx]
            # check if the tube is maintainable
            # 1. label match
            filter_1 = []
            for bbox_idx, label in enumerate(cur_frame_results['labels']):
                if bbox_idx in tubed_bbox_idx:  # already used
                    continue
                if tube['label'] == label:
                    filter_1.append(bbox_idx)

            if len(filter_1) > 0:
                # 2. bbox center point distance match
                cx_tube = (tube['bboxes'][-1][0] + tube['bboxes'][-1][2]) / 2
                cy_tube = (tube['bboxes'][-1][1] + tube['bboxes'][-1][3]) / 2

                # 2-1. filter by translation threshold
                filter_2_1 = []
                filter_2_1_dist = []
                for bbox_idx in filter_1:
                    bbox = cur_frame_results['bboxes'][bbox_idx]
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    if abs(cx - cx_tube) < (args.tube_trans_thresh * video_w) and abs(cy - cy_tube) < (
                            args.tube_trans_thresh * video_h):
                        filter_2_1.append(bbox_idx)
                        filter_2_1_dist.append((cx - cx_tube) ** 2 + (cy - cy_tube) ** 2)

                if len(filter_2_1) > 0:
                    # 2-2. choose minimal distance bbox
                    filter_2_1_dist = np.array(filter_2_1_dist)
                    matched_bbox = filter_2_1[np.argmin(filter_2_1_dist)]

                    # 3. confidence check
                    if cur_frame_results['scores'][matched_bbox] > args.tube_end_thresh:
                        # maintain
                        action_tubes[prog_tube_idx]['bboxes'].append(cur_frame_results['bboxes'][matched_bbox].tolist())
                        tubed_bbox_idx.append(matched_bbox)
                        continue
            # tube not maintainable. finishing tube.
            action_tubes[prog_tube_idx]['end_frame'] = int(results_dict[cur_frame_idx - 1]['frame_idx'])
            finishing_tube_idx_list.append(prog_tube_list_idx)

        # remove finished tubes
        for fin_tube_list_idx in reversed(finishing_tube_idx_list):
            progressing_tube_idx_list.pop(fin_tube_list_idx)

        # discover new action tubes
        for bbox_idx, score in enumerate(cur_frame_results['scores']):
            # ignore used bbox
            if bbox_idx in tubed_bbox_idx:
                continue
            # check if the bbox is start-able
            if score > args.tube_start_thresh:
                # start new tube
                action_tubes.append({'start_frame': int(cur_frame_results['frame_idx']),
                                     'end_frame': 0,
                                     'label': int(cur_frame_results['labels'][bbox_idx]),
                                     'bboxes': [cur_frame_results['bboxes'][bbox_idx].tolist()]})
                progressing_tube_idx_list.append(len(action_tubes) - 1)
                break
    for prog_tube_idx in progressing_tube_idx_list:
        action_tubes[prog_tube_idx]['end_frame'] = int(video_length - 1)

    print(json.dumps(action_tubes))

    if args.save_vid:
        tube_bboxes = []
        for i in range(len(results_dict) + 1):
            tube_bboxes.append([])
        for tube in action_tubes:
            for frame_idx in range(tube['start_frame'], tube['end_frame'] + 1):
                tube_bboxes[frame_idx].append(tube['bboxes'][frame_idx - tube['start_frame']])
        tube_labels = []
        for i in range(len(results_dict) + 1):
            tube_labels.append([])
        for tube in action_tubes:
            for frame_idx in range(tube['start_frame'], tube['end_frame'] + 1):
                tube_labels[frame_idx].append(tube['label'])

        old_bboxes = []
        old_labels = []
        for frame_idx, frame in enumerate(frames):
            bboxes = tube_bboxes[video_skip_frame * frame_idx + 1]
            labels = tube_labels[video_skip_frame * frame_idx + 1]
            if (frame_idx + 1) % video_skip_frame != 0:
                bboxes = old_bboxes
                labels = old_labels
                # duplicate previous bboxes
            # one hot
            frame = vis_detection(
                frame=frame,
                scores=None,
                labels=labels,
                bboxes=bboxes,
                vis_thresh=None,
                class_names=class_names,
                class_colors=class_colors
            )
            old_bboxes = bboxes
            old_labels = labels
            out.write(frame)
        out.release()
        print('Saved video to {}'.format(save_name))


if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
    )

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    detect(args=args,
           d_cfg=d_cfg,
           model=model,
           device=device,
           transform=basetransform,
           class_names=class_names,
           class_colors=class_colors)
