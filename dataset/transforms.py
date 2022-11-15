import random
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2
from PIL import Image


# Augmentation for Training
class Augmentation(object):
    def __init__(self, img_size=224, pixel_mean=[0., 0., 0.], pixel_std=[1., 1., 1.], jitter=0.2, hue=0.1,
                 saturation=1.5, exposure=1.5, img_processing='PIL', inferred_tiling=False, it_jitter=0.1,
                 it_drop=0.1, it_wrong=0.3, it_wrong_surplus=0.5):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.img_processing = img_processing
        self.inferred_tiling = inferred_tiling
        self.it_jitter = it_jitter
        self.it_drop = it_drop
        self.it_wrong = it_wrong
        self.it_wrong_surplus = it_wrong_surplus

    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1):
            return scale

        return 1. / scale

    def random_distort_image(self, video_clip, inferred_tiles):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)

        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())
            cs[1] = cs[1].point(lambda i: i * dsat)
            cs[2] = cs[2].point(lambda i: i * dexp)

            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            cs[0] = cs[0].point(change_hue)
            image = Image.merge(image.mode, tuple(cs))

            image = image.convert('RGB')

            video_clip_.append(image)

        if self.inferred_tiling:
            inferred_tiles_ = []
            for tile in inferred_tiles:
                tile = tile.convert('HSV')
                cs = list(tile.split())
                cs[1] = cs[1].point(lambda i: i * dsat)
                cs[2] = cs[2].point(lambda i: i * dexp)

                def change_hue(x):
                    x += dhue * 255
                    if x > 255:
                        x -= 255
                    if x < 0:
                        x += 255
                    return x

                cs[0] = cs[0].point(change_hue)
                tile = Image.merge(tile.mode, tuple(cs))

                tile = tile.convert('RGB')

                inferred_tiles_.append(tile)
        else:
            inferred_tiles_ = None

        return video_clip_, inferred_tiles_


    def random_tile_crop(self, img, height, width):
        cx = np.random.randint(int(width * 0.05), int(width * 0.95))
        cy = np.random.randint(int(height * 0.05), int(height * 0.95))
        w = np.random.randint(int(width * 0.05), int(width * 0.2))
        h = np.random.randint(int(height * 0.05), int(height * 0.2))
        t_left = cx - (w // 2)
        t_right = cx + (w // 2)
        t_top = cy - (h // 2)
        t_bot = cy + (h // 2)
        return img.crop((t_left, t_top, t_right, t_bot))


    def random_crop(self, target, video_clip, width, height):
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)

        # Adjust to not crop any subset of key-frame bbox
        bbox_min_left = np.min(target[..., 0])
        bbox_min_top = np.min(target[..., 1])
        bbox_max_right = np.max(target[..., 2])
        bbox_max_bottom = np.max(target[..., 3])

        if self.img_processing == 'PIL':
            pleft = random.randint(-dw, int(np.minimum(dw, bbox_min_left)))
            pright = random.randint(-dw, int(np.minimum(dw, width - bbox_max_right)))
            ptop = random.randint(-dh, int(np.minimum(dh, bbox_min_top)))
            pbot = random.randint(-dh, int(np.minimum(dh, height - bbox_max_bottom)))
        elif self.img_processing == 'pyvips':
            pleft = random.randint(0, int(np.minimum(dw, bbox_min_left)))
            pright = random.randint(0, int(np.minimum(dw, width - bbox_max_right)))
            ptop = random.randint(0, int(np.minimum(dh, bbox_min_top)))
            pbot = random.randint(0, int(np.minimum(dh, height - bbox_max_bottom)))

        swidth = width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth) / width
        sy = float(sheight) / height

        dx = (float(pleft) / width) / sx
        dy = (float(ptop) / height) / sy

        # random crop
        if self.img_processing == 'PIL':
            cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]
        elif self.img_processing == 'pyvips':
            cropped_clip = [img.crop(pleft, ptop, swidth, sheight) for img in video_clip]

        if self.inferred_tiling:
            """
            image = np.array(video_clip[-1])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            """
            cropped_tiles = []
            random_det = np.random.rand()
            for box in target:
                if random_det < self.it_drop:  # drop
                    continue
                elif random_det < self.it_drop + self.it_wrong:  # wrong crop
                    cropped_tiles.append(self.random_tile_crop(video_clip[-1], height, width))
                    random_det_surplus = np.random.rand()
                    if random_det_surplus < 0.5:
                        cropped_tiles.append(self.random_tile_crop(video_clip[-1], height, width))
                else:  # normal crop
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1
                    h = y2 - y1
                    dw = int(w * self.it_jitter)
                    dh = int(h * self.it_jitter)
                    tl = x1 - dw
                    tr = x2 + dw
                    tt = y1 - dh
                    tb = y2 + dh
                    t_left  = random.randint(-dw + tl, dw + tl)
                    t_right = random.randint(-dw + tr, dw + tr)
                    t_top   = random.randint(-dh + tt, dh + tt)
                    t_bot   = random.randint(-dh + tb, dh + tb)
                    cropped_tiles.append(video_clip[-1].crop((t_left, t_top, t_right, t_bot)))
                    """
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    image = cv2.rectangle(image, (dw + tl, dh + tt), (tr - dw, tb - dh), (0, 0, 255), 2)  # min
                    image = cv2.rectangle(image, (tl - dw, tt - dh), (dw + tr, dh + tb), (0, 0, 255), 2)  # max
            image = cv2.resize(image, (1280, 720))
            cv2.imshow('image', image)
            cv2.waitKey(0)
            """
            if cropped_tiles == []:
                cropped_tiles = None
        else:
            cropped_tiles = None

        return cropped_clip, cropped_tiles, dx, dy, sx, sy

    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy, inferred_tile=False):
        sx, sy = 1. / sx, 1. / sy
        # apply deltas on bbox
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx))
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy))
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx))
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy))

        # refine target
        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow
            bh = (tgt[3] - tgt[1]) * oh

            if not inferred_tile:
                if bw < 1. or bh < 1.:
                    continue

            refine_target.append(tgt)

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target

    def to_tensor(self, video_clip):
        return [F.normalize(F.to_tensor(image), self.pixel_mean, self.pixel_std) for image in video_clip]

    def __call__(self, video_clip, target, target_nf=None):
        # Initialize Random Variables
        oh = video_clip[0].height
        ow = video_clip[0].width

        # random crop
        if self.inferred_tiling:
            video_clip, inferred_tiles, dx, dy, sx, sy = self.random_crop(target, video_clip, ow, oh)
        else:
            video_clip, _, dx, dy, sx, sy = self.random_crop(target, video_clip, ow, oh)

        # resize
        if self.img_processing == 'PIL':
            video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]
        elif self.img_processing == 'pyvips':
            video_clip = [img.thumbnail_image(self.img_size, height=self.img_size, size='force') for img in video_clip]
            video_clip = [Image.fromarray(image.numpy()).convert('RGB') for image in video_clip]
        if self.inferred_tiling:
            if inferred_tiles is not None:
                inferred_tiles = [tile.resize([self.img_size, self.img_size]) for tile in inferred_tiles]

        # random flip
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video_clip]
            if self.inferred_tiling:
                if inferred_tiles is not None:
                    inferred_tiles = [tile.transpose(Image.FLIP_LEFT_RIGHT) for tile in inferred_tiles]

        # distort
        if self.inferred_tiling:
            if inferred_tiles is not None:
                video_clip, inferred_tiles = self.random_distort_image(video_clip, inferred_tiles)
        else:
            video_clip, _ = self.random_distort_image(video_clip, None)

        # process target
        if target is not None:
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
            if target_nf is not None:
                target_nf = self.apply_bbox(target_nf, ow, oh, dx, dy, sx, sy, True)
                if flip:
                    target_nf[..., [0, 2]] = 1.0 - target_nf[..., [2, 0]]
            else:
                target_nf = np.array([])
        else:
            target = np.array([])

        # to tensor
        video_clip = self.to_tensor(video_clip)
        if self.inferred_tiling:
            if inferred_tiles is not None:
                inferred_tiles = self.to_tensor(inferred_tiles)
        else:
            inferred_tiles = None
        target = torch.as_tensor(target).float()
        target_nf = torch.as_tensor(target_nf).float()

        return video_clip, target, target_nf, inferred_tiles

    # Transform for Testing


class BaseTransform(object):
    def __init__(self, img_size=224, pixel_mean=[0., 0., 0.], pixel_std=[1., 1., 1.], img_processing='PIL'):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_processing = img_processing

    def to_tensor(self, video_clip):
        return [F.normalize(F.to_tensor(image), self.pixel_mean, self.pixel_std) for image in video_clip]

    def __call__(self, video_clip, target=None, normalize=True, target_nf=None):
        oh = video_clip[0].height
        ow = video_clip[0].width

        # resize
        if self.img_processing == 'PIL':
            video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]
        elif self.img_processing == 'pyvips':
            video_clip = [img.thumbnail_image(self.img_size, self.img_size) for img in video_clip]
            video_clip = [Image.fromarray(image.numpy()) for image in video_clip]

        # normalize target
        if target is not None:
            if normalize:
                target[..., [0, 2]] /= ow
                target[..., [1, 3]] /= oh
        else:
            target = np.array([])

        if target_nf is not None:
            if normalize:
                target_nf[..., [0, 2]] /= ow
                target_nf[..., [1, 3]] /= oh
        else:
            target_nf = np.array([])

        # to tensor
        video_clip = self.to_tensor(video_clip)
        target = torch.as_tensor(target).float()
        target_nf = torch.as_tensor(target_nf).float()

        return video_clip, target, target_nf, None
