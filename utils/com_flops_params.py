import torch
from thop import profile


def FLOPs_and_Params(model, img_size, len_clip, device, inferred_tiling=False):
    # generate init video clip
    video_clip = torch.randn(1, 3, len_clip, img_size, img_size).to(device)
    if inferred_tiling:
        inferred_tiles = torch.randn(1, 2, 3, img_size, img_size).to(device)
    else:
        inferred_tiles = None

    # set eval mode
    model.trainable = False
    model.eval()

    print('==============================')
    flops, params = profile(model, inputs=(video_clip, inferred_tiles,))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))

    # set train mode.
    model.trainable = True
    model.train()


if __name__ == "__main__":
    pass
