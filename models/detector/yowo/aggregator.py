import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def pe_sincos_2d(patch, bbox, temperature=10000):
    _, h, w, dim, device, dtype = *patch.shape, patch.device, patch.dtype

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

    y, x = torch.meshgrid(torch.linspace(y1, y2, steps=h, device=device),
                          torch.linspace(x1, x2, steps=w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=128):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x, orig_x):
        x = self.norm(x)

        qk = self.to_qk(x).chunk(2, dim=-1)  # (tensor(1, num_obj, dim_head), tensor(1, num_obj, dim_head))

        # (tensor(1, 1, num_obj, dim_head), tensor(1, 1, num_obj, dim_head))
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        v = torch.stack(orig_x, dim=1)  # tensor(1, num_obj, 425, 7, 7)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # tensor(1, 1, num_obj, num_obj)

        attn = self.attend(dots)[0, 0, 0]  # only choose attention score of first tile

        ret = (v * attn.view(1, -1, 1, 1, 1)).sum(dim=1)  # tensor(1, num_obj, 425, 7, 7) -> tensor(1, 425, 7, 7)

        return ret


class SelfAttAggregator(nn.Module):
    def __init__(self, *, img_size=7, channels=425):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            Rearrange('b c h w -> b h w c')
        )

        self.attention = Attention(32 * img_size ** 2)
        self.whole_bbox = torch.tensor([[0., 0., 1., 1.]])

    def forward(self, tile_list, bbox_list):  # [tensor(1, 425, 7, 7), ...]
        tile_list_ = [self.to_patch_embedding(tile) for tile in tile_list]  # [tensor(1, 7, 7, 64), ...]
        if bbox_list is not None:
            bbox_list = torch.cat((self.whole_bbox, bbox_list), 0)
        else:
            bbox_list = self.whole_bbox
        pe_list = [pe_sincos_2d(tile, bbox) for tile, bbox in zip(tile_list_, bbox_list)]  # [tensor(1, 49, 64), ...]
        tile_list_ = [rearrange(rearrange(tile, 'b ... d -> b (...) d') + pe, 'b ... -> b (...)')
                       for tile, pe in zip(tile_list_, pe_list)]  # [tensor(1, 3136), ...]
        tile_list_ = torch.stack(tile_list_, dim=1)  # tensor(1, 1 + num_obj, 3136)
        agg_tile = self.attention(tile_list_, tile_list)  # tensor(1, 425, 7, 7)
        return agg_tile


if __name__ == '__main__':
    v = SelfAttAggregator(
        img_size=7,
        channels=425,
    )

    tile_list = [torch.randn(1, 425, 7, 7), torch.randn(1, 425, 7, 7), torch.randn(1, 425, 7, 7)]
    bbox_list = [[0.4, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.4]]

    preds = v(tile_list, bbox_list)  # (1, 1000)