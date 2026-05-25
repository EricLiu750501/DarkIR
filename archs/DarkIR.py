import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from arch_model import EBlock, DBlock
    from arch_util import CustomSequential
except:
    from archs.arch_model import EBlock, DBlock
    from .arch_util import CustomSequential

class TemporalAttentionFusion(nn.Module):

    def __init__(self, channels, reduction=8, init_alpha=0.1):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        pooled = self.pool(x.view(b * t, c, h, w)).view(b, t, c)
        logits = self.mlp(pooled)
        weights = torch.softmax(logits, dim=1).view(b, t, 1, 1, 1)
        fused = (x * weights).sum(dim=1)
        center = x[:, t // 2]
        return center + self.alpha * (fused - center)


class DarkIR(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations = [1, 4, 9], 
                 extra_depth_wise = True,
                 temporal_window = 1,
                 temporal_fusion = False,
                 temporal_reduction = 8):
        super(DarkIR, self).__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.temporal_window = temporal_window
        self.temporal_fusion = temporal_fusion and temporal_window > 1
        if self.temporal_fusion:
            self.temporal_fusion_layer = TemporalAttentionFusion(width, reduction=temporal_reduction)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)        
        
        # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
        self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel, 
                                kernel_size = 3, stride=1, padding=1)
        
    def forward(self, input, side_loss = False, use_adapter = None):

        if input.dim() == 5:
            b, t, c, h, w = input.shape
            center_index = t // 2
            H, W = h, w
            input_4d = input.view(b * t, c, h, w)
            input_4d = self.check_image_size(input_4d)
            _, _, pad_h, pad_w = input_4d.shape
            input_5d = input_4d.view(b, t, c, pad_h, pad_w)
            x = self.intro(input_4d).view(b, t, -1, pad_h, pad_w)
            if self.temporal_fusion:
                x = self.temporal_fusion_layer(x)
            else:
                x = x[:, center_index]
            input_base = input_5d[:, center_index]
        else:
            _, _, H, W = input.shape
            input = self.check_image_size(input)
            x = self.intro(input)
            input_base = input
        
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input_base
        out = x[:, :, :H, :W] # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x      

if __name__ == '__main__':
    
    img_channel = 3
    width = 32
    
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net = DarkIR(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise)
    
    new_state_dict = net.state_dict()

    inp_shape = (3, 256, 256)

    net.load_state_dict(new_state_dict)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    
    weights = net.state_dict()
    adapter_weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    
    
