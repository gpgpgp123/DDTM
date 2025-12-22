import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False) 
        w = (w - m) / torch.sqrt(v + 1e-5) 
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

# demo1
# class RPBlock(nn.Module):
#     def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
#         super(RPBlock,self).__init__()
#         self.branches = nn.ModuleList()
#         for i, ratio in enumerate(ratios):
#             conv = nn.Sequential(
#                 nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
#                 nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
#                 nn.ReLU()
#             )
#             self.branches.append(conv)
        
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(input_chs, momentum=bn_momentum),
#             nn.ReLU()
#         )

#         self.conv_31 = nn.Sequential(
#             nn.Conv2d(input_chs, input_chs, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=input_chs),
#             # nn.BatchNorm2d(input_chs, momentum=bn_momentum),
#             # nn.ReLU()
#         )

#         self.conv_13 = nn.Sequential(
#             nn.Conv2d(input_chs, input_chs, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=input_chs),
#             # nn.BatchNorm2d(input_chs, momentum=bn_momentum),
#             # nn.ReLU()
#         )

#         self.conv_11 = nn.Sequential(
#             nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(input_chs, momentum=bn_momentum),
#             nn.ReLU()
#         )
    
#     def forward(self, x):
#         branches = torch.cat([branch(x) for branch in self.branches], dim=1)
#         output = self.fuse_conv(branches)
#         output = self.conv_31(output)
#         output = self.conv_13(output)
#         output = self.conv_11(output) + x
#         return output
  
# demo2
class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock,self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

        self.post_conv = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(input_chs, input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

        self.caa_block = CAABlock(input_chs, 5, 5)
#
    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches)
        rp_output = output
        attn_factor = self.caa_block(x)
        rp_output = rp_output + attn_factor * rp_output
        return self.post_conv(rp_output)

class CAABlock(nn.Module):
    def __init__(self, channels: int, h_kernel_size: int = 5, v_kernel_size: int = 5, momentum=0.1, eps=0.001): # momentum=0.03
        super(CAABlock, self).__init__()
        self.pool = nn.AvgPool2d(3, 1, 1) # 713
        # self.pool = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
        #     nn.BatchNorm2d(channels, momentum=momentum, eps=eps),
        #     nn.SiLU()
        # )

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels, momentum=momentum, eps=eps),
            nn.SiLU()
        )
        # group=1
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, h_kernel_size), stride=1, padding=(0, h_kernel_size // 2), groups=channels),
            nn.BatchNorm2d(channels, momentum=momentum, eps=eps),
            nn.SiLU()
        )
        # group=1
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(v_kernel_size, 1), stride=1, padding=(v_kernel_size // 2, 0), groups=channels),
            nn.BatchNorm2d(channels, momentum=momentum, eps=eps),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels, momentum=momentum, eps=eps),
            nn.SiLU()
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.pool(x))))))
        return attn_factor

# base
class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock,self).__init__()
        self.branches = nn.ModuleList()
        # self.adjust1 = nn.Conv2d(int(input_chs*0.25), input_chs, kernel_size=1, stride=1, padding=0)
        # self.adjust2 = nn.Conv2d(int(input_chs*0.5), input_chs, kernel_size=1, stride=1, padding=0)
        for i, ratio in enumerate(ratios):
            # base
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )

            # split
            # if i != 0:
            #     conv = nn.Sequential(
            #         nn.Conv2d(input_chs, input_chs, kernel_size=((2 * i + 1), 1), stride=1, padding=(i, 0)),
            #         # nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            #         # nn.ReLU(),
            #         nn.Conv2d(input_chs, input_chs, kernel_size=(1, (2 * i + 1)), stride=1, padding=(0, i)),
            #         # nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            #         # nn.ReLU(),
            #         nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=1, stride=1, padding=0),
            #         nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
            #         nn.ReLU()
            #     )
            # else:
            # conv = nn.Sequential(
            #     nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
            #     nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
            #     nn.ReLU()
            # )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        # f5 = self.branches[2](x)
        # f3 = self.branches[1](identity + self.adjust1(f5))
        # f1 = self.branches[0](identity + self.adjust2(f3))
        # # output = f1
        # branches = torch.cat([f5, f3, f1], dim=1)
        output = self.fuse_conv(branches) + x
        return output

class L2HNet(nn.Module):
    def __init__(self, 
                 width, # width=64 for light mode; width=128 for normal mode
                 image_band=4, # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 output_chs=128, 
                 length=5, 
                 ratios=[1, 0.5, 0.25],
                 bn_momentum=0.1):
        super(L2HNet,self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)
        
        self.out_conv1 = nn.Sequential(
            StdConv2d(self.width * length, output_chs * length, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, output_chs*5, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv2 = nn.Sequential(
            StdConv2d(output_chs * length, 1024, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv3 = nn.Sequential(
            StdConv2d(1024, 1024, kernel_size=5, stride=4, bias=False, padding=1),
            nn.GroupNorm(32,1024, eps=1e-6),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.startconv(x)
        output_d1 = []
        for rpblk in self.rpblocks:
            x = rpblk(x)
            output_d1.append(x)
        output_d1 = self.out_conv1(torch.cat(output_d1, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        features = [output_d1, output_d2, output_d3, x]
        return output_d3, features[::-1]
