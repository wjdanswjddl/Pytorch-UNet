import torch
from torch import nn
from torch.nn import functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class PadCat(nn.Module):
    '''
    pad x1 to x2 in H, W dims and concatenate them in C dim
    '''
    def __init__(self):
        super(PadCat, self).__init__()

    def forward(self, x):
        n = len(x)
        assert n>1, "At least 2 tensors"
        x2 = x[0]
        x_paded = [x2]
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        for i in range(1, n):
            x1 = x[i]
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
            x_paded.append(x1)

        x = torch.cat(x_paded, dim=1)
        return x

class NestedUNet(nn.Module):
    def __init__(self, input_channels, output_channels, deepsupervision=False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.deepsupervision = deepsupervision

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cat = PadCat()

        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.output_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.output_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.output_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.output_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.output_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.cat([x0_0, self.up(x1_0)]))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.cat([x1_0, self.up(x2_0)]))
        x0_2 = self.conv0_2(self.cat([x0_0, x0_1, self.up(x1_1)]))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.cat([x2_0, self.up(x3_0)]))
        x1_2 = self.conv1_2(self.cat([x1_0, x1_1, self.up(x2_1)]))
        x0_3 = self.conv0_3(self.cat([x0_0, x0_1, x0_2, self.up(x1_2)]))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.cat([x3_0, self.up(x4_0)]))
        x2_2 = self.conv2_2(self.cat([x2_0, x2_1, self.up(x3_1)]))
        x1_3 = self.conv1_3(self.cat([x1_0, x1_1, x1_2, self.up(x2_2)]))
        x0_4 = self.conv0_4(self.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)]))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return torch.sigmoid(output)