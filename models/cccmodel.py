import torch
import torch.nn as nn
from models.mobilenetv2 import mobilenet_v2
import collections


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(collections.OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('relu', nn.ReLU6(inplace=True))]))


class SFM(nn.Module):

    def __init__(self, feature):
        super(SFM, self).__init__()

        self.gp = nn.AdaptiveAvgPool2d(1)
        self.down = ConvBNReLU(in_planes=feature, out_planes=feature // 2, kernel_size=1)
        self.up = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels=feature // 2, out_channels=feature, kernel_size=1, stride=1, padding=0)),
            ('bn', nn.BatchNorm2d(feature)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, rgb, dep):
        fus = self.gp(rgb + dep)
        fus = self.down(fus)
        logit = self.up(fus)
        out = rgb + dep * logit
        return out


class FFM(nn.Module):
    def __init__(self, low_chan, high_chan, dilations=(1, 2, 4)):
        super(FFM, self).__init__()
        self.conv1 = ConvBNReLU(low_chan, low_chan, kernel_size=3, dilation=dilations[0], groups=low_chan)
        self.conv2 = ConvBNReLU(low_chan, low_chan, kernel_size=3, dilation=dilations[1], groups=low_chan)
        self.conv3 = ConvBNReLU(low_chan, low_chan, kernel_size=3, dilation=dilations[2], groups=low_chan)

        self.conv_aspp_out = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(low_chan * 4, high_chan, kernel_size=1, padding=0, stride=1)),
            ('dropout', nn.Dropout2d(0.2)),
            ('relu', nn.ReLU6(inplace=True))
        ]))
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_out = ConvBNReLU(high_chan * 2, low_chan, kernel_size=1)

    def forward(self, low_level, high_level):
        # low level
        feat1 = self.conv1(low_level)
        feat2 = self.conv2(low_level)
        feat3 = self.conv3(low_level)
        feat = self.conv_aspp_out(torch.cat([low_level, feat1, feat2, feat3], 1))

        # high level
        if low_level.size(2) != high_level.size(2):
            high_level = self.up2x(high_level)

        out = self.conv_out(torch.cat([feat, high_level], 1)) + low_level

        return out


class MobileDual(nn.Module):

    def __init__(self, n_classes):
        super(MobileDual, self).__init__()

        self.mobile_rgb = mobilenet_v2(pretrained=True)
        self.mobile_dep = mobilenet_v2(pretrained=True)

        self.sfm1 = SFM(24)
        self.sfm2 = SFM(32)
        self.sfm3 = SFM(64)
        self.sfm4 = SFM(96)
        self.sfm5 = SFM(160)

        self.ffm1 = FFM(24, 32)
        self.ffm2 = FFM(32, 64)
        self.ffm3 = FFM(64, 96)
        self.ffm4 = FFM(96, 160)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)

        self.classifier = nn.Sequential(
            ConvBNReLU(376, 376, kernel_size=1, stride=1),
            self.up2x,
            nn.Conv2d(376, n_classes, kernel_size=3, padding=1),
            self.up2x
        )

    def forward(self, rgb, dep=None):  # rgb, dep
        # rgb = ins[:, :3, :, :]
        # dep = ins[:, 3:, :, :]
        if dep is None:
            dep = rgb
        dep1 = self.mobile_dep.features[0:4](dep)
        dep2 = self.mobile_dep.features[4:7](dep1)
        dep3 = self.mobile_dep.features[7:11](dep2)
        dep4 = self.mobile_dep.features[11:14](dep3)
        dep5 = self.mobile_dep.features[14:17](dep4)

        fus = self.mobile_rgb.features[0:4](rgb)
        fus1 = self.sfm1(fus, dep1)
        fus2 = self.sfm2(self.mobile_rgb.features[4:7](fus1), dep2)
        fus3 = self.sfm3(self.mobile_rgb.features[7:11](fus2), dep3)
        fus4 = self.sfm4(self.mobile_rgb.features[11:14](fus3), dep4)
        fus5 = self.sfm5(self.mobile_rgb.features[14:17](fus4), dep5)
        # print([i.shape for i in [dep1, dep2, dep3, dep4, dep5]])

        # 自上而下融合
        out4 = self.ffm4(fus4, fus5)
        out3 = self.ffm3(fus3, out4)
        out2 = self.ffm2(fus2, out3)
        out1 = self.ffm1(fus1, out2)

        # # 两层两层融合
        # out4 = self.ffm4(fus4, fus5)
        # out3 = self.ffm3(fus3, fus4)
        # out2 = self.ffm2(fus2, fus3)
        # out1 = self.ffm1(fus1, fus2)

        out5 = self.up4x(fus5)
        out4 = self.up4x(out4)
        out3 = self.up4x(out3)
        out2 = self.up2x(out2)
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)

        out = self.classifier(out)

        return out


if __name__ == '__main__':
    # model = MobileDual(41)
    # input = torch.randn((2, 3, 480, 640))
    # model.get_params()
    # model(input, input)

    # import torchvision
    # model = torchvision.models.mobilenet_v2(pretrained=True)
    # print(model)

    # from torchsummary import summary
    #
    # model = MobileDual(41).cuda()
    # summary(model, [(3, 480, 640), (3, 480, 640)])  # 1024, 2048

    from toolbox import compute_speed
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        net = MobileDual(n_classes=41)
        flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
        print('Flops:  ' + flops)
        print('Params: ' + params)

    compute_speed(net, input_size=(1, 3, 480, 640), iteration=500)

    # Flops:  18.44 GMac
    # Params: 7.49 M
    # =========Eval Forward Time=========
    # Elapsed Time: [6.19 s / 500 iter]
    # Speed Time: 12.39 ms / iter   FPS: 80.74
