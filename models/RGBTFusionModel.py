import torch
import torch.nn as nn

import torchvision

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


class FFM(nn.Module):

    def __init__(self, feature):
        super(FFM, self).__init__()

        self.conv = nn.Sequential(
            ConvBNReLU(feature * 2, feature, kernel_size=1),
            ConvBNReLU(feature, feature, kernel_size=3)
        )

    def forward(self, feat_rgb, feat_dep):
        feat = torch.cat([feat_rgb, feat_dep], dim=1)
        feat = self.conv(feat)
        return feat


class RGBTFusionModel(nn.Module):

    def __init__(self, n_classes):
        super(RGBTFusionModel, self).__init__()

        self.channels = [256, 512, 1024, 2048]
        self.n_classes = n_classes
        self.backbone_rgb = torchvision.models.resnet50(pretrained=True)
        self.backbone_dep = torchvision.models.resnet50(pretrained=True)

        self.fusion1 = FFM(256)
        self.fusion2 = FFM(512)
        self.fusion3 = FFM(1024)
        self.fusion4 = FFM(2048)

        # -----------------  decoder  -------------------
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)

        # ---------------------------------------- seg --------------------------------------
        self.bridge_conv4 = ConvBNReLU(self.channels[1], self.channels[1], kernel_size=1, stride=1)
        self.bridge_conv8 = ConvBNReLU(self.channels[2], self.channels[2], kernel_size=1, stride=1)
        self.bridge_conv16 = ConvBNReLU(self.channels[3], self.channels[3], kernel_size=1, stride=1)

        self.fuse_layer_32 = nn.Sequential(
            # ConvBNReLU(self.channels[4] * 2 + self.n_classes, self.channels[3], kernel_size=1, stride=1),
            ConvBNReLU(self.channels[3], self.channels[2], kernel_size=1, stride=1),
            self.up2x,
            ConvBNReLU(self.channels[2], self.channels[2], kernel_size=3, stride=1),
        )

        self.fuse_layer_16 = nn.Sequential(
            ConvBNReLU(self.channels[2] * 2 + self.n_classes, self.channels[1], kernel_size=1, stride=1),
            self.up2x,
            ConvBNReLU(self.channels[1], self.channels[1], kernel_size=3, stride=1),
        )

        self.fuse_layer_8 = nn.Sequential(
            ConvBNReLU(self.channels[1] * 2 + self.n_classes, self.channels[0], kernel_size=1, stride=1),
            self.up2x,
            ConvBNReLU(self.channels[0], self.channels[0], kernel_size=3, stride=1),
        )

        self.fuse_layer_4 = nn.Sequential(
            ConvBNReLU(self.channels[0] * 2 + self.n_classes, self.channels[0], kernel_size=1, stride=1),
            self.up2x,
            ConvBNReLU(self.channels[0], self.channels[0], kernel_size=3, stride=1),
        )

        self.aux_x8_layer = nn.Sequential(
            ConvBNReLU(self.channels[1] // 2, 64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.aux_x16_layer = nn.Sequential(
            ConvBNReLU(self.channels[2] // 2, 64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.aux_x32_layer = nn.Sequential(
            ConvBNReLU(self.channels[3] // 2, 64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )

        # self.aux_g_cls_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channels[5], out_channels=512, kernel_size=9, padding=0, stride=9, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, padding=0, stride=1, bias=True),
        # )


        self.final_layer_out = nn.Sequential(
            ConvBNReLU(self.channels[0], 64, kernel_size=1, stride=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )

    def forward(self, rgb_in, dep_in=None):  # rgb, dep

        if dep_in is None:
            dep_in = rgb_in

        rgb = self.backbone_rgb.conv1(rgb_in)
        rgb = self.backbone_rgb.bn1(rgb)
        rgb = self.backbone_rgb.relu(rgb)
        rgb = self.backbone_rgb.maxpool(rgb)
        rgb1 = self.backbone_rgb.layer1(rgb)

        dep = self.backbone_dep.conv1(dep_in)
        dep = self.backbone_dep.bn1(dep)
        dep = self.backbone_dep.relu(dep)
        dep = self.backbone_dep.maxpool(dep)
        dep1 = self.backbone_dep.layer1(dep)

        fusion_feat1 = self.fusion1(rgb1, dep1)
        rgb1 = rgb1 + fusion_feat1
        dep1 = dep1 + fusion_feat1

        rgb2 = self.backbone_rgb.layer2(rgb1)
        dep2 = self.backbone_dep.layer2(dep1)
        fusion_feat2 = self.fusion2(rgb2, dep2)
        rgb2 = rgb2 + fusion_feat2
        dep2 = dep2 + fusion_feat2

        rgb3 = self.backbone_rgb.layer3(rgb2)
        dep3 = self.backbone_dep.layer3(dep2)
        fusion_feat3 = self.fusion3(rgb3, dep3)
        rgb3 = rgb3 + fusion_feat3
        dep3 = dep3 + fusion_feat3

        rgb4 = self.backbone_rgb.layer4(rgb3)
        dep4 = self.backbone_dep.layer4(dep3)
        fusion_feat4 = self.fusion4(rgb4, dep4)


        # ---------------------------- SEG -------------------------------

        # print([i.shape for i in [fusion_feat1, fusion_feat2, fusion_feat3, fusion_feat4]])

        feat32_deco_for_aux = self.fuse_layer_32[0:1](fusion_feat4)
        out_x32 = self.aux_x32_layer(feat32_deco_for_aux)
        feat32_deco = self.fuse_layer_32[1:3](feat32_deco_for_aux)
        print(feat32_deco.shape)

        feat16_deco = torch.cat([fusion_feat3, feat32_deco, self.up2x(out_x32)], dim=1)
        feat16_deco_for_aux = self.fuse_layer_16[0:1](feat16_deco)
        out_x16 = self.aux_x16_layer(feat16_deco_for_aux)
        feat16_deco = self.fuse_layer_16[1:3](feat16_deco_for_aux)
        print(feat16_deco.shape)

        feat8_deco = torch.cat([fusion_feat2, feat16_deco, self.up2x(out_x16)], dim=1)
        feat8_deco_for_aux = self.fuse_layer_8[0:1](feat8_deco)
        out_x8 = self.aux_x8_layer(feat8_deco_for_aux)
        feat8_deco = self.fuse_layer_8[1:3](feat8_deco_for_aux)
        print(feat8_deco.shape)

        feat4_deco = torch.cat([fusion_feat1, feat8_deco, self.up2x(out_x8)], dim=1)
        feat4_deco_for_final = self.fuse_layer_4(feat4_deco)

        pred_cls = self.final_layer_out(feat4_deco_for_final)
        pred_cls = self.up4x(pred_cls)

        out = {
            'pred_cls': pred_cls
        }

        if self.training:
            out['pred_cls_out8'] = out_x8
            out['pred_cls_out16'] = out_x16
            out['pred_cls_out32'] = out_x32
            # out['pred_g_cls'] = out_g_cls


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

    # with torch.cuda.device(0):
    net = RGBTFusionModel(n_classes=41)
    flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)

    # compute_speed(net, input_size=(1, 3, 480, 640), iteration=500)

    # Flops:  18.44 GMac
    # Params: 7.49 M
    # =========Eval Forward Time=========
    # Elapsed Time: [6.19 s / 500 iter]
    # Speed Time: 12.39 ms / iter   FPS: 80.74
