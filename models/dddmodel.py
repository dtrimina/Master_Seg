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
            ('relu', nn.ReLU(inplace=True))]))


class Attation(nn.Module):

    def __init__(self, feature):
        super(Attation, self).__init__()

        self.conv1 = ConvBNReLU(feature, feature // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(feature // 2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        feat1 = self.conv1(feat)
        feat2 = self.conv2(feat1)
        return feat1, self.sigmoid(feat2) * feat


class ModelDual(nn.Module):

    def __init__(self, n_classes):
        super(ModelDual, self).__init__()

        self.channels = [64, 128, 256, 512]
        self.n_classes = n_classes
        self.backbone_rgb = torchvision.models.resnet34(pretrained=True)
        self.backbone_dep = torchvision.models.resnet34(pretrained=True)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)

        # ------------------- dep fusion ---------------
        self.dep1_fusion_layer_dep2conv1x1 = ConvBNReLU(128, 64, kernel_size=1)
        self.dep1_fusion_layer_dep3conv1x1 = ConvBNReLU(256, 64, kernel_size=1)
        self.dep1_fusion_layer_dep4conv1x1 = ConvBNReLU(512, 64, kernel_size=1)
        self.dep1_fusion_layer = ConvBNReLU(64 * 4, 64, kernel_size=1)

        self.dep2_fusion_layer_dep3conv1x1 = ConvBNReLU(256, 128, kernel_size=1)
        self.dep2_fusion_layer_dep4conv1x1 = ConvBNReLU(512, 128, kernel_size=1)
        self.dep2_fusion_layer = ConvBNReLU(128 * 3, 128, kernel_size=1)

        self.dep3_fusion_layer_dep4conv1x1 = ConvBNReLU(512, 256, kernel_size=1)
        self.dep3_fusion_layer = ConvBNReLU(256 * 2, 256, kernel_size=1)

        self.dep4_fusion_layer = ConvBNReLU(512 * 1, 512, kernel_size=1)

        # ------------------- saliency branch
        self.att1 = Attation(64)
        self.att2 = Attation(128)
        self.att3 = Attation(256)
        self.att4 = Attation(512)

        self.saliency_layer = nn.Sequential(
            ConvBNReLU(480, 64, kernel_size=1),
            self.up2x,
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            self.up2x,
        )

        # ------------------- segmentation layer
        self.deconv4_3 = nn.Sequential(
            ConvBNReLU(512, 256, kernel_size=1),
            ConvBNReLU(256, 256, kernel_size=3)
        )
        self.deconv3_2 = nn.Sequential(
            ConvBNReLU(256, 128, kernel_size=1),
            ConvBNReLU(128, 128, kernel_size=3)
        )
        self.deconv2_1 = nn.Sequential(
            ConvBNReLU(128, 64, kernel_size=1),
            ConvBNReLU(64, 64, kernel_size=3)
        )

        self.seg_fusion_layer_4_3_conv1x1 = ConvBNReLU(512, 256, kernel_size=1)
        self.seg_fusion_layer_4_2_conv1x1 = ConvBNReLU(512, 128, kernel_size=1)
        self.seg_fusion_layer_4_1_conv1x1 = ConvBNReLU(512, 64, kernel_size=1)
        self.seg_fusion_layer_3_2_conv1x1 = ConvBNReLU(256, 128, kernel_size=1)
        self.seg_fusion_layer_3_1_conv1x1 = ConvBNReLU(256, 64, kernel_size=1)
        self.seg_fusion_layer_2_1_conv1x1 = ConvBNReLU(128, 64, kernel_size=1)

        self.seg_layer = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=1),
            ConvBNReLU(64, 64, kernel_size=3),
            self.up2x,
            nn.Conv2d(64, self.n_classes, kernel_size=3, padding=1),
            self.up2x
        )

        # ------------------- boundary layer
        self.boundary_layer = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=1),
            ConvBNReLU(64, 64, kernel_size=3),
            self.up2x,
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            self.up2x,
        )

    def forward(self, rgb_in, dep_in=None):  # rgb, dep

        if dep_in is None:
            dep_in = rgb_in

        #  --------------------  dep fusion layers
        dep = self.backbone_dep.conv1(dep_in)
        dep = self.backbone_dep.bn1(dep)
        dep = self.backbone_dep.relu(dep)
        dep = self.backbone_dep.maxpool(dep)
        dep1 = self.backbone_dep.layer1(dep)
        dep2 = self.backbone_dep.layer2(dep1)
        dep3 = self.backbone_dep.layer3(dep2)
        dep4 = self.backbone_dep.layer4(dep3)

        dep_fus1 = torch.cat([
            dep1,
            self.up2x(self.dep1_fusion_layer_dep2conv1x1(dep2)),
            self.up4x(self.dep1_fusion_layer_dep3conv1x1(dep3)),
            self.up8x(self.dep1_fusion_layer_dep4conv1x1(dep4))
        ], dim=1)
        dep_fus1 = self.dep1_fusion_layer(dep_fus1)

        dep_fus2 = torch.cat([
            dep2,
            self.up2x(self.dep2_fusion_layer_dep3conv1x1(dep3)),
            self.up4x(self.dep2_fusion_layer_dep4conv1x1(dep4))
        ], dim=1)
        dep_fus2 = self.dep2_fusion_layer(dep_fus2)

        dep_fus3 = torch.cat([
            dep3,
            self.up2x(self.dep3_fusion_layer_dep4conv1x1(dep4))
        ], dim=1)
        dep_fus3 = self.dep3_fusion_layer(dep_fus3)

        dep_fus4 = self.dep4_fusion_layer(dep4)

        # ---------------------------------  rgb + dep fusion
        rgb = self.backbone_rgb.conv1(rgb_in)
        rgb = self.backbone_rgb.bn1(rgb)
        rgb = self.backbone_rgb.relu(rgb)
        rgb = self.backbone_rgb.maxpool(rgb)
        rgb1 = self.backbone_rgb.layer1(rgb)

        rgb1 = rgb1 + dep_fus1
        rgb2 = self.backbone_rgb.layer2(rgb1)
        rgb2 = rgb2 + dep_fus2
        rgb3 = self.backbone_rgb.layer3(rgb2)
        rgb3 = rgb3 + dep_fus3
        rgb4 = self.backbone_rgb.layer4(rgb3)
        rgb4 = rgb4 + dep_fus4

        # --------------------------------- saliency supervision
        saliency_feat1, feat1 = self.att1(rgb1)
        saliency_feat2, feat2 = self.att2(rgb2)
        saliency_feat3, feat3 = self.att3(rgb3)
        saliency_feat4, feat4 = self.att4(rgb4)

        saliency_feat = torch.cat([
            saliency_feat1,
            self.up2x(saliency_feat2),
            self.up4x(saliency_feat3),
            self.up8x(saliency_feat4)
        ], dim=1)
        pred_saliency = self.saliency_layer(saliency_feat)

        # --------------------------------- segmentation supervision
        seg4 = self.deconv4_3(feat4)
        seg3 = self.up2x(seg4)
        seg3 = seg3 + feat3 + self.up2x(self.seg_fusion_layer_4_3_conv1x1(feat4))

        seg2 = self.deconv3_2(seg3)
        seg2 = self.up2x(seg2)
        seg2 = seg2 + feat2 + self.up2x(self.seg_fusion_layer_3_2_conv1x1(feat3)) + \
               self.up4x(self.seg_fusion_layer_4_2_conv1x1(feat4))

        seg1 = self.deconv2_1(seg2)
        seg1 = self.up2x(seg1)
        seg1 = seg1 + feat1 + self.up2x(self.seg_fusion_layer_2_1_conv1x1(feat2)) + \
               self.up4x(self.seg_fusion_layer_3_1_conv1x1(feat3)) + \
               self.up8x(self.seg_fusion_layer_4_1_conv1x1(feat4))

        pred_cls = self.seg_layer(seg1)

        # ---------------------------------- boundary
        pred_boundary = self.boundary_layer(seg1)

        out = {
            'pred_cls': pred_cls,
            'pred_saliency': pred_saliency,
            'pred_boundary': pred_boundary
        }

        return out



if __name__ == '__main__':
    model = ModelDual(9)
    input = torch.randn((2, 3, 480, 640))
    model(input, input)

    # import torchvision
    # model = torchvision.models.mobilenet_v2(pretrained=True)
    # print(model)

    # from torchsummary import summary
    #
    # model = ModelDual(9).cuda()
    # summary(model, [(3, 480, 640), (3, 480, 640)])  # 1024, 2048

    from utils.utils import compute_speed
    from ptflops import get_model_complexity_info

    # with torch.cuda.device(0):
    net = ModelDual(n_classes=9)
    flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)

    # compute_speed(net, input_size=(1, 3, 480, 640), iteration=500)

    # Flops:  18.44 GMac
    # Params: 7.49 M
    # =========Eval Forward Time=========
    # Elapsed Time: [6.19 s / 500 iter]
    # Speed Time: 12.39 ms / iter   FPS: 80.74
