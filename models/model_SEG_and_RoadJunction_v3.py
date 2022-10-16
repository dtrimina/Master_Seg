import torch
from torch import nn


class ConvBnReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        padding = (kernel // 2) * dilation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding,
                              dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Model_Seg(nn.Module):
    def __init__(self, n_classes):
        super(Model_Seg, self).__init__()

        # self.channels = [24, 48, 96, 192, 384, 384]
        self.channels = [32, 64, 128, 256, 512, 512]
        scale = 1.0
        self.channels = [int(i * scale) for i in self.channels]

        self.backbone = nn.Sequential(
            ConvBnReLU(3, self.channels[0], kernel=3, stride=2),  # 0

            ConvBnReLU(self.channels[0], self.channels[1], kernel=3, stride=2),  # 1
            ConvBnReLU(self.channels[1], self.channels[1], kernel=3, stride=1),  # 2
            ConvBnReLU(self.channels[1], self.channels[1], kernel=3, stride=1),  # 3
            ConvBnReLU(self.channels[1], self.channels[1], kernel=3, stride=1),  # 4

            ConvBnReLU(self.channels[1], self.channels[2], kernel=3, stride=2),  # 5
            ConvBnReLU(self.channels[2], self.channels[2], kernel=3, stride=1),  # 6
            ConvBnReLU(self.channels[2], self.channels[2], kernel=3, stride=1),  # 7
            ConvBnReLU(self.channels[2], self.channels[2], kernel=3, stride=1),  # 8

            ConvBnReLU(self.channels[2], self.channels[3], kernel=3, stride=2),  # 9
            ConvBnReLU(self.channels[3], self.channels[3], kernel=3, stride=1),  # 10
            ConvBnReLU(self.channels[3], self.channels[3], kernel=3, stride=1),  # 11
            ConvBnReLU(self.channels[3], self.channels[3], kernel=3, stride=1),  # 12

            ConvBnReLU(self.channels[3], self.channels[4], kernel=3, stride=2),  # 13
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),  # 14
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),  # 15
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),  # 16
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),  # 17
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),  # 18

            ConvBnReLU(self.channels[4], self.channels[5], kernel=3, stride=2),  # 19
            ConvBnReLU(self.channels[5], self.channels[5], kernel=1, stride=1),  # 20
            ConvBnReLU(self.channels[5], self.channels[5], kernel=3, stride=1, dilation=2),  # 21
            ConvBnReLU(self.channels[5], self.channels[5], kernel=3, stride=1, dilation=2),  # 22
            ConvBnReLU(self.channels[5], self.channels[5], kernel=3, stride=1, dilation=2),  # 23
        )

        self.up2x = nn.UpsamplingNearest2d(scale_factor=2)
        self.up18x = nn.UpsamplingNearest2d(scale_factor=18)
        self.bilinear_up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.bilinear_up2x = nn.UpsamplingBilinear2d(scale_factor=2)

        # ---------------------------------------- seg --------------------------------------
        self.bridge_conv4 = ConvBnReLU(self.channels[1], self.channels[1], kernel=1, stride=1)
        self.bridge_conv8 = ConvBnReLU(self.channels[2], self.channels[2], kernel=1, stride=1)
        self.bridge_conv16 = ConvBnReLU(self.channels[3], self.channels[3], kernel=1, stride=1)
        self.bridge_conv32 = ConvBnReLU(self.channels[4], self.channels[4], kernel=1, stride=1)

        self.fuse_layer_64 = nn.Sequential(
            ConvBnReLU(self.channels[5], self.channels[4], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),
        )
        self.fuse_layer_32 = nn.Sequential(
            ConvBnReLU(self.channels[4] * 2 + 16, self.channels[3], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[3], self.channels[3], kernel=3, stride=1),
        )

        self.fuse_layer_16 = nn.Sequential(
            ConvBnReLU(self.channels[3] * 2 + 16, self.channels[2], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[2], self.channels[2], kernel=3, stride=1),
        )

        self.fuse_layer_8 = nn.Sequential(
            ConvBnReLU(self.channels[2] * 2 + 16, self.channels[1], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[1], self.channels[1], kernel=3, stride=1),
        )

        self.fuse_layer_4 = nn.Sequential(
            ConvBnReLU(self.channels[1] * 2 + 16, self.channels[0], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[0], self.channels[0], kernel=3, stride=1),
        )

        self.aux_x8_layer = nn.Sequential(
            ConvBnReLU(self.channels[2] // 2, 64, kernel=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.aux_x16_layer = nn.Sequential(
            ConvBnReLU(self.channels[3] // 2, 64, kernel=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.aux_x32_layer = nn.Sequential(
            ConvBnReLU(self.channels[4] // 2, 64, kernel=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )

        self.aux_g_cls_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[5], out_channels=512, kernel_size=9, padding=0, stride=9, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, padding=0, stride=1, bias=True),
        )

        self.body_layer = nn.Sequential(
            ConvBnReLU(self.channels[1] * 2 + 16, 64, kernel=1, stride=1),
            ConvBnReLU(64, 64, kernel=3, stride=1),
        )
        self.body_layer_out = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True)

        self.edge_layer = nn.Sequential(
            ConvBnReLU(self.channels[0] * 2, 32, kernel=1, stride=1),
            ConvBnReLU(32, 32, kernel=3, stride=1),
        )
        self.edge_layer_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1, bias=True)

        self.final_layer_out = nn.Sequential(
            ConvBnReLU(64 + 32, 64, kernel=1, stride=1),
            ConvBnReLU(64, 64, kernel=3, stride=1),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, stride=1, bias=True),
        )

        # ------------------------- od Road Junction --------------------------------
        self.od_bridge_conv8 = ConvBnReLU(self.channels[2], self.channels[2], kernel=1, stride=1)
        self.od_bridge_conv16 = ConvBnReLU(self.channels[3], self.channels[3], kernel=1, stride=1)
        self.od_bridge_conv32 = ConvBnReLU(self.channels[4], self.channels[4], kernel=1, stride=1)

        self.od_fuse_layer_64 = nn.Sequential(
            ConvBnReLU(self.channels[5], self.channels[4], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[4], self.channels[4], kernel=3, stride=1),
        )
        self.od_fuse_layer_32 = nn.Sequential(
            ConvBnReLU(self.channels[4] * 2, self.channels[3], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[3], self.channels[3], kernel=3, stride=1),
        )

        self.od_fuse_layer_16 = nn.Sequential(
            ConvBnReLU(self.channels[3] * 2, self.channels[2], kernel=1, stride=1),
            self.up2x,
            ConvBnReLU(self.channels[2], self.channels[2], kernel=3, stride=1),
        )

        self.od_fuse_layer_8 = nn.Sequential(
            ConvBnReLU(self.channels[2] * 2, self.channels[1], kernel=1, stride=1),
            ConvBnReLU(self.channels[1], self.channels[1], kernel=3, stride=1),
        )

        self.od_hm = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.od_wh = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.od_reg = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        self.od_heads = {'od_hm': 1, 'od_wh': 2, 'od_reg': 2}

        self.od_init_weights()

    def od_init_weights(self):
        for head in self.od_heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[0] == self.od_heads[head]:
                        if 'hm' in head:
                            print('init hm')
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat2 = self.backbone[0:1](x)
        feat4 = self.backbone[1:5](feat2)
        feat8 = self.backbone[5:9](feat4)
        feat16 = self.backbone[9:13](feat8)
        feat32 = self.backbone[13:19](feat16)
        feat64 = self.backbone[19:24](feat32)

        # ---------------------------- SEG -------------------------------
        seg_feat4 = self.bridge_conv4(feat4)
        seg_feat8 = self.bridge_conv8(feat8)
        seg_feat16 = self.bridge_conv16(feat16)
        seg_feat32 = self.bridge_conv32(feat32)

        feat64_deco_for_aux = self.fuse_layer_64[0:1](feat64)
        feat64_deco = self.fuse_layer_64[1:3](feat64_deco_for_aux)
        out_g_cls = self.aux_g_cls_layer(feat64_deco_for_aux)

        feat32_deco = torch.cat([seg_feat32, feat64_deco, self.up18x(out_g_cls)], dim=1)
        feat32_deco_for_aux = self.fuse_layer_32[0:1](feat32_deco)
        feat32_deco = self.fuse_layer_32[1:3](feat32_deco_for_aux)
        out_x32 = self.aux_x32_layer(feat32_deco_for_aux)

        feat16_deco = torch.cat([seg_feat16, feat32_deco, self.up2x(out_x32)], dim=1)
        feat16_deco_for_aux = self.fuse_layer_16[0:1](feat16_deco)
        feat16_deco = self.fuse_layer_16[1:3](feat16_deco_for_aux)
        out_x16 = self.aux_x16_layer(feat16_deco_for_aux)

        feat8_deco = torch.cat([seg_feat8, feat16_deco, self.up2x(out_x16)], dim=1)
        feat8_deco_for_aux = self.fuse_layer_8[0:1](feat8_deco)
        feat8_deco = self.fuse_layer_8[1:3](feat8_deco_for_aux)
        out_x8 = self.aux_x8_layer(feat8_deco_for_aux)

        feat4_deco = torch.cat([seg_feat4, feat8_deco, self.up2x(out_x8)], dim=1)
        feat4_deco_for_edge = self.fuse_layer_4(feat4_deco)

        feat_body = self.body_layer(feat4_deco)
        pred_body = self.body_layer_out(feat_body)

        feat_edge = torch.cat([feat2, feat4_deco_for_edge], dim=1)
        feat_edge = self.edge_layer(feat_edge)
        pred_edge = self.edge_layer_out(feat_edge)

        feat_final = torch.cat([self.up2x(feat_body), feat_edge], dim=1)
        pred_cls = self.final_layer_out(feat_final)

        # print(feat_body.shape, feat_edge.shape)

        # feat4_final = self.final_layer_4(feat4_deco)
        #
        # pred_cls = self.final_bilinear_up4x(feat4_final)

        pred_body = self.bilinear_up4x(pred_body)
        pred_edge = self.bilinear_up2x(pred_edge)
        pred_cls = self.bilinear_up2x(pred_cls)

        out = {
            'pred_body': pred_body,
            'pred_edge': pred_edge,
            'pred_cls': pred_cls
        }

        if self.training:
            # out['pred_cls_out8'] = nn.UpsamplingBilinear2d(scale_factor=8)(out_x8)
            # out['pred_cls_out16'] = nn.UpsamplingBilinear2d(scale_factor=16)(out_x16)
            # out['pred_cls_out32'] = nn.UpsamplingBilinear2d(scale_factor=32)(out_x16)
            out['pred_cls_out8'] = out_x8
            out['pred_cls_out16'] = out_x16
            out['pred_cls_out32'] = out_x32
            out['pred_g_cls'] = out_g_cls


        #------------------------------- od Road Junction ------------------
        od_feat8 = self.od_bridge_conv8(feat8)
        od_feat16 = self.od_bridge_conv16(feat16)
        od_feat32 = self.od_bridge_conv32(feat32)

        od_feat64_deco = self.od_fuse_layer_64[0:3](feat64)

        od_feat32_deco = torch.cat([od_feat32, od_feat64_deco], dim=1)
        od_feat32_deco = self.od_fuse_layer_32(od_feat32_deco)

        od_feat16_deco = torch.cat([od_feat16, od_feat32_deco], dim=1)
        od_feat16_deco = self.od_fuse_layer_16(od_feat16_deco)

        od_feat8_deco = torch.cat([od_feat8, od_feat16_deco], dim=1)
        od_feat8_deco = self.od_fuse_layer_8(od_feat8_deco)

        pred_hm = self.od_hm(od_feat8_deco)
        pred_wh = self.od_wh(od_feat8_deco)
        pred_reg = self.od_reg(od_feat8_deco)

        out['pred_hm'] = pred_hm
        out['pred_wh'] = pred_wh
        out['pred_reg'] = pred_reg


        return out


if __name__ == '__main__':
    model = Model_Seg(n_classes=16)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 576, 576), print_per_layer_stat=False, as_strings=True)
    print('GFlops:  ', flops)
    print('MParams: ', params)

    # # baseline for 576 x 576
    # GFlops: 16.93 GMac
    # MParams: 20.22 M

    # # new model for 576 x 576
    # GFlops: 18.07 GMac
    # MParams: 29.59 M
