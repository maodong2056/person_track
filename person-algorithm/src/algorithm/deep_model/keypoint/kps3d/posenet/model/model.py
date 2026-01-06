import torch
import torch.nn as nn
from torch.nn import functional as F
from src.algorithm.deep_model.arch.backbones.mobilenext import MobileNeXt
from src.algorithm.deep_model.arch.necks import DepthwiseShortcutDeconv

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def soft_argmax(heatmaps, joint_num):
    # N, K, H, W = heatmaps.shape
    N, K, H, W = 1, 1152, 64, 64
    # depth_dim = int(K / joint_num)
    depth_dim = 64

    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim * H * W))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim, H, W))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(W).float().to(heatmaps.device)[None, None, :]
    accu_y = accu_y * torch.arange(H).float().to(heatmaps.device)[None, None, :]
    accu_z = accu_z * torch.arange(depth_dim).float().to(heatmaps.device)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class MobilePoseNet(nn.Module):
    def __init__(self, joint_num, depth_dim=64, head_conv=64, out_stages=(1, 2, 4, 7,)):
        super(MobilePoseNet, self).__init__()

        self.backbone = MobileNeXt(width_mult=1.,
                                   identity_tensor_multiplier=1.0,
                                   out_stages=out_stages,
                                   last_channel=1280,
                                   activation='ReLU6'

                                   )
        self.backbone.init_weights()

        self.neck = DepthwiseShortcutDeconv(in_channels=(144, 192, 384, 1280,),
                                            deconv_channels=(256, 128, 128,),
                                            activation='ReLU6')

        self.inplanes = 128
        self.outplanes = 64

        # self.final_layer = nn.Conv2d(
        #     in_channels=self.inplanes,
        #     out_channels=joint_num * depth_dim,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        self.final_layer = nn.Sequential(
            nn.Conv2d(self.inplanes, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, joint_num * depth_dim, 1, 1, 0, bias=True)
        )
        # self.heat2D = nn.Conv2d(
        #     in_channels=self.inplanes,
        #     out_channels=joint_num,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        self.heat2D = nn.Sequential(
            nn.Conv2d(self.inplanes, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, joint_num, 1, 1, 0, bias=True)
        )
        # for m in self.final_layer:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        # for m in self.heat2D:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        # self.heat2D[-1].bias.data.fill_(-4.595)
        self.joint_num = joint_num
        self.vis_loss = nn.MSELoss(reduction='none')

    def forward(self, input_img, target=None):

        fm = self.backbone(input_img)

        features = self.neck(fm)

        hm = self.final_layer(features[0])

        heat2D = self.heat2D(features[0])

        # print(hm.shape)
        coord = soft_argmax(hm, self.joint_num)

        N, K, H, W = heat2D.shape
        # N, K, H, W = 1, 18, 64, 64
        coord_index = (coord[:, :, 1].int() * W + coord[:, :, 0].int()).long()
        coord_index += (W * H * torch.arange(N * K, device=coord.device)).reshape(N, -1)
        heat_conf = heat2D.reshape(heat2D.shape[0], -1)
        # conf = heat2D.flatten()[coord_index].unsqueeze(-1)
        conf = heat2D.reshape(-1)[coord_index].unsqueeze(-1)
        # conf = heat2D.reshape(-1)
        coord = torch.cat([coord, conf], dim=2)
        # return hm, conf
        # print(conf)
        if target is None:
            return coord
        # else:
        #     target_coord = target['coord']
        #     target_vis = target['vis']
        #     target_have_depth = target['have_depth']
        #
        #     ## coordinate loss
        #     loss_coord = torch.abs(coord - target_coord) * target_vis
        #     loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * target_have_depth) / 3.
        #
        #     target_heat, target_weight = generate_gaussian_heatmap(target_coord, target_vis,
        #                                                            (H, W),
        #                                                            self.joint_num)
        #
        #     loss_vis = self.vis_loss(torch.mul(heat2D, target_weight), target_heat).mean(dim=(-1, -2))
        #     # print(loss_vis)
        #
        #     return loss_coord + loss_vis

def soft_argmax_acc(acc):
    accu = acc * torch.arange(acc.shape[2]).float().to(acc.device)[None, None, :]
    accu = accu.sum(dim=2, keepdim=True)
    return accu

class MobilePoseNetLite(nn.Module):
    def __init__(self, joint_num, depth_dim=64, head_conv=64, output_shape=(256, 256), simdr_split_ratio=1., out_stages=(7,)):
        super(MobilePoseNetLite, self).__init__()
        self.output_shape = output_shape
        self.backbone = MobileNeXt(width_mult=1.,
                                   identity_tensor_multiplier=1.0,
                                   out_stages=out_stages,
                                   last_channel=1280,
                                   activation='ReLU6'

                                   )
        self.backbone.init_weights()
        self.final_layer = nn.Conv2d(
            in_channels=1280,
            out_channels=joint_num * 14,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # head
        self.mlp_head_x = nn.Linear(14 * 8 * 8, int(output_shape[0] * simdr_split_ratio))
        self.mlp_head_y = nn.Linear(14 * 8 * 8, int(output_shape[1] * simdr_split_ratio))
        self.mlp_head_z = nn.Linear(14 * 8 * 8, int(depth_dim * simdr_split_ratio))
        self.vis_prob = nn.Conv1d(
            in_channels=14 * 8 * 8,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.softmax = nn.Softmax(dim=-1)
        # self.loss = NMTCritierion(label_smoothing=0.1)

        # self.neck = DepthwiseShortcutDeconv(in_channels=(144, 192, 384, 1280,),
        #                                     deconv_channels=(256, 128, 128,),
        #                                     activation='ReLU6')
        #
        # self.inplanes = 128
        # self.outplanes = 64
        #
        # self.final_layer = nn.Conv2d(
        #     in_channels=self.inplanes,
        #     out_channels=joint_num * cfg.depth_dim,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        # self.heat2D = nn.Conv2d(
        #     in_channels=self.inplanes,
        #     out_channels=joint_num,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        self.joint_num = joint_num
        self.channel_per_joint = 14
        self.vis_loss = nn.MSELoss(reduction='none')

    def forward(self, input_img, target=None):

        fm = self.backbone(input_img)[0]

        # features = self.neck(fm)

        x = self.final_layer(fm)
        # x = rearrange(x, 'b (k t) h w -> b k (t h w)', k=self.joint_num, t=self.channel_per_joint)
        N, K, H, W = x.shape
        depth_dim = int(K / self.joint_num)
        x = x.reshape((-1, self.joint_num, depth_dim * H * W))

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)
        pred_z = self.mlp_head_z(x)

        soft_x = self.softmax(pred_x)
        soft_y = self.softmax(pred_y)
        soft_z = self.softmax(pred_z)

        idx_x = torch.argmax(soft_x, dim=-1, keepdim=True).float()
        idx_y = torch.argmax(soft_y, dim=-1, keepdim=True).float()
        idx_z = soft_argmax_acc(soft_z)
        prob = self.vis_prob(x.permute(0, 2, 1))
        # max_idx_x = torch.argmax(soft_x, dim=-1, keepdim=True).float()
        # max_idx_y = torch.argmax(soft_y, dim=-1, keepdim=True).float()
        # max_idx_z = torch.argmax(soft_z, dim=-1, keepdim=True).float()
        # print(hm.shape)
        N, K, W = soft_x.shape
        x_idx = soft_x.argmax(dim=-1).int().reshape(N, -1) + (W * torch.arange(N * K, device=idx_x.device)).reshape(N, -1)
        conf_x = soft_x.flatten()[x_idx]
        N, K, H = soft_y.shape
        y_idx = soft_y.argmax(dim=-1).int().reshape(N, -1) + (H * torch.arange(N * K, device=idx_x.device)).reshape(N, -1)
        conf_y = soft_y.flatten()[y_idx]
        conf = torch.where(conf_x > conf_y, conf_x, conf_y).unsqueeze(-1)
        coord = torch.cat((idx_x, idx_y, idx_z, conf), dim=2)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * target_have_depth) / 3.
            loss_vis = self.vis_loss(prob.permute(0, 2, 1)[:, :, 0], target_vis[:, :, 0])
            #
            # target_heat = generate_gaussian_heatmap(target_coord, target_vis,
            #                                         (cfg.output_shape[0], cfg.output_shape[1]),
            #                                         self.joint_num)
            #
            # loss_vis = self.vis_loss(heat2D, target_heat).mean(dim=(-1, -2))
            # print(loss_vis)
            # loss = self.loss(pred_x, pred_y, pred_z, target_coord, target_vis, target_have_depth)

        return loss_coord + loss_vis

def generate_gaussian_heatmap(target_coord, target_vis, heatmap_size, joint_number, factor=0.0546875):
    target = torch.zeros((target_coord.shape[0], joint_number, heatmap_size[1] * heatmap_size[0]),
                         device=target_coord.device, dtype=target_coord.dtype)
    target_weight = target_vis[:, :, :, None]
    feat_width = heatmap_size[0]
    feat_height = heatmap_size[1]
    feat_x_int = torch.arange(0, feat_width, device=target_coord.device)
    feat_y_int = torch.arange(0, feat_height, device=target_coord.device)
    feat_y_int, feat_x_int = torch.meshgrid(feat_y_int, feat_x_int)
    feat_x_int = feat_x_int.flatten()
    feat_y_int = feat_y_int.flatten()
    valid_radius = factor * heatmap_size[1]
    for i in range(target_coord.shape[0]):
        # batch
        for j in range(joint_number):
            # joint
            mu_x = target_coord[i][j][0]
            mu_y = target_coord[i][j][1]
            x_offset = (mu_x - feat_x_int) / valid_radius
            y_offset = (mu_y - feat_y_int) / valid_radius
            dis = x_offset ** 2 + y_offset ** 2
            keep_pos = torch.where(dis <= 1)[0]
            v = target_vis[i][j]
            if v > 0.5:
                target[i, j, keep_pos] = 1
                # print(1)
    target = target.view((target_coord.shape[0], joint_number, heatmap_size[1], heatmap_size[0]))
    target = target.reshape((target_coord.shape[0], joint_number, heatmap_size[1], heatmap_size[0]))
    return target, target_weight


def get_pose_net(joint_num):
    model = MobilePoseNet(joint_num)
    return model