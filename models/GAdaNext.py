import torch
import torch.nn as nn
import torch.nn.functional as F


from .convnext import ConvNeXt, Block, LayerNorm, trunc_normal_

class GAdaNext(nn.Module):
    def __init__(self, input_channels=6, out_channels=1, depths=(2, 4, 2), dims=(128, 256, 512),
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, head_init_scale=1.):
        super(GAdaNext, self).__init__()

        super().__init__()

        self.rgb_feat = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                      LayerNorm(16, eps=1e-6, data_format="channels_first"),
                                      nn.GELU(),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                      LayerNorm(16, eps=1e-6, data_format="channels_first"),
                                      nn.GELU(),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.depth_feat = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2),
                                        LayerNorm(16, eps=1e-6, data_format="channels_first"),
                                        nn.GELU(),
                                        nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                        LayerNorm(16, eps=1e-6, data_format="channels_first"),
                                        nn.GELU(),
                                        nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.gripper_feat = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1),
                                          LayerNorm(16, eps=1e-6, data_format="channels_first"),
                                          nn.GELU(),
                                          nn.Conv2d(16, 16, kernel_size=3, padding=1))

        self.n_depths = len(depths)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(48, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], out_channels)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.n_depths):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, scene_inp, gripper_inp):
        B, Nc, H, W = scene_inp.shape
        _, Ns, = gripper_inp.shape[:2]
        rgb, depth = scene_inp[:, :3], scene_inp[:, 3:]
        rgb_feat = self.rgb_feat(rgb)
        gripper_feat = self.gripper_feat(gripper_inp.reshape(B*Ns, 1, H, W))
        depth_feat = self.depth_feat(depth)
        rgbd_feat = torch.cat((rgb_feat, depth_feat), dim=1)
        #depth_gripper = torch.cat((depth.unsqueeze(1).repeat(1, Ns, 1, 1, 1), gripper_inp.unsqueeze(2)), dim=2)
        #depth_gripper_feat = self.depth_gripper_feat(depth_gripper.reshape(B*Ns, 2, H, W))

        x_feat = torch.cat((rgbd_feat.unsqueeze(1).repeat(1, Ns, 1, 1, 1).reshape(B*Ns, 32, H, W), gripper_feat), dim=1)
                             # [B*Ns, 64, H, W]
        # x_in = torch.cat((scene_inp.unsqueeze(1).repeat(1, Ns, 1, 1, 1), gripper_inp.unsqueeze(2)), dim=2)
        x_out = self.forward_features(x_feat)
        x_out = self.head(x_out)

        x_out = x_out.squeeze(-1).reshape(B, Ns)

        return x_out