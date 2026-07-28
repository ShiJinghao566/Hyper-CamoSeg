import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip import get_tokenizer

from prompts import get_prompt_template_by_name


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        # c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        c2wh = dict([ (192, 112), (384, 56), (768, 28), (1536, 14)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation


class SpatialSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        if in_channels % num_heads != 0:
            valid_heads = [head for head in range(num_heads, 0, -1) if in_channels % head == 0]
            num_heads = valid_heads[0]

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return identity + x


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_connection_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8):
        super(UpsampleBlock, self).__init__()

        in_channels = in_channels + skip_connection_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.attention_layer = SpatialSelfAttentionBlock(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip_connection=None):
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.attention_layer(x)
        x = self.conv2(x)

        return x

class CascadedSubDecoderBinary(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 scale_factor,
                 interpolation_mode='bilinear'):
        super(CascadedSubDecoderBinary, self).__init__()

        self.output_map_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_distance_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_boundary_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=interpolation_mode, align_corners=True)
        self.count = 0

    def forward(self, x):
        map = self.output_map_conv(x) # B, 1, H, W
        distance = self.output_distance_conv(x) * torch.sigmoid(map)
        boundary = self.output_boundary_conv(x) * torch.sigmoid(distance)

        boundary = self.upsample(boundary)
        distance = self.upsample(distance) + torch.sigmoid(boundary)
        map = self.upsample(map) + torch.sigmoid(distance)

        return map, distance, boundary

class MFMSNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 cnn_backbone='resnet50'):
        super(MFMSNet, self).__init__()


        self.num_classes = num_classes
        if cnn_backbone not in ['resnet50', 'res2net50_v1b_26w_4s', 'resnest50']:
            print('Wrong CNN Backbone model')
            sys.exit()

        if cnn_backbone in ['resnet50', 'res2net50_v1b_26w_4s', 'resnest50']:
            self.in_channels = 2048
            self.skip_channel_list = [1024, 512, 256, 64]
            self.decoder_channel_list = [256, 128, 64, 32]
        else:
            print("Wrong CNN Backbone...")
            sys.exit()

        self.feature_encoding.fc = nn.Identity()

        self.skip_channel_down_list = [64, 64, 64, 64]

        self.skip_connection1 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[0], self.skip_channel_down_list[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[0]), nn.ReLU(inplace=True))
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[1], self.skip_channel_down_list[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[1]), nn.ReLU(inplace=True))
        self.skip_connection3 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[2], self.skip_channel_down_list[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[2]), nn.ReLU(inplace=True))
        self.skip_connection4 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[3], self.skip_channel_down_list[3], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[3]), nn.ReLU(inplace=True))

        self.decoder_stage1 = UpsampleBlock(self.in_channels, self.decoder_channel_list[0], self.skip_channel_down_list[0], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage2 = UpsampleBlock(self.decoder_channel_list[0], self.decoder_channel_list[1], self.skip_channel_down_list[1], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage3 = UpsampleBlock(self.decoder_channel_list[1], self.decoder_channel_list[2], self.skip_channel_down_list[2], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage4 = UpsampleBlock(self.decoder_channel_list[2], self.decoder_channel_list[3], self.skip_channel_down_list[3], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)

        # Sub-Decoder
        self.sub_decoder_stage1 = CascadedSubDecoderBinary(self.decoder_channel_list[0], num_classes, scale_factor=16)
        self.sub_decoder_stage2 = CascadedSubDecoderBinary(self.decoder_channel_list[1], num_classes, scale_factor=8)
        self.sub_decoder_stage3 = CascadedSubDecoderBinary(self.decoder_channel_list[2], num_classes, scale_factor=4)
        self.sub_decoder_stage4 = CascadedSubDecoderBinary(self.decoder_channel_list[3], num_classes, scale_factor=2)

    def forward(self, x, mode='train'):
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)
        _, _, H, W = x.shape

        features, x = self.feature_encoding.forward_feature(x, out_block_stage=4)

        x1 = self.decoder_stage1(x, self.skip_connection1(features[0]))
        x2 = self.decoder_stage2(x1, self.skip_connection2(features[1]))
        x3 = self.decoder_stage3(x2, self.skip_connection3(features[2]))
        x4 = self.decoder_stage4(x3, self.skip_connection4(features[3]))
        if mode == 'train':
            map_output1, distance_output1, boundary_output1 = self.sub_decoder_stage1(x1)
            map_output2, distance_output2, boundary_output2 = self.sub_decoder_stage2(x2)
            map_output3, distance_output3, boundary_output3 = self.sub_decoder_stage3(x3)
            map_output4, distance_output4, boundary_output4 = self.sub_decoder_stage4(x4)


            return [map_output1, distance_output1, boundary_output1], \
                   [map_output2, distance_output2, boundary_output2], \
                   [map_output3, distance_output3, boundary_output3], \
                   [map_output4, distance_output4, boundary_output4]
        else:
            map, _, _ = self.sub_decoder_stage4(x4)

            return map

    def _calculate_criterion(self, y_pred, y_true):
        loss = self.structure_loss(y_pred, y_true)

        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

class ViTCLIP(nn.Module):
    def __init__(
            self,
            model_name="ViT-L-14",
            pretrained="laion",
            local_model_path=None,
            template_set="camoprompts",
            input_size=384  # 指定默认输入尺寸，需与实际输入一致
    ):
        super().__init__()

        # 加载ViT-L/14模型
        if local_model_path:
            self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
                model_name, pretrained=local_model_path
            )
        else:
            self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

        self.mean = open_clip.OPENAI_DATASET_MEAN
        self.std = open_clip.OPENAI_DATASET_STD
        self.text_tokenizer = get_tokenizer(model_name)
        self.template_set = get_prompt_template_by_name(template_set)
        self.model_type = "vit"
        self.feat_chans = [256, 512, 1024, 1024]  # 对应不同阶段的特征通道数
        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.out_strides = {"stem": 16, "res2": 16, "res3": 16, "res4": 16, "res5": 16, "emb": -1}
        self.out_chans = {
            "stem": self.feat_chans[0],
            "res2": self.feat_chans[1],
            "res3": self.feat_chans[2],
            "res4": self.feat_chans[3],
            "res5": self.feat_chans[3],
            "emb": self.dim_latent,
        }
        self.input_size = input_size  # 输入图像尺寸
        self.patch_size = 14  # ViT-L/14的patch大小为14x14

    def output_shape(self):
        return {
            name: dict(channels=self.out_chans[name], stride=self.out_strides[name])
            for name in ["stem", "res2", "res3", "res4", "res5", "emb"]
        }

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device

    @torch.no_grad()
    def get_text_embs(self, text_list, normalize=True):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        text_tokens = self.text_tokenizer(text_list).to(self.device)

        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # 从eot embedding获取特征
        text_embs = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection

        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)
        return text_embs

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称使用模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        text_embs = []
        for text in text_list:
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(self.device)

            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)

            x = x + self.clip_model.positional_embedding.to(cast_dtype)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)

            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768

        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768

    def visual_feats_to_embs(self, x, normalize: bool = True):
        """将图像特征转换为图像嵌入向量"""
        self.eval()

        x = self.clip_model.visual(x)
        return F.normalize(x, dim=-1) if normalize else x

    @torch.no_grad()
    def get_visual_feats(self, x):
        """提取ViT-L/14的多阶段视觉特征（修复位置嵌入适配问题）"""
        self.eval()
        out = {}
        batch_size = x.shape[0]

        # 调整输入图像尺寸（保持为patch_size的整数倍）
        h, w = x.shape[2], x.shape[3]
        if h != self.input_size or w != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False
            )

        # 初始卷积和分块
        x = self.clip_model.visual.conv1(x)  # 初始卷积
        _, c, h, w = x.shape
        x = x.reshape(batch_size, c, -1)
        x = x.permute(0, 2, 1)  # [batch, width*height, c]

        # 类嵌入（添加到每个样本）
        class_emb = self.clip_model.visual.class_embedding.to(x.dtype)
        class_emb = class_emb.expand(batch_size, -1, -1)  # 扩展为批次大小

        # 动态适配位置嵌入
        pos_emb = self.clip_model.visual.positional_embedding.to(x.dtype)
        num_patches = (self.input_size // self.patch_size) ** 2  # 计算理论patch数量

        # 处理位置嵌入形状不匹配的情况
        if pos_emb.shape[1] != num_patches + 1:  # +1为类嵌入位置
            # 分离类嵌入和patch嵌入
            cls_pos_emb = pos_emb[:, 0:1, :]
            patch_pos_emb = pos_emb[:, 1:, :]

            # 调整patch嵌入尺寸（假设原始为正方形）
            orig_patch_num = int(np.sqrt(patch_pos_emb.shape[1]))
            new_patch_num = int(np.sqrt(num_patches))
            patch_pos_emb = patch_pos_emb.reshape(1, orig_patch_num, orig_patch_num, -1)
            patch_pos_emb = patch_pos_emb.permute(0, 3, 1, 2)  # [1, dim, H, W]

            # 插值调整尺寸
            patch_pos_emb = F.interpolate(
                patch_pos_emb,
                size=(new_patch_num, new_patch_num),
                mode="bilinear",
                align_corners=False
            )

            # 恢复为序列形式
            patch_pos_emb = patch_pos_emb.permute(0, 2, 3, 1)  # [1, H, W, dim]
            patch_pos_emb = patch_pos_emb.reshape(1, -1, patch_pos_emb.shape[-1])  # [1, num_patches, dim]

            # 重组位置嵌入
            pos_emb = torch.cat([cls_pos_emb, patch_pos_emb], dim=1)

        # 组合类嵌入和图像patch
        x = torch.cat([class_emb, x], dim=1)
        x = x + pos_emb  # 现在位置嵌入形状已适配
        x = self.clip_model.visual.ln_pre(x)

        # 提取不同阶段的特征（保持原有逻辑）
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            x = block(x)
            if i == 0:
                num_patches = x.shape[1] - 1  # 减去类嵌入
                feat_size = int(num_patches ** 0.5)
                out["stem"] = x[:, 1:].reshape(batch_size, feat_size, feat_size, self.feat_chans[0]).permute(0, 3, 1,
                                                                                                             2).contiguous()
            elif i == 3:
                out["res2"] = x[:, 1:].reshape(batch_size, feat_size // 2, feat_size // 2, self.feat_chans[1]).permute(
                    0, 3, 1, 2).contiguous()
            elif i == 7:
                out["res3"] = x[:, 1:].reshape(batch_size, feat_size // 4, feat_size // 4, self.feat_chans[2]).permute(
                    0, 3, 1, 2).contiguous()
            elif i == 11:
                out["res4"] = x[:, 1:].reshape(batch_size, feat_size // 8, feat_size // 8, self.feat_chans[3]).permute(
                    0, 3, 1, 2).contiguous()

        out["res5"] = out["res4"].contiguous()
        x = self.clip_model.visual.ln_post(x[:, 0, :])

        out["clip_vis_dense"] = x.contiguous()
        return out
