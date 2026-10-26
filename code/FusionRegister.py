"""
models/FusionRegister.py  —  最终修复版

修复记录:
  [Fix-1] TDSR 输入从 (z, z) 改为 (irf, vif)
          论文 §III-C: depth features 来自 IR 特征, task features 来自 VI 特征.
          原版 depth_feat = task_feat = z.clone() 语义错误且无差异.
  [Fix-2] forward() 在训练模式下返回 (necessity, rv_hat) 元组作为第 8 个返回值,
          替代原来的 self.last_necessity / self.last_rv_hat 属性 hack.
  [Fix-3] 移除 self.last_necessity 和 self.last_rv_hat 两个 side-effect 属性,
          消除多线程/多 GPU 下的竞态条件隐患.
  [Fix-4] 推理模式 (inference=True) forward 路径统一, 返回单张 out.

训练时返回签名 (共 8 项):
  out, outputs_fil, mask, flow, s1_kernal_bias, nobias, feat_list, (necessity, rv_hat)

推理时返回签名:
  out
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
from models.layers import *
from models.covmlp import *
import random
import torchvision.transforms.functional as TF

# ── TDSR 模块 ──────────────────────────────────────────────
from train.tdsr_module import TDSR


# ─────────────────────────────────────────────────────────────
# 工具函数 (保持不变)
# ─────────────────────────────────────────────────────────────

def ma(mask, percent, edge):
    B, _, H, W = mask.shape
    binary_mask = torch.zeros_like(mask, dtype=torch.float32)
    if H <= 2 or W <= 2:
        return binary_mask
    non_edge_mask = mask[:, :, edge:-edge, edge:-edge]
    non_edge_values = non_edge_mask.flatten(start_dim=1)
    for b in range(B):
        values = non_edge_values[b]
        N = values.numel()
        if N == 0:
            continue
        sorted_values, _ = torch.sort(values)
        top_k = int(N * percent)
        bottom_k = int(N * percent)
        if top_k == 0 or bottom_k == 0:
            continue
        top_threshold = sorted_values[-top_k]
        bottom_threshold = sorted_values[bottom_k - 1]
        non_edge_binary = (
            (non_edge_mask[b] >= top_threshold) |
            (non_edge_mask[b] <= bottom_threshold)
        )
        binary_mask[b, :, edge:-edge, edge:-edge] = non_edge_binary.float()

    return binary_mask.to(mask.device)


class fusionnet(nn.Module):
    def __init__(self):
        super(fusionnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0),
        )
        self.act1 = nn.Sequential(nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0),
        )
        self.act2 = nn.Sequential(nn.Tanh())

    def forward(self, input):
        G11 = self.conv1(input)
        G11_1 = self.act1(G11)
        G21 = self.conv2(G11_1)
        return self.act2(G21)

def make_gate(self, necessity, force_necessity=None):
    """
    necessity: [B, 1, H, W], range usually [0, 1]
    return:
        gate: [B, 1, H, W]
    """

    # 强制消融模式必须确定，不走 gumbel 随机采样
    if force_necessity is not None:
        return torch.ones_like(necessity) * float(force_necessity)

    gate_mode = getattr(self, "gate_mode", "soft")
    gate_threshold = getattr(self, "gate_threshold", 0.5)

    if gate_mode == "soft":
        gate = necessity

    elif gate_mode in ["threshold", "hard"]:
        gate = (necessity > gate_threshold).float()

    elif gate_mode == "gumbel":
        tau = float(getattr(self, "gumbel_tau", 1.0))
        tau = max(tau, 0.3)  # 安全下限，避免过硬导致数值不稳定

        hard = getattr(self, "gumbel_hard", True)
        deterministic_eval = getattr(self, "gumbel_eval_deterministic", False)

        if (not self.training) and deterministic_eval:
            if hard:
                gate = (necessity > gate_threshold).float()
            else:
                gate = necessity
        else:
            # 关键：AMP 下必须转 float32，否则 p 接近 0/1 时 log 可能出 inf/nan
            p = necessity.float().clamp(1e-4, 1.0 - 1e-4)

            gate_logits_skip = torch.log(1.0 - p)
            gate_logits_reg = torch.log(p)
            gate_logits = torch.cat([gate_logits_skip, gate_logits_reg], dim=1)

            # gumbel_softmax 也用 float32 做
            y = F.gumbel_softmax(
                gate_logits,
                tau=tau,
                hard=hard,
                dim=1
            )

            gate = y[:, 1:2, :, :].to(dtype=necessity.dtype)

    elif gate_mode == "hybrid":
        # 可选：soft + threshold 混合
        gate = necessity
        hard_gate = (necessity > gate_threshold).float()
        alpha = getattr(self, "hybrid_alpha", 0.5)
        gate = alpha * gate + (1.0 - alpha) * hard_gate

    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    return gate

def filter_and_connect_mask(mask_tensor, min_area=100, morph_kernel=9):
    B, _, H, W = mask_tensor.shape
    output = []
    for i in range(B):
        mask_np = mask_tensor[i, 0].cpu().numpy().astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        filtered = np.zeros_like(closed)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label] = 255
        filtered_tensor = torch.from_numpy(filtered.astype(np.float32) / 255.0).unsqueeze(0)
        output.append(filtered_tensor)

    return torch.stack(output, dim=0).to(mask_tensor.device)


def flow_to_color(flow, sigma=1.0, magnitude_threshold=0.01, value_power=0.5):
    flow_x = flow[0, :, :]
    flow_y = flow[1, :, :]
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    angle = np.arctan2(flow_y, flow_x)
    mask = magnitude < magnitude_threshold
    magnitude[mask] = 0
    max_magnitude = np.max(magnitude)
    if max_magnitude > 1e-5:
        magnitude = magnitude / max_magnitude
    magnitude = np.clip(magnitude, 0, 1)
    magnitude_adjusted = np.power(magnitude, value_power)
    hue = ((angle + np.pi) / (2 * np.pi)) * 360
    saturation = magnitude_adjusted
    value = np.ones_like(magnitude)
    hsv = np.stack([hue, saturation * 255, value * 255], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[mask] = [255, 255, 255]
    return rgb


def save_tensor_as_image(tensor, i, name, fea=False):
    output_dir = './results/epoch_' + str(i) + '/'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{name}.png")
    if fea:
        if tensor.shape[1] == 2:
            flow = tensor[0].detach().cpu().numpy()
            rgb_image = flow_to_color(flow)
            H, W = tensor.shape[2], tensor.shape[3]
            figsize = (W / 100, H / 100)
            plt.figure(figsize=figsize)
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            fu1 = torch.mean(tensor, dim=1)
            image = fu1.squeeze().detach().cpu().numpy()
            plt.imshow(image, cmap='jet')
            plt.axis('off')
            plt.savefig(filename)
            plt.close()
    else:
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)
        save_image(normalized_tensor, filename)


# ─────────────────────────────────────────────────────────────
# 网络组件
# ─────────────────────────────────────────────────────────────

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


def random_light_transform(vi, ir, vi_ori, ir_ori, cb, cr, cls):
    B, C, H_in, W_in = vi.shape
    vi_output_list, ir_output_list, mask_output_list = [], [], []
    final_vi_ori_list, final_ir_ori_list, cb_list, cr_list = [], [], [], []
    blur_transform = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
    target_H, target_W = 256, 256
    crop_h_start = (H_in - target_H) // 2
    crop_w_start = (W_in - target_W) // 2

    for i in range(B):
        item_vi = vi[i].clone()
        item_ir = ir[i].clone()
        item_vi_ori = vi_ori[i].clone()
        item_ir_ori = ir_ori[i].clone()
        item_cb = cb[i].clone()
        item_cr = cr[i].clone()
        current_final_vi_output = None
        current_final_ir_output = None
        current_final_mask_output = None

        if cls[i] == 0:
            current_mask = torch.ones_like(item_vi)
            angle = random.uniform(-2, 2)
            trans_x = random.uniform(-2, 2)
            trans_y = random.uniform(-2, 2)
            scale = random.uniform(0.95, 1.08)
            transformed_ir = TF.affine(item_ir, angle=angle, translate=(trans_x, trans_y),
                                       scale=scale, shear=0,
                                       interpolation=TF.InterpolationMode.BILINEAR)
            transformed_mask = TF.affine(current_mask, angle=angle, translate=(trans_x, trans_y),
                                         scale=scale, shear=0,
                                         interpolation=TF.InterpolationMode.NEAREST)
            transformed_vi = item_vi
            if random.random() < 0.5:
                transform_choice = random.choice(['hflip', 'vflip', 'rot90'])
                tensors_to_transform = [transformed_vi, transformed_ir, transformed_mask,
                                        item_vi_ori, item_ir_ori, item_cb, item_cr]
                transformed_list = []
                if transform_choice == 'hflip':
                    for t in tensors_to_transform:
                        transformed_list.append(TF.hflip(t))
                elif transform_choice == 'vflip':
                    for t in tensors_to_transform:
                        transformed_list.append(TF.vflip(t))
                elif transform_choice == 'rot90':
                    k_rot = random.choice([1, 2, 3])
                    for t in tensors_to_transform:
                        transformed_list.append(torch.rot90(t, k_rot, dims=[1, 2]))
                if transformed_list:
                    (transformed_vi, transformed_ir, transformed_mask,
                     item_vi_ori, item_ir_ori, item_cb, item_cr) = transformed_list

            if H_in >= target_H and W_in >= target_W:
                s = slice(crop_h_start, crop_h_start + target_H)
                sw = slice(crop_w_start, crop_w_start + target_W)
                current_final_vi_output = transformed_vi[:, s, sw]
                current_final_ir_output = transformed_ir[:, s, sw]
                current_final_mask_output = transformed_mask[:, s, sw]
            else:
                current_final_vi_output = transformed_vi
                current_final_ir_output = transformed_ir
                current_final_mask_output = transformed_mask

            current_final_mask_output = (current_final_mask_output >= 0.5).float()
            shrinkage = random.randint(2, 3)
            if (current_final_mask_output.shape[1] > 2 * shrinkage and
                    current_final_mask_output.shape[2] > 2 * shrinkage):
                current_final_mask_output[:, :shrinkage, :] = 0
                current_final_mask_output[:, -shrinkage:, :] = 0
                current_final_mask_output[:, :, :shrinkage] = 0
                current_final_mask_output[:, :, -shrinkage:] = 0
            final_vi_ori_list.append(item_vi_ori)
            final_ir_ori_list.append(item_ir_ori)

        elif cls[i] == 1:
            k_rot = random.choice([1, 2, 3])
            vi_rotated = torch.rot90(item_vi, k_rot, dims=[1, 2])
            ir_rotated = torch.rot90(item_ir, k_rot, dims=[1, 2])
            item_vi_ori_rotated = torch.rot90(item_vi_ori, k_rot, dims=[1, 2])
            item_ir_ori_rotated = torch.rot90(item_ir_ori, k_rot, dims=[1, 2])
            item_cb = torch.rot90(item_cb, k_rot, dims=[1, 2])
            item_cr = torch.rot90(item_cr, k_rot, dims=[1, 2])
            current_mask_cls1 = torch.ones_like(item_vi)
            mask_rotated_cls1 = torch.rot90(current_mask_cls1, k_rot, dims=[1, 2])
            ir_blurred_rotated = blur_transform(ir_rotated)
            if H_in >= target_H and W_in >= target_W:
                s = slice(crop_h_start, crop_h_start + target_H)
                sw = slice(crop_w_start, crop_w_start + target_W)
                current_final_vi_output = vi_rotated[:, s, sw]
                current_final_ir_output = ir_blurred_rotated[:, s, sw]
                current_final_mask_output = mask_rotated_cls1[:, s, sw]
            else:
                current_final_vi_output = vi_rotated
                current_final_ir_output = ir_blurred_rotated
                current_final_mask_output = mask_rotated_cls1
            final_vi_ori_list.append(item_vi_ori_rotated)
            final_ir_ori_list.append(item_ir_ori_rotated)

        vi_output_list.append(current_final_vi_output)
        ir_output_list.append(current_final_ir_output)
        mask_output_list.append(current_final_mask_output)
        cb_list.append(item_cb)
        cr_list.append(item_cr)

    return (torch.stack(vi_output_list), torch.stack(ir_output_list),
            torch.stack(final_vi_ori_list), torch.stack(final_ir_ori_list),
            torch.stack(cb_list), torch.stack(cr_list),
            torch.stack(mask_output_list))


def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0],
                        [1.403, -0.714, 0.0],
                        [0.0, -0.344, 1.773]]).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    return out.clamp(0, 1.0)


def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb.squeeze(0), Cr.squeeze(0)


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()
        self.layers = nn.Sequential(*[ResBlock(out_channel) for _ in range(num_res)])

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()
        self.layers = nn.Sequential(*[ResBlock(channel) for _ in range(num_res)])

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))


class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        return self.conv(torch.cat([x, self.main(x)], dim=1))


class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return x1 + self.merge(x1 * x2)


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f'The spatial sizes of input ({x.size()[-2:]}) and '
            f'flow ({flow.size()[1:3]}) are not the same.'
        )
    _, _, h, w = x.size()
    device = flow.device
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)
    grid.requires_grad = False
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3).type(x.type())
    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


# ─────────────────────────────────────────────────────────────
# 主模型 FusionRegister
# ─────────────────────────────────────────────────────────────

class FusionRegister(nn.Module):
    """
    FusionRegister —— 带 TDSR 选择性配准的红外/可见光融合网络.

    训练 forward 返回 8 个值:
        out, outputs_fil, mask, flow, s1_kernal_bias, nobias, feat_list, (necessity, rv_hat)

        necessity : [B, 1, H, W]  ∈ [0,1]  — 逐像素配准必要性图
        rv_hat    : [B, 1, H, W]  ∈ [0,1]  — 预测的 Registration Value (供损失计算)

    推理 forward 返回 1 个值:
        out  : [B, C, H, W]  — 融合图像
    """

    def __init__(
            self,
            inp_channels=3,
            out_channels=3,
            dim=16,
            num_blocks=[12, 12],
            kernel_size=7,
            inference=False,
            tdsr_mid_ch=None,
            gate_mode="soft",
            gate_threshold=0.5,
            gumbel_tau=1.0,
            gumbel_hard=True,
            gumbel_eval_deterministic=False,
            hybrid_alpha=0.5,
    ):
        super(FusionRegister, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)
        self.gate_mode = gate_mode
        self.gate_threshold = gate_threshold
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard
        self.gumbel_eval_deterministic = gumbel_eval_deterministic
        self.hybrid_alpha = hybrid_alpha

        if not inference:
            BasicConv = BasicConv_do
            DwConv = lambda *args, **kwargs: Conv_dw(*args, simam=False, **kwargs)
        else:
            BasicConv = BasicConv_do_eval
            DwConv = lambda *args, **kwargs: Conv_dw(*args, simam=False, **kwargs)

        base_channel = dim  # = 16

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_blocks[0]),
            EBlock(base_channel * 2, num_blocks[0]),
            EBlock(base_channel, num_blocks[1]),   # for vif
            EBlock(base_channel, num_blocks[1]),   # for irf
        ])

        self.feat_extract = nn.ModuleList([
            DwConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            DwConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            DwConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            DwConv(3, base_channel, kernel_size=3, relu=True, stride=1),   # vi → vif
            DwConv(1, base_channel, kernel_size=3, relu=True, stride=1),   # ir → irf
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 2, num_blocks[0]),
            DBlock(base_channel, num_blocks[0])
        ])

        self.Convs = nn.ModuleList([
            DwConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 3, base_channel * 1, BasicConv=BasicConv),
            AFF(base_channel * 3, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)
        self.att_v = CMWMLP_block(base_channel, base_channel, use_corr=True, att="chan")
        self.att_i = CMWMLP_block(base_channel, base_channel, use_corr=True, att="space")

        self.KernelPredictFlow = nn.ModuleList([
            DwConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
            DwConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])
        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([
            DwConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            DwConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()

        self.KernelOutBias = nn.ModuleList([
            DwConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.up = Upsample(base_channel * 2)


        _tdsr_mid = tdsr_mid_ch if tdsr_mid_ch is not None else base_channel
        self.use_tdsr = True
        self.tdsr = TDSR(
            depth_in_ch=base_channel,   # 16  ← IR 编码特征通道数（固定）
            task_in_ch=base_channel,    # 16  ← VI 编码特征通道数（固定）
            mid_ch=_tdsr_mid,           # 可配置: 16(轻量) 或 64(完整)
        )

    # ──────────────────────────────────────────────────────────
    def forward(self, x, vi, ir, force_necessity=None):

        x_2 = F.interpolate(x, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        mask = []
        flow = []
        outputs_fil = []

        # ── 编码器 ────────────────────────────────────────────
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        res2 = self.AFFs[1](z12, res2)
        res1 = self.AFFs[0](res1, z21)

        # ── 模态特征 (irf=IR 深度代理, vif=VI 任务特征) ────────
        # [Fix-1] vif 和 irf 语义明确分离, 与论文 §III-C 对应.
        vif = self.Encoder[2](self.feat_extract[3](vi))   # [B, 16, H, W]
        irf = self.Encoder[3](self.feat_extract[4](ir))   # [B, 16, H, W]

        # ── 粗配准 (尺度 2, H/2×W/2) ───────────────────────────
        z = self.Decoder[0](res2)
        s2_kernal_flow = self.KernelPredictFlow[0](z)
        s2_kernal_flowmask = self.sigmoid(self.KernelPredictFlowMask[0](z))
        flow.append(s2_kernal_flow)

        zx2 = torch.cat([z, x_2], 1)
        s2_feat0, x_2_0 = torch.split(
            flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)),
            self.dim * 2, dim=1)
        s2_feat1, x_2_1 = torch.split(
            flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)),
            self.dim * 2, dim=1)

        if not self.inference:
            x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1 - s2_kernal_flowmask)
            mask.append(s2_kernal_flowmask)
            outputs_fil.append(x_2)

        z = torch.cat([
            z,
            s2_feat0 * s2_kernal_flowmask + s2_feat1 * (1 - s2_kernal_flowmask)
        ], 1)

        # ── 解码 & 上采样 ─────────────────────────────────────
        z = self.feat_extract[2](z)
        z = self.up(z)                          # [B, 16, H, W]
        z = torch.cat([z, res1], dim=1)         # [B, 32, H, W]
        z = self.Convs[0](z)                    # [B, 16, H, W]
        z = self.Decoder[1](z)                  # [B, 16, H, W]

        # ── TDSR 选择性精细配准 ──────────────────────────────
        # [Fix-1] 使用语义正确的输入:
        #   - depth_feat = irf : IR 编码特征代表深度/错位严重性
        #   - task_feat  = vif : VI 编码特征代表任务语义重要性
        # [Fix-2] 直接返回 necessity 和 rv_hat, 不再写 self.last_* 属性
        # [R0-Fix] base_flowmask 与 necessity 严格分离:
        #   旧版: s1_kernal_flowmask = sigmoid(KPF) * necessity
        #         → necessity=0 时 flowmask=0 → x = x_1（反向 warp，不是原始特征）
        #   新版: base_flowmask = sigmoid(KPF)（全配准掩码，不含 necessity）
        #         x_reg = x_0 * base_flowmask + x_1 * (1 - base_flowmask)（完整配准结果）
        #         x = x_reg * necessity + x_original * (1 - necessity)
        #         → necessity=0 → 完全保留原始未变形特征（论文公式3的正确实现）
        #         → necessity=1 → 完整配准结果
        base_flowmask = self.sigmoid(self.KernelPredictFlowMask[1](z))

        if force_necessity is not None:
            force_value = float(force_necessity)
            necessity = torch.full_like(base_flowmask, force_value)
            rv_hat = necessity
            logits = torch.logit(necessity.clamp(1e-4, 1.0 - 1e-4))
        elif self.use_tdsr and not self.inference:
            necessity, rv_hat, logits = self.tdsr(irf, vif)
        elif self.use_tdsr and self.inference:
            necessity, rv_hat, logits = self.tdsr(irf, vif)
        else:
            necessity = torch.ones_like(base_flowmask)
            rv_hat = necessity
            logits = torch.logit(necessity.clamp(1e-4, 1.0 - 1e-4))

        gate = make_gate(self, necessity, force_necessity=force_necessity)

        s1_kernal_flowmask = base_flowmask * gate

        s1_kernal_flow = (
            self.KernelPredictFlow[1](z) + self.flowup(s2_kernal_flow) * 2
        )
        mask.append(s1_kernal_flowmask)
        flow.append(s1_kernal_flow)

        # [R0-Fix] 保存原始未变形特征（在任何 warp 之前）
        x_original = x.clone()
        z_original = z.clone()

        zx = torch.cat([z, x], 1)
        s1_feat0, x_0 = torch.split(
            flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)),
            self.dim, dim=1)
        s1_feat1, x_1 = torch.split(
            flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)),
            self.dim, dim=1)


        x_reg = x_0 * base_flowmask + x_1 * (1 - base_flowmask)

        x = x_reg * gate + x_original * (1 - gate)

        z_1 = z
        # [R0-Fix] fea 与 x 保持一致：完整配准的特征 vs 原始 z 特征
        fea_reg = s1_feat0 * base_flowmask + s1_feat1 * (1 - base_flowmask)
        fea = fea_reg * gate + z_original * (1 - gate)
        z_v = self.att_v(fea, vif)
        z_i = self.att_i(fea, irf)
        z = z_i + z_v + fea
        z_6 = z

        s1_kernal_bias = self.KernelOutBias[0](z)
        out_norm = (x - x.max()) / (x.max() - x.min() + 1e-6)
        nobias = out_norm
        out = out_norm + s1_kernal_bias

        # ── 返回 ───────────────────────────────────────────────
        if not self.inference:
            outputs_fil.append(x)
            # [Fix-2] 第 8 个返回值为 (necessity, rv_hat, logits) 元组
            return (out, outputs_fil[::-1], mask, flow,
                    s1_kernal_bias, nobias,
                    [z_1, z_v, z_i, z_6],
                    (necessity, rv_hat, logits, gate))
        else:
            return out


# ─────────────────────────────────────────────────────────────
# 参数统计工具
# ─────────────────────────────────────────────────────────────

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


def calculate_parameters(model):
    total_params = total_size = 0
    for param in model.parameters():
        n = param.numel()
        total_params += n
        total_size += param.element_size() * n
    return total_params, total_size


# ─────────────────────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FusionRegister().cuda()
    x = torch.randn(1, 3, 480, 640).cuda()   # vi / main input
    z = torch.randn(1, 1, 480, 640).cuda()   # ir

    # 训练模式测试
    model.train()
    model.inference = False

    outputs = model(x, x, z)

    out, outputs_fil, masks, flows, bias, nobias, feats, tdsr_out = outputs

    necessity = tdsr_out[0]
    rv_hat = tdsr_out[1]
    _logits = tdsr_out[2] if len(tdsr_out) > 2 else None
    _gate = tdsr_out[3] if len(tdsr_out) > 3 else None

    print(f"[Train] out: {out.shape}")
    print(f"[Train] necessity: {necessity.shape}, rv_hat: {rv_hat.shape}")

    if _logits is not None:
        print(f"[Train] logits: {_logits.shape}")
    if _gate is not None:
        print(f"[Train] gate: {_gate.shape}")
        print(
            f"[Train] gate mean={_gate.mean().item():.4f}, "
            f"skip={(1.0 - _gate).mean().item() * 100:.1f}%"
        )

    print(
        f"[Train] N mean={necessity.mean().item():.4f}, "
        f"std={necessity.std().item():.4f}, "
        f"skip={(necessity < 0.5).float().mean().item() * 100:.1f}%"
    )

    # 推理模式测试
    model.eval()
    model.inference = True
    with torch.no_grad():
        out_inf = model(x, x, z)

    print(f"[Infer] out: {out_inf.shape}")

    # 恢复，避免后续误用
    model.inference = False

    num_params, storage_size = calculate_parameters(model)
    print(f"总参数数量: {num_params:,}")
    print(f"模型存储大小: {storage_size / (1024 ** 2):.2f} MB")