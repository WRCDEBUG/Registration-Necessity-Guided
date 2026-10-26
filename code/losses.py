"""
train/losses.py — 损失函数 + proxy delta
稳定模块，正常情况下不需要修改。
"""

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Proxy ΔL_task（不依赖 YOLO，速度快）
# ══════════════════════════════════════════════════════════════

def compute_spatial_delta_loss(
    ir: torch.Tensor,
    vi: torch.Tensor,
    fused: torch.Tensor,
    patch_size: int = 32,
    proxy_mode: str = "multiplicative",
    beta_ir: float = 0.5,
    use_consistency: bool = True,
) -> torch.Tensor:
    """
    Proxy ΔL_task / registration-necessity target.

    Default matches the manuscript:
        E_IR * E_VIS * (0.5 + 0.5 C) + beta * P_IR

    Ablation variants:
        proxy_mode="additive": replace E_IR * E_VIS with 0.5(E_IR + E_VIS)
        use_consistency=False: remove the C modulator
        beta_ir=0: remove the infrared saliency prior
    """
    B, _, H, W = ir.shape
    device = ir.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=ir.dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=ir.dtype, device=device).view(1, 1, 3, 3)

    gx = F.conv2d(ir, sobel_x, padding=1)
    gy = F.conv2d(ir, sobel_y, padding=1)
    depth_grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    vi_gray = 0.299 * vi[:, 0:1] + 0.587 * vi[:, 1:2] + 0.114 * vi[:, 2:3]
    ex = F.conv2d(vi_gray, sobel_x, padding=1)
    ey = F.conv2d(vi_gray, sobel_y, padding=1)
    edge_density = torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    if fused.shape[1] == 1:
        fused_gray = fused
    else:
        fused_gray = (0.299 * fused[:, 0:1] + 0.587 * fused[:, 1:2]
                      + 0.114 * fused[:, 2:3])
    if use_consistency:
        pixel_diff = (vi_gray - fused_gray).abs()
        diff_max = pixel_diff.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        consistency = pixel_diff / diff_max
        modulator = 0.5 + 0.5 * consistency
    else:
        modulator = torch.ones_like(depth_grad)

    if proxy_mode == "multiplicative":
        delta = depth_grad * edge_density * modulator
    elif proxy_mode == "additive":
        delta = 0.5 * (depth_grad + edge_density) * modulator
    else:
        raise ValueError(f"Unsupported proxy_mode: {proxy_mode}")

    if beta_ir > 0:
        ir_min = ir.amin(dim=[2, 3], keepdim=True)
        ir_max = ir.amax(dim=[2, 3], keepdim=True)
        ir_prior = (ir - ir_min) / (ir_max - ir_min + 1e-6)
        if ir_prior.shape[1] != 1:
            ir_prior = ir_prior.mean(dim=1, keepdim=True)
        delta = delta + float(beta_ir) * ir_prior

    delta = F.avg_pool2d(delta, kernel_size=patch_size, stride=patch_size // 2,
                         padding=patch_size // 4)
    delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)

    d_min = delta.amin(dim=[2, 3], keepdim=True)
    d_max = delta.amax(dim=[2, 3], keepdim=True)
    scale = d_max - d_min
    delta_norm = (delta - d_min) / (scale + 1e-6)
    delta = torch.where(scale > 1e-4, delta_norm, torch.zeros_like(delta_norm))

    return delta.detach()


def compute_measured_target(
    geom_mag: torch.Tensor,
    appearance: torch.Tensor,
    geom_scale: float = 16.0,
    appearance_floor: float = 0.5,
) -> torch.Tensor:
    """Measurable registration-utility target.

        N_gt(x) = normalize( |u_gt(x)| * (floor + (1 - floor) * D(x)) )

    `|u_gt|` is the ground-truth correction displacement produced by the known
    synthetic perturbation (geometric demand) and `D` is the appearance
    importance map (edges x infrared saliency).  Multiplying the two keeps only
    the locations where registration can actually change the representation
    *and* the change matters for perception; unlike the appearance-only proxy,
    the target is tied to an independently measurable quantity.
    """
    geom = (geom_mag / float(geom_scale)).clamp(0.0, 1.0)
    target = geom * (float(appearance_floor)
                     + (1.0 - float(appearance_floor)) * appearance)
    t_min = target.amin(dim=[2, 3], keepdim=True)
    t_max = target.amax(dim=[2, 3], keepdim=True)
    scale = t_max - t_min
    out = (target - t_min) / (scale + 1e-6)
    return torch.where(scale > 1e-4, out, torch.zeros_like(out)).detach()


# ══════════════════════════════════════════════════════════════
# RV 监督损失（Regression + Ranking + Consistency）
# ══════════════════════════════════════════════════════════════
def compute_rv_loss(
    rv_hat: torch.Tensor,
    delta: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.3,
    margin: float = 0.05,
) -> tuple:
    """
    delta 可以来自 proxy 或真实 ΔL_task，接口相同。
    Returns: (total_loss, dict of sub-losses)
    """
    loss_reg = F.mse_loss(rv_hat, delta)

    B = rv_hat.shape[0]
    flat_rv = rv_hat.view(B, -1)
    flat_gt = delta.view(B, -1)

    n_pairs = 256
    idx_i = torch.randint(0, flat_rv.shape[1], (B, n_pairs), device=rv_hat.device)
    idx_j = torch.randint(0, flat_rv.shape[1], (B, n_pairs), device=rv_hat.device)

    rv_i = flat_rv.gather(1, idx_i)
    rv_j = flat_rv.gather(1, idx_j)
    gt_i = flat_gt.gather(1, idx_i)
    gt_j = flat_gt.gather(1, idx_j)

    gt_diff = gt_i - gt_j
    valid = gt_diff.abs() > 1e-3
    sign = gt_diff.sign()

    rank_raw = F.relu(margin - sign * (rv_i - rv_j))
    if valid.any():
        loss_rank = rank_raw[valid].mean()
    else:
        loss_rank = torch.zeros((), device=rv_hat.device, dtype=rv_hat.dtype)

    H, W = rv_hat.shape[2], rv_hat.shape[3]
    rv_down = F.avg_pool2d(rv_hat, 2, 2)
    rv_up = F.interpolate(rv_down, size=(H, W), mode="bilinear", align_corners=False)
    loss_cons = F.l1_loss(rv_hat, rv_up)

    total = loss_reg + alpha * loss_rank + beta * loss_cons

    return total, {
        "regression": loss_reg.item(),
        "ranking": loss_rank.item(),
        "consistency": loss_cons.item(),
    }

# ══════════════════════════════════════════════════════════════
# Sparsity & Bimodality 正则（Stage 2/3）
# ══════════════════════════════════════════════════════════════

def compute_sparsity_loss(necessity, target_sparsity=0.40,
                          temperature=0.25, logits=None):
    """
        约束 skip_ratio，即 necessity < 0.5 的比例。
        注意：这里的 sparsity 不是 selected_ratio，而是不执行精细配准的区域比例。
        """
    if logits is not None:
        soft_below = torch.sigmoid(-logits / temperature)
    else:
        soft_below = torch.sigmoid((0.5 - necessity) / temperature)
    current_sparsity = soft_below.mean()
    diff = current_sparsity - target_sparsity
    # 不对称：低于目标时惩罚系数 ×3，高于目标时系数 ×1
    weight = torch.where(diff < 0,
                         torch.tensor(3.0, device=necessity.device),
                         torch.tensor(1.0, device=necessity.device))
    return (weight * diff ** 2)


def compute_bimodality_loss(necessity: torch.Tensor,
                            logits: torch.Tensor = None,
                            margin: float = 1.0) -> torch.Tensor:
    """
    双峰化损失（鼓励 necessity 趋向 0 或 1）。

    当 logits 可用时，直接在 logits 空间操作：
      惩罚 |logits| < margin 的部分（即把 logit 推离 0，从而远离 0.5）。
      N*(1-N) 在 N=0.5 处梯度为零（极大值），推力最弱；logits 版在 logit=0 处
      梯度最大，正好弥补这一缺陷，与 Gumbel tau 退火协同驱动双峰分化。
    当 logits 不可用时退化为原始 N*(1-N)。
    """
    if logits is not None:
        return F.relu(margin - logits.abs()).mean()
    return (necessity * (1.0 - necessity)).mean()

def balanced_necessity_target(
    delta: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    把 delta 转成平衡的 soft necessity target。
    避免直接用 delta 导致大面积低值背景把 necessity 压塌。
    """
    mean = delta.mean(dim=[2, 3], keepdim=True)
    std = delta.std(dim=[2, 3], keepdim=True).clamp_min(eps)
    target = torch.sigmoid((delta - mean) / (std * temperature))
    return target.detach()


def compute_necessity_alignment_loss(
    necessity: torch.Tensor,
    delta: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    轻量约束 necessity 与 task/proxy delta 保持排序/强弱一致。
    主要用于 Stage 3，防止 sparsity/bimodal 正则把 necessity 从 RV 语义中拉偏。
    """
    target = balanced_necessity_target(delta, temperature=temperature)
    return F.mse_loss(necessity, target)
