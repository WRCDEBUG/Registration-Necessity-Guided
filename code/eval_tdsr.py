import argparse
import sys
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ── 项目路径 ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random
import torchvision.transforms.functional as TF

from train.FusionRegister import FusionRegister
from train.tdsr_module import count_parameters


# ══════════════════════════════════════════════════════════
# 0. 错位增强（与训练脚本完全一致，保证 train/eval 分布对齐）
# ══════════════════════════════════════════════════════════
def apply_misalignment(
    ir_tensor: torch.Tensor,
    angle_range: float = 3.0,
    translate_range: float = 8.0,
    scale_range: float = 0.03,
    seed: int = None,
) -> torch.Tensor:

    if seed is not None:
        rng_state = random.getstate()
        random.seed(seed)

    angle     = random.uniform(-angle_range, angle_range)
    translate = [random.uniform(-translate_range, translate_range),
                 random.uniform(-translate_range, translate_range)]
    scale     = random.uniform(1.0 - scale_range, 1.0 + scale_range)

    if seed is not None:
        random.setstate(rng_state)

    return TF.affine(
        ir_tensor,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0.0,
    )


# ══════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════
class FusionTestDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, img_h=480, img_w=640, split="test",
                 misalign_prob: float = 1.0, use_fixed_seed: bool = True):
        self.root = Path(root_dir)
        self.misalign_prob  = misalign_prob
        self.use_fixed_seed = use_fixed_seed
        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ])

        ir_dir  = self.root / "infrared" / split
        vis_dir = self.root / "visible"  / split

        # 兼容 .jpg / .png
        self.ir_files = sorted(
            glob.glob(str(ir_dir / "*.jpg")) +
            glob.glob(str(ir_dir / "*.png"))
        )
        self.vis_files = sorted(
            glob.glob(str(vis_dir / "*.jpg")) +
            glob.glob(str(vis_dir / "*.png"))
        )

        n = min(len(self.ir_files), len(self.vis_files))
        if n == 0:
            raise FileNotFoundError(
                f"测试集图像未找到！检查路径: {ir_dir} / {vis_dir}\n"
                f"如果数据集没有单独的 test 子目录，请用 --split train"
            )
        print(f"[TestDataset] split={split}, 找到 {n} 对图像")
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ir  = Image.open(self.ir_files[idx]).convert("L")
        vis = Image.open(self.vis_files[idx]).convert("RGB")
        name = Path(self.ir_files[idx]).stem

        ir_tensor  = self.transform(ir)   # [1,H,W]
        vis_tensor = self.transform(vis)  # [3,H,W]

        # 错位增强：评估集默认全部错位 + 固定种子（可复现）
        if self.misalign_prob > 0 and random.random() < self.misalign_prob:
            seed = idx if self.use_fixed_seed else None
            ir_tensor = apply_misalignment(ir_tensor, seed=seed)

        return ir_tensor, vis_tensor, name


# ══════════════════════════════════════════════════════════
# 2. 图像质量指标 (无需标注)
# ══════════════════════════════════════════════════════════
def entropy(img_np: np.ndarray) -> float:
    gray = (img_np * 255).astype(np.uint8)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    return -np.sum(hist * np.log2(hist + 1e-8))


def spatial_frequency(img_np: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(img_np, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(img_np, axis=1) ** 2))
    return np.sqrt(rf ** 2 + cf ** 2)


def compute_image_metrics(tensor: torch.Tensor) -> dict:
    np_img = tensor.squeeze(0).mean(0).cpu().numpy() if tensor.shape[0] > 1 \
             else tensor.squeeze().cpu().numpy()
    return {
        "EN": entropy(np_img),
        "SF": spatial_frequency(np_img),
    }


# ══════════════════════════════════════════════════════════
# 3. 三种融合条件的前向推理
# ══════════════════════════════════════════════════════════
@torch.no_grad()
def fuse_no_reg(ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:

    ir_3ch = ir.repeat(1, 3, 1, 1) if ir.shape[1] == 1 else ir
    return 0.5 * ir_3ch + 0.5 * vi


def _normalize_fused(fused: torch.Tensor) -> torch.Tensor:

    f_min = fused.min()
    f_max = fused.max()
    if (f_max - f_min).abs() > 1e-6:
        fused = (fused - f_min) / (f_max - f_min)
    else:
        fused = torch.zeros_like(fused)
    if fused.shape[1] == 1:
        fused = fused.repeat(1, 3, 1, 1)
    return fused.clamp(0, 1)


@torch.no_grad()
def fuse_full_reg(model: FusionRegister,
                  ir: torch.Tensor, vi: torch.Tensor,
                  device) -> torch.Tensor:
    try:
        # Force N=1 inside FusionRegister so this is the true full-registration
        # counterpart of the internal TDSR selective path.
        outputs = model(vi, vi, ir, force_necessity=1.0)
        fused = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        fused = _normalize_fused(fused)

    except Exception as e:
        print(f"  [警告] FusionRegister 前向失败: {e}，回退到无配准")
        fused = fuse_no_reg(ir, vi)

    return fused.clamp(0, 1)


@torch.no_grad()
def fuse_tdsr(model: FusionRegister,
              ir: torch.Tensor, vi: torch.Tensor,
              device) -> tuple:

    try:
        outputs = model(vi, vi, ir)
        fused_out = outputs[0]

        raw_tdsr = outputs[7]

        if isinstance(raw_tdsr, (list, tuple)):
            necessity = raw_tdsr[0]
            gate = raw_tdsr[3] if len(raw_tdsr) > 3 else necessity
        else:
            necessity = raw_tdsr
            gate = necessity

        fused_out = _normalize_fused(fused_out)

    except Exception as e:
        print(f"  [警告] FusionRegister 前向失败: {e}，回退到无配准 + 均匀 necessity")
        fused_out = fuse_no_reg(ir, vi)
        necessity = torch.ones(1, 1, ir.shape[2], ir.shape[3],
                               device=device) * 0.5
        gate = necessity

    gate_mode = getattr(model, "gate_mode", "soft")

    if gate_mode in ["hard", "threshold"]:
        sparsity = (gate < 0.5).float().mean().item()

    elif gate_mode == "gumbel":
        # gumbel_hard=True 时 gate 为 0/1；gumbel_hard=False 时 gate 为连续概率
        if getattr(model, "gumbel_hard", True):
            sparsity = (gate < 0.5).float().mean().item()
        else:
            sparsity = (1.0 - gate).mean().item()

    else:
        # soft / hybrid 用潜在 skip ratio
        sparsity = (1.0 - gate).mean().item()

    return fused_out, necessity, sparsity


@torch.no_grad()
def fuse_residual(pixel_fused: torch.Tensor,
                  tdsr_fused: torch.Tensor,
                  gamma: float) -> torch.Tensor:
    """Task-preserving residual correction: F = F_pix + gamma * (F_tdsr - F_pix)."""
    return (pixel_fused + gamma * (tdsr_fused - pixel_fused)).clamp(0, 1)

# ══════════════════════════════════════════════════════════
# 4. 可视化：保存 Necessity Map 对比图
# ══════════════════════════════════════════════════════════
def save_necessity_vis(ir, vi, fused_tdsr, necessity, name, out_dir: Path):
    """保存单张图的四格对比图（IR / VIS / Fused / Necessity Map）"""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def show(ax, t, title, cmap="gray"):
        if t.shape[0] > 1:
            arr = t.permute(1, 2, 0).cpu().numpy()
            ax.imshow(arr.clip(0, 1))
        else:
            ax.imshow(t.squeeze().cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    show(axes[0], ir[0],         "IR Input",        cmap="gray")
    show(axes[1], vi[0],         "VIS Input")
    show(axes[2], fused_tdsr[0], "TDSR Fused")
    im = axes[3].imshow(necessity[0, 0].cpu().numpy(),
                        cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Necessity Map N(p)", fontsize=12)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.suptitle(f"Sample: {name}", fontsize=13)
    plt.tight_layout()
    save_path = out_dir / f"{name}_necessity.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1 or 3, H, W] → PIL Image"""
    arr = t.squeeze(0)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_fused_image(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor[0]).save(str(path))



def _infer_nc_and_names(label_dir: Path):

    M3FD_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    if label_dir is None:
        return 6, M3FD_NAMES

    class_ids = set()
    for txt in label_dir.glob("*.txt"):
        try:
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
        except Exception:
            pass

    if not class_ids:
        # 标注文件为空或扫描失败，回退到 M3FD 默认值
        return 6, M3FD_NAMES

    nc = max(class_ids) + 1
    # 尽量用已知名称，否则用 class0/class1/...
    names = M3FD_NAMES[:nc] if nc <= len(M3FD_NAMES) else [f"class{i}" for i in range(nc)]
    return nc, names


def run_yolo_eval(fused_dir: Path, yolo_ckpt: str,
                  label_dir: Path, img_h: int, img_w: int) -> dict:

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [跳过] ultralytics 未安装，跳过 YOLO 评估")
        print("  安装: pip install ultralytics")
        return {}

    if label_dir is None or not Path(label_dir).exists():
        print(f"  [跳过] 标注目录不存在或未指定: {label_dir}")
        print("  请用 --label_dir 指定 YOLO 格式标注目录")
        return {}

    label_dir = Path(label_dir)

    import yaml, tempfile, shutil

    # ── 构建临时目录，YOLO 要求 images/ 和 labels/ 同级 ──────────
    tmp_root = Path(tempfile.mkdtemp(prefix="yolo_eval_"))
    tmp_img_dir = tmp_root / "images" / "val"
    tmp_lbl_dir = tmp_root / "labels" / "val"
    tmp_img_dir.mkdir(parents=True)
    tmp_lbl_dir.mkdir(parents=True)

    try:
        # 把融合图软链接（或复制）到 tmp_img_dir
        for img_path in sorted(fused_dir.glob("*.png")) + sorted(fused_dir.glob("*.jpg")):
            dst = tmp_img_dir / img_path.name
            try:
                dst.symlink_to(img_path.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(str(img_path), str(dst))

        # 把对应标注软链接（或复制）到 tmp_lbl_dir
        linked = 0
        for img_path in tmp_img_dir.iterdir():
            stem = img_path.stem
            lbl_src = label_dir / f"{stem}.txt"
            if not lbl_src.exists():
                # 宽容匹配：有些数据集标注名带不同后缀前缀
                candidates = list(label_dir.glob(f"{stem}.*"))
                lbl_src = candidates[0] if candidates else None
            if lbl_src and lbl_src.exists():
                dst = tmp_lbl_dir / f"{stem}.txt"
                try:
                    dst.symlink_to(lbl_src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(str(lbl_src), str(dst))
                linked += 1

        if linked == 0:
            print(f"  [警告] 在 {label_dir} 中没有找到任何匹配的标注文件，跳过 YOLO 评估")
            return {}
        print(f"  标注匹配: {linked} 个文件")

        # 自动推断类别数
        nc, names = _infer_nc_and_names(label_dir)
        print(f"  自动推断类别: nc={nc}, names={names}")

        # 构建 data yaml（Ultralytics 强制要求同时有 train: 和 val:，用同一目录绕过）
        data_cfg = {
            "path":  str(tmp_root),
            "train": "images/val",   # 仅占位，实际只跑 val
            "val":   "images/val",
            "nc":    nc,
            "names": names,
        }
        tmp_yaml = tmp_root / "data.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.dump(data_cfg, f, allow_unicode=True)

        # 运行 YOLO val
        model = YOLO(yolo_ckpt)
        results = model.val(
            data=str(tmp_yaml),
            imgsz=max(img_h, img_w),
            batch=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
            plots=False,   # headless: skip curve/confusion plotting (font issues)
        )

        return {
            "mAP50":     round(float(results.box.map50), 4),
            "mAP50_95":  round(float(results.box.map),   4),
            "Precision": round(float(results.box.mp),    4),
            "Recall":    round(float(results.box.mr),    4),
        }

    except Exception as e:
        print(f"  [错误] YOLO 评估失败: {e}")
        import traceback; traceback.print_exc()
        return {}

    finally:
        # 无论成功与否，清理临时目录
        shutil.rmtree(str(tmp_root), ignore_errors=True)

def infer_tdsr_mid_ch_from_ckpt(ckpt) -> int:

    state = None

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
            key_candidates = [
                "tdsr.mse.stem.0.weight",
                "module.tdsr.mse.stem.0.weight",
            ]
        elif "tdsr" in ckpt and isinstance(ckpt["tdsr"], dict):
            state = ckpt["tdsr"]
            key_candidates = [
                "mse.stem.0.weight",
                "module.mse.stem.0.weight",
            ]
        else:
            state = ckpt
            key_candidates = [
                "tdsr.mse.stem.0.weight",
                "module.tdsr.mse.stem.0.weight",
                "mse.stem.0.weight",
                "module.mse.stem.0.weight",
            ]
    elif isinstance(ckpt, dict):
        state = ckpt
        key_candidates = [
            "tdsr.mse.stem.0.weight",
            "module.tdsr.mse.stem.0.weight",
        ]
    else:
        return 64

    for k in key_candidates:
        if k in state:
            return int(state[k].shape[0])

    # 兜底：模糊搜索
    if isinstance(state, dict):
        for k, v in state.items():
            if k.endswith("tdsr.mse.stem.0.weight") or k.endswith("mse.stem.0.weight"):
                if hasattr(v, "shape") and len(v.shape) >= 1:
                    return int(v.shape[0])

    print("  ⚠️  未能从 checkpoint 推断 tdsr_mid_ch，默认使用 64")
    return 64

@torch.no_grad()
def fuse_model_no_fine_reg(model: FusionRegister,
                           ir: torch.Tensor, vi: torch.Tensor,
                           device) -> torch.Tensor:
    try:
        outputs = model(vi, vi, ir, force_necessity=0.0)
        fused = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        fused = _normalize_fused(fused)
    except Exception as e:
        print(f"  [警告] model no-fine-reg 前向失败: {e}，回退到像素平均")
        fused = fuse_no_reg(ir, vi)
    return fused.clamp(0, 1)

def set_seed(seed=123):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ══════════════════════════════════════════════════════════
# 7. 主评估流程
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TDSR 评估脚本")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据集根目录（含 infrared/test + visible/test）；"
                             "--yolo_only 模式下可省略")
    parser.add_argument("--ckpt", type=str,
                        default="results/checkpoints/tdsr_final.pth",
                        help="TDSR 最终权重路径（tdsr_final.pth）")
    parser.add_argument("--output", type=str, default="results/eval",
                        help="评估结果输出目录")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集子集名称（test / val / train）")
    parser.add_argument("--yolo", type=str, default=None,
                        help="YOLOv8 权重路径（可选，填写后自动跑检测评估）")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="YOLO 格式标注目录（YOLO 评估时需要）")
    parser.add_argument("--img_h", type=int, default=480)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--misalign_prob", type=float, default=1.0,
                        help="测试集 IR 错位概率 (默认 1.0=全部错位; 0=关闭，用于消融)")
    parser.add_argument("--max_vis", type=int, default=10,
                        help="保存 necessity map 可视化的最大张数")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="调试用：只评估前 N 张；0 表示完整评估集")
    parser.add_argument("--yolo_only", action="store_true",
                        help="跳过图像生成，直接对 --output/images/ 下已有图像跑 YOLO 评估")
    parser.add_argument("--gate_mode", type=str, default="soft",
                        choices=["soft", "threshold", "hard", "gumbel", "hybrid"],
                        help="TDSR selective gate mode")

    parser.add_argument("--gate_threshold", type=float, default=0.5,
                        help="threshold/hard gate 的阈值")
    parser.add_argument("--gumbel_tau", type=float, default=1.0,
                        help="Gumbel-Softmax temperature, only used when --gate_mode gumbel")

    parser.add_argument("--gumbel_hard", type=int, default=1,
                        help="Gumbel hard sampling: 1=hard ST binary gate, 0=soft sample")

    parser.add_argument("--gumbel_eval_deterministic", action="store_true",
                        help="Eval 时禁用 Gumbel 随机采样；hard=1 时退化为 threshold，hard=0 时退化为 soft")

    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                        help="Hybrid gate: alpha*soft + (1-alpha)*hard")
    parser.add_argument("--residual_gamma", type=float, default=None,
                        help="If set, also evaluate residual selective fusion: "
                             "PixelAvg + gamma * (TDSR - PixelAvg)")
    parser.add_argument("--seed", type=int, default=123,
                        help="随机种子；用于 Gumbel stochastic eval 的可复现性")

    args = parser.parse_args()
    set_seed(args.seed)

    # --yolo_only 模式下提前校验
    if args.yolo_only:
        if not args.yolo:
            parser.error("--yolo_only 需要同时指定 --yolo <权重路径>")
        if not args.label_dir:
            parser.error("--yolo_only 需要同时指定 --label_dir <标注目录>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # --yolo_only：跳过图像生成，直接对已有图像跑 YOLO
    # ══════════════════════════════════════════════════════
    if args.yolo_only:
        dir_pixel_noreg = out_dir / "images" / "pixel_no_reg"
        dir_model_noreg = out_dir / "images" / "model_no_fine_reg"
        dir_fullreg = out_dir / "images" / "full_reg"
        dir_tdsr = out_dir / "images" / "tdsr"
        dir_residual = out_dir / "images" / "residual_tdsr"

        existing = [(n, d) for n, d in [
            ("pixel_no_reg", dir_pixel_noreg),
            ("model_no_fine_reg", dir_model_noreg),
            ("full_reg", dir_fullreg),
            ("tdsr", dir_tdsr),
            ("residual_tdsr", dir_residual),
        ] if d.exists() and any(d.glob("*.png"))]

        if not existing:
            print(f"  [错误] {out_dir / 'images'} 下未找到任何已生成图像。\n"
                  f"  请先不加 --yolo_only 运行一次以生成融合图像。")
            sys.exit(1)

        print(f"\n[yolo_only] 跳过图像生成，直接评估已有图像")
        for n, d in existing:
            cnt = len(list(d.glob("*.png")))
            print(f"  {n}: {cnt} 张图像  ({d})")

        label_dir    = Path(args.label_dir)
        yolo_results = {}
        print("\n开始 YOLO 检测评估...")
        for cond_name, img_dir in [
            ("pixel_no_reg", dir_pixel_noreg),
            ("model_no_fine_reg", dir_model_noreg),
            ("full_reg", dir_fullreg),
            ("tdsr", dir_tdsr),
            ("residual_tdsr", dir_residual),
        ]:
            if not img_dir.exists() or not any(img_dir.glob("*.png")):
                print(f"  [跳过] {cond_name}: 目录不存在或为空")
                continue
            print(f"  评估条件: {cond_name}...")
            res = run_yolo_eval(img_dir, args.yolo, label_dir,
                                args.img_h, args.img_w)
            yolo_results[cond_name] = res

        report_lines = [
            "=" * 62,
            "  TDSR YOLO-Only 评估报告",
            "=" * 62,
            f"  输出目录: {out_dir}",
            "",
            "  ─── YOLO 检测评估 (mAP@0.5) ───────────────────",
            f"  {'条件':<22} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}",
            "  " + "-" * 62,
        ]
        for cond, res in yolo_results.items():
            if res:
                report_lines.append(
                    f"  {cond:<22} {res.get('mAP50', '-'):>8} "
                    f"{res.get('mAP50_95', '-'):>10} "
                    f"{res.get('Precision', '-'):>10} "
                    f"{res.get('Recall', '-'):>8}"
                )
            else:
                report_lines.append(f"  {cond:<22}   (无结果)")
        report_lines += ["", "=" * 62]

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        report_path = out_dir / "eval_report_yolo.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n报告已保存: {report_path}")
        return   # ← 提前退出，不走后续完整流程

    # ── 数据集 ──────────────────────────────────────────
    if not args.data_root:
        parser.error("非 --yolo_only 模式下必须指定 --data_root")
    print(f"  错位增强: misalign_prob={args.misalign_prob} "
          f"({'开启，固定种子' if args.misalign_prob > 0 else '关闭'})")
    dataset = FusionTestDataset(
        args.data_root, args.img_h, args.img_w,
        split=args.split,
        misalign_prob=args.misalign_prob,
        use_fixed_seed=True,        # 评估集始终固定种子，保证可复现
    )
    if args.max_samples and args.max_samples > 0:
        max_samples = min(args.max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
        print(f"  调试模式: 仅评估前 {max_samples} 张")
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=False, num_workers=0)

    print("\n加载模型...")

    ckpt_path = Path(args.ckpt)
    ckpt = None

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        inferred_mid_ch = infer_tdsr_mid_ch_from_ckpt(ckpt)
        print(f"  自动推断 tdsr_mid_ch={inferred_mid_ch}")
    else:
        inferred_mid_ch = 64
        print(f"  ⚠️  权重文件不存在: {ckpt_path}，使用 tdsr_mid_ch={inferred_mid_ch} 随机初始化")

    backbone = FusionRegister(
        tdsr_mid_ch=inferred_mid_ch,
        gate_mode=args.gate_mode,
        gate_threshold=args.gate_threshold,
        gumbel_tau=args.gumbel_tau,
        gumbel_hard=bool(args.gumbel_hard),
        gumbel_eval_deterministic=args.gumbel_eval_deterministic,
        hybrid_alpha=args.hybrid_alpha,
    ).to(device).eval()
    ckpt_path = Path(args.ckpt)
    if ckpt is not None:
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                missing, unexpected = backbone.load_state_dict(ckpt["model"], strict=False)

                real_missing = [k for k in missing
                                if not k.endswith(("total_ops", "total_params"))]
                real_unexpected = [k for k in unexpected
                                   if not k.endswith(("total_ops", "total_params"))]

                if real_missing:
                    print(f"  ⚠️  缺失 key ×{len(real_missing)}: "
                          f"{real_missing[:5]}{'...' if len(real_missing) > 5 else ''}")
                if real_unexpected:
                    print(f"  ℹ️  忽略多余 key ×{len(real_unexpected)}")

                print(f"  ✅ 加载完整 backbone 权重（含 backbone.tdsr）: {ckpt_path}")

            elif "tdsr" in ckpt:
                missing, unexpected = backbone.tdsr.load_state_dict(
                    ckpt["tdsr"], strict=False
                )
                print(f"  ✅ 加载 backbone.tdsr 权重（仅 tdsr 子模块）: {ckpt_path}")

            else:
                print(f"  ⚠️  checkpoint 中未找到 'model' 或 'tdsr' key，"
                      f"可用 key: {list(ckpt.keys())}")

        else:
            missing, unexpected = backbone.load_state_dict(ckpt, strict=False)
            print(f"  ✅ 加载裸权重格式: {ckpt_path}")
    tdsr_params = count_parameters(backbone.tdsr)
    print(f"  backbone.tdsr 参数量: {tdsr_params['total']/1e6:.3f}M")

    backbone.gate_mode = args.gate_mode
    backbone.gate_threshold = args.gate_threshold
    backbone.gumbel_tau = args.gumbel_tau
    backbone.gumbel_hard = bool(args.gumbel_hard)
    backbone.gumbel_eval_deterministic = args.gumbel_eval_deterministic
    backbone.hybrid_alpha = args.hybrid_alpha
    backbone.inference = False
    backbone.eval()

    print(
        f"  Gate mode: {backbone.gate_mode}, "
        f"threshold={backbone.gate_threshold}, "
        f"gumbel_tau={backbone.gumbel_tau}, "
        f"gumbel_hard={backbone.gumbel_hard}, "
        f"gumbel_eval_deterministic={backbone.gumbel_eval_deterministic}, "
        f"hybrid_alpha={backbone.hybrid_alpha}"
    )
    # ── 输出目录 ─────────────────────────────────────────
    dir_pixel_noreg = out_dir / "images" / "pixel_no_reg"
    dir_model_noreg = out_dir / "images" / "model_no_fine_reg"
    dir_fullreg = out_dir / "images" / "full_reg"
    dir_tdsr = out_dir / "images" / "tdsr"
    dir_residual = out_dir / "images" / "residual_tdsr"
    dir_vis = out_dir / "necessity_maps"

    output_dirs = [dir_pixel_noreg, dir_model_noreg, dir_fullreg, dir_tdsr, dir_vis]
    if args.residual_gamma is not None:
        output_dirs.append(dir_residual)
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ── 主循环 ───────────────────────────────────────────
    metrics = {
        "pixel_no_reg": [],
        "model_no_fine_reg": [],
        "full_reg": [],
        "tdsr": [],
        "residual_tdsr": [],
        "sparsity": [],
    }

    print(f"\n开始评估（共 {len(dataset)} 张图）...\n")

    for idx, (ir, vi, names) in enumerate(loader):
        ir = ir.to(device)
        vi = vi.to(device)
        name = names[0] if isinstance(names, (list, tuple)) else names

        # Baseline 0: 像素级无配准平均融合
        f_pixel_noreg = fuse_no_reg(ir, vi)

        # Condition A: 同一 FusionRegister backbone，关闭精细配准
        f_model_noreg = fuse_model_no_fine_reg(backbone, ir, vi, device)

        # Condition B: 同一 FusionRegister backbone，全量精细配准
        f_fullreg = fuse_full_reg(backbone, ir, vi, device)

        # Condition C: TDSR selective
        f_tdsr, necessity, sparsity = fuse_tdsr(backbone, ir, vi, device)
        f_residual = None
        if args.residual_gamma is not None:
            f_residual = fuse_residual(f_pixel_noreg, f_tdsr, args.residual_gamma)

        save_fused_image(f_pixel_noreg, dir_pixel_noreg / f"{name}.png")
        save_fused_image(f_model_noreg, dir_model_noreg / f"{name}.png")
        save_fused_image(f_fullreg, dir_fullreg / f"{name}.png")
        save_fused_image(f_tdsr, dir_tdsr / f"{name}.png")
        if f_residual is not None:
            save_fused_image(f_residual, dir_residual / f"{name}.png")

        # 指标
        m_pixel_noreg = compute_image_metrics(f_pixel_noreg[0])
        m_model_noreg = compute_image_metrics(f_model_noreg[0])
        m_fullreg = compute_image_metrics(f_fullreg[0])
        m_tdsr = compute_image_metrics(f_tdsr[0])
        m_residual = compute_image_metrics(f_residual[0]) if f_residual is not None else None

        metrics["pixel_no_reg"].append(m_pixel_noreg)
        metrics["model_no_fine_reg"].append(m_model_noreg)
        metrics["full_reg"].append(m_fullreg)
        metrics["tdsr"].append(m_tdsr)
        if m_residual is not None:
            metrics["residual_tdsr"].append(m_residual)
        metrics["sparsity"].append(sparsity)

        # 可视化（前 max_vis 张）
        if idx < args.max_vis:
            save_necessity_vis(ir, vi, f_tdsr, necessity, name, dir_vis)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx + 1:4d}/{len(dataset)}]  "
                  f"Skip={sparsity * 100:.1f}%  "
                  f"EN: pixelNoReg={m_pixel_noreg['EN']:.3f} "
                  f"modelNoReg={m_model_noreg['EN']:.3f} "
                  f"fullReg={m_fullreg['EN']:.3f} "
                  f"TDSR={m_tdsr['EN']:.3f}"
                  f"{' residual=' + format(m_residual['EN'], '.3f') if m_residual else ''}")

    # ── 汇总指标 ──────────────────────────────────────────
    def avg(lst, key):
        return np.mean([d[key] for d in lst])

    avg_sparsity = np.mean(metrics["sparsity"])

    summary = {
        "Pixel NoReg": {
            "EN": avg(metrics["pixel_no_reg"], "EN"),
            "SF": avg(metrics["pixel_no_reg"], "SF"),
        },
        "Model NoFineReg": {
            "EN": avg(metrics["model_no_fine_reg"], "EN"),
            "SF": avg(metrics["model_no_fine_reg"], "SF"),
        },
        "Full Registration": {
            "EN": avg(metrics["full_reg"], "EN"),
            "SF": avg(metrics["full_reg"], "SF"),
        },
        "TDSR Selective": {
            "EN": avg(metrics["tdsr"], "EN"),
            "SF": avg(metrics["tdsr"], "SF"),
        },
    }
    if metrics["residual_tdsr"]:
        summary[f"Residual Selective (gamma={args.residual_gamma:g})"] = {
            "EN": avg(metrics["residual_tdsr"], "EN"),
            "SF": avg(metrics["residual_tdsr"], "SF"),
        }

    # ── 可选 YOLO 评估 ────────────────────────────────────
    yolo_results = {}
    if args.yolo:
        label_dir = Path(args.label_dir) if args.label_dir else None
        print("\n开始 YOLO 检测评估...")
        for cond_name, img_dir in [
            ("pixel_no_reg", dir_pixel_noreg),
            ("model_no_fine_reg", dir_model_noreg),
            ("full_reg", dir_fullreg),
            ("tdsr", dir_tdsr),
            ("residual_tdsr", dir_residual),
        ]:
            if cond_name == "residual_tdsr" and args.residual_gamma is None:
                continue
            print(f"  评估条件: {cond_name}...")
            res = run_yolo_eval(img_dir, args.yolo, label_dir,
                                args.img_h, args.img_w)
            yolo_results[cond_name] = res

    # ── 生成报告 ──────────────────────────────────────────
    report_lines = [
        "=" * 62,
        "  TDSR 评估报告",
        "=" * 62,
        f"  数据集:  {args.data_root}",
        f"  Split:   {args.split}",
        f"  样本数:  {len(dataset)}",
        f"  权重:    {args.ckpt}",
        f"  Gate:    {args.gate_mode}",
        f"  Thresh:  {args.gate_threshold}",
        f"  Gumbel tau:   {args.gumbel_tau}",
        f"  Gumbel hard:  {bool(args.gumbel_hard)}",
        f"  Gumbel det:   {args.gumbel_eval_deterministic}",
        f"  Hybrid alpha: {args.hybrid_alpha}",
        f"  Residual gamma: {args.residual_gamma}",
        f"  Seed:         {args.seed}",
        "",
        "  ─── 图像质量指标 ────────────────────────────────",
        f"  {'条件':<22} {'EN':>8} {'SF':>8}",
        "  " + "-" * 42,
    ]
    for cond, vals in summary.items():
        report_lines.append(f"  {cond:<22} {vals['EN']:>8.4f} {vals['SF']:>8.4f}")

    report_lines += [
        "",
        f"  平均稀疏率 (Sparsity): {avg_sparsity*100:.1f}%",
        f"  → 约 {avg_sparsity * 100:.1f}% 的像素跳过了配准或具有潜在跳过贡献（基于 gate 统计）",
        "",
        "  ─── 潜在 FLOPs 节省估算 ───────────────────────",
        f"  全量配准理论 FLOPs:          54.11 G (来自 flops_report.txt)",
        f"  根据 skip ratio 的潜在有效 FLOPs: {54.11 * (1 - avg_sparsity):.2f} G",
        f"  潜在节省:                    {avg_sparsity * 100:.1f}%",
        f"  注: 当前实现仍为 dense forward，该数值表示若采用稀疏执行时的理论节省。",
        "",
    ]

    if yolo_results:
        report_lines += [
            "  ─── YOLO 检测评估 (mAP@0.5) ───────────────────",
            f"  {'条件':<22} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}",
            "  " + "-" * 62,
        ]
        for cond, res in yolo_results.items():
            if res:
                report_lines.append(
                    f"  {cond:<22} {res.get('mAP50','-'):>8} "
                    f"{res.get('mAP50_95','-'):>10} "
                    f"{res.get('Precision','-'):>10} "
                    f"{res.get('Recall','-'):>8}"
                )
    else:
        report_lines += [
            "  ─── YOLO 检测评估 ──────────────────────────────",
            "  未运行（需传入 --yolo <权重路径>）",
            "  运行方式:",
            "    python eval/eval_tdsr.py ... --yolo path/to/yolov8n.pt",
        ]

    report_lines += ["", "=" * 62]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = out_dir / "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")
    print(f"融合图像已保存: {out_dir / 'images'}")
    print(f"可视化已保存:   {dir_vis}")


if __name__ == "__main__":
    main()
