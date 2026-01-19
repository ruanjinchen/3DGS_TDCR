#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo_export_pointclouds.py

把 3D Gaussian Splatting（含形变）训练结果导出成“点云（PLY ascii 1.0）”，
用于与你的“点云->点云”方法做对比。

导出内容：
  <out_root>/gt/<sid:06d>.ply
  <out_root>/pred/<sid:06d>.ply

- GT：从 --gt_pcd_dir 读取对应帧的 .ply，然后重写成 ascii 1.0（保留颜色）。
- Pred：从训练输出目录 -m 的 point_cloud/iteration_xxxx/point_cloud.ply 读取高斯中心，
        并用 checkpoint 的 GaussianDeformer 在每个 joint 配置下做前向变形，输出变形后的中心点。
        颜色使用 SH 的 DC 项（view-independent）：rgb = f_dc*C0 + 0.5。

新增：统一“最终保存点数”的下采样
--------------------------------
你可以通过 --final_num_points 指定 GT 和 Pred *最终保存* 的点数 N。
- 该下采样发生在每帧的 GT/PRED 点云都生成出来之后（即写 ply 之前）。
- 这是一个全局 N：每一帧都会被随机无放回下采样到 N。
- 约束：
  - 若任意帧 GT 点数 < N -> 直接报错
  - 若 Pred（opacity 过滤 + 可选 max_points 预裁剪后）点数 < N -> 直接报错

说明：
- 该脚本默认会按 opacity(sigmoid(opacity_param)) 过滤掉低 opacity 的 Gaussians。
- 你提到可能存在“scale 不一致”。在这个 repo/数据管线里，3DGS 初始化 points3D.txt 通常来自 GT 点云，
  理论上尺度应一致；但如果你确实观察到缩放差异，可以用 --align scale 或 --align sim3。

示例：
python demo_export_pointclouds.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.all \
  -m out_tdcr2_no_base_stage2 \
  --iteration 30000 \
  --gt_pcd_dir /data/yxk/K-data/K/fllm-sm/sim/2m_no_base/pointcloud \
  --out_root demo_out/2m_no_base \
  --opacity_thresh 0.005 \
  --final_num_points 50000

如果想自动估计尺度并对 pred 做统一缩放（参考一帧）：
  --align scale
或同时做尺度+平移（相似变换）：
  --align sim3

注意：
- 需要能正常 import 本仓库训练所需依赖（torch/cuda、pytorch3d、plyfile、sklearn 等）。
- 本脚本不依赖 rasterizer，不会渲染图片。
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from arguments import ModelParams, get_combined_args
from F_kinematic import GaussianDeformer
from utils.system_utils import searchForMaxIteration
from utils.sh_utils import C0

try:
    import open3d as o3d  # 用于读取 GT ply（支持 binary/ascii）
except Exception:
    o3d = None

try:
    from plyfile import PlyData
except Exception:
    PlyData = None


# -------------------------- basic IO --------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_lines(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _parse_sid_from_any_string(s: str) -> Optional[int]:
    """从字符串中解析 sid。

    注意：相机名可能包含数字（例如 cam0_000123.png），
    如果用 re.search 会误把 cam0 里的 0 当成 sid。
    因此这里取“最后一段连续数字”作为 sid。
    """

    nums = re.findall(r"(\d+)", s)
    if not nums:
        return None
    return int(nums[-1])


def read_frame_ids(dataset_root: Path) -> List[int]:
    """读取 make_selfmodel_gs_dataset.py 写出的 frame_ids.txt（每行一个 6 位 sid）。"""

    f = dataset_root / "frame_ids.txt"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}（你的数据集是否由 make_selfmodel_gs_dataset.py 导出？）")
    sids: List[int] = []
    for ln in _read_lines(f):
        m = re.search(r"(\d+)", ln)
        if m:
            sids.append(int(m.group(1)))
    if not sids:
        raise RuntimeError(f"{f} 为空或无法解析")
    return sids


def read_joint_txt(dataset_root: Path) -> np.ndarray:
    """读取 joint.txt（每行形如 [0.1 0.2 ...]），返回 (N, D) float32。"""

    f = dataset_root / "joint.txt"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}")
    rows: List[List[float]] = []
    for ln in _read_lines(f):
        ln = ln.strip()
        if ln.startswith("[") and ln.endswith("]"):
            ln = ln[1:-1].strip()
        if not ln:
            continue
        vals = [float(x) for x in ln.split()]
        rows.append(vals)
    if not rows:
        raise RuntimeError(f"{f} 为空或无法解析")
    arr = np.asarray(rows, dtype=np.float32)
    return arr


def read_sids_from_info_json(info_json: Path) -> List[int]:
    """从 info_*.json 里解析帧 sid 列表。

    说明：info json 里的 images 会包含多相机，因此同一个 sid 会出现多次。
    这里会去重并按数值排序返回。
    """

    if not info_json.exists():
        raise FileNotFoundError(f"找不到 {info_json}")

    obj = json.loads(info_json.read_text(encoding="utf-8"))
    images = obj.get("images", [])
    if not images:
        return []

    sids: List[int] = []
    for im in images:
        nm = str(im.get("name", ""))
        sid = _parse_sid_from_any_string(nm)
        if sid is not None:
            sids.append(sid)
    sids = sorted(set(sids))
    return sids


def read_frame_ids_by_split(dataset_root: Path, split: str) -> List[int]:
    """根据 split 选择导出哪些帧。

    - all : frame_ids.txt（全部帧）
    - train/val/test : info_all_<split>.json
    - zero : info_zero_train.json（通常只有一帧）
    """

    split = split.lower()
    if split == "all":
        return read_frame_ids(dataset_root)
    if split in {"train", "val", "test"}:
        return read_sids_from_info_json(dataset_root / f"info_all_{split}.json")
    if split == "zero":
        return read_sids_from_info_json(dataset_root / "info_zero_train.json")
    raise ValueError(f"未知 split: {split}")


def parse_zero_sid_from_info(dataset_root: Path) -> Optional[int]:
    """尝试从 info_zero_train.json 的 image name 中解析 zero_sid。"""

    p = dataset_root / "info_zero_train.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        images = obj.get("images", [])
        if not images:
            return None
        nm = images[0].get("name", "")  # e.g. cam0_000123.png
        return _parse_sid_from_any_string(nm)
    except Exception:
        return None


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussians_from_pointcloud_ply(ply_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 point_cloud.ply 读取：
    - xyz: (N,3)
    - f_dc: (N,3)  (SH 的 DC 系数)
    - opacity_param: (N,1)  (注意是 sigmoid 之前的参数)

    只读取我们导出点云需要的字段，避免依赖 GaussianModel/simple_knn 等编译扩展。
    """

    if PlyData is None:
        raise ImportError("缺少 plyfile：pip install plyfile")
    if not ply_path.exists():
        raise FileNotFoundError(f"找不到 {ply_path}")

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError(f"{ply_path} 不包含 vertex")
    v = ply["vertex"].data

    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    for k in required:
        if k not in v.dtype.names:
            raise RuntimeError(f"{ply_path} 缺少字段 {k}。当前字段: {v.dtype.names}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    opacity_param = np.asarray(v["opacity"], dtype=np.float32).reshape(-1, 1)
    return xyz, f_dc, opacity_param


def rgb_from_fdc_uint8(f_dc: np.ndarray) -> np.ndarray:
    """把 SH DC 系数转成 RGB uint8。

    - f_dc 存的是 (rgb-0.5)/C0
    - 所以 rgb = f_dc*C0 + 0.5
    """

    rgb = f_dc * float(C0) + 0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return rgb_u8


def write_ply_xyzrgb_ascii(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    """写 PLY ascii 1.0：x y z + uchar rgb。"""

    assert xyz.shape[0] == rgb_u8.shape[0]
    n = int(xyz.shape[0])
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb_u8):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def load_gt_xyzrgb(gt_ply: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 GT ply，返回 xyz(float32) 和 rgb(uint8)。

    优先用 open3d（支持 binary/ascii）；若 open3d 不可用则尝试 plyfile。
    """

    if not gt_ply.exists():
        raise FileNotFoundError(f"找不到 GT 点云: {gt_ply}")

    if o3d is not None:
        pcd = o3d.io.read_point_cloud(str(gt_ply))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        col = np.asarray(pcd.colors, dtype=np.float32)
        if xyz.size == 0:
            raise RuntimeError(f"GT 点云为空: {gt_ply}")
        if col.size == 0:
            col = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        # open3d 的颜色通常是 0..1
        if col.max() <= 1.0 + 1e-6:
            rgb_u8 = np.clip(col * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb_u8 = np.clip(col, 0.0, 255.0).astype(np.uint8)
        return xyz, rgb_u8

    # fallback: plyfile（只能可靠处理带 red/green/blue 的 ply）
    if PlyData is None:
        raise ImportError("无法读取 GT ply：请安装 open3d 或 plyfile")

    ply = PlyData.read(str(gt_ply))
    v = ply["vertex"].data
    names = v.dtype.names
    for k in ["x", "y", "z"]:
        if k not in names:
            raise RuntimeError(f"GT ply 缺少 {k}: {gt_ply}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if all(k in names for k in ["red", "green", "blue"]):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)

    return xyz, rgb


def estimate_scale_and_shift(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> Tuple[float, np.ndarray]:
    """用“均方半径”估计一个粗糙的相似变换：x' = x*s + t。

    - s：让 pred/gt 的 RMS 半径一致
    - t：让缩放后的 pred 与 gt 的质心对齐

    这不是 ICP，只是一个不需要点对应关系的粗对齐。
    """

    cp = pred_xyz.mean(axis=0)
    cg = gt_xyz.mean(axis=0)

    rp = float(np.sqrt(((pred_xyz - cp) ** 2).sum(axis=1).mean()) + 1e-12)
    rg = float(np.sqrt(((gt_xyz - cg) ** 2).sum(axis=1).mean()) + 1e-12)

    s = rg / rp
    t = cg - cp * s
    return s, t.astype(np.float32)


def _subsample_xyzrgb(
    xyz: np.ndarray,
    rgb_u8: np.ndarray,
    n_keep: int,
    rng: np.random.Generator,
    *,
    err_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机无放回下采样到 n_keep。

    - 若 n_keep <= 0：原样返回
    - 若 xyz 点数 < n_keep：抛异常
    """

    if n_keep <= 0:
        return xyz, rgb_u8

    n_total = int(xyz.shape[0])
    if n_total < n_keep:
        raise RuntimeError(f"{err_prefix} 点数不足：{n_total} < final_num_points={n_keep}")

    if n_total == n_keep:
        return xyz, rgb_u8

    idx = rng.choice(n_total, size=n_keep, replace=False)
    return xyz[idx], rgb_u8[idx]


def main():
    parser = ArgumentParser(description="Export GT & predicted point clouds for 3DGS TDCR demo")
    mp = ModelParams(parser, sentinel=True)

    parser.add_argument("--iteration", type=int, default=-1, help="加载的迭代轮次；-1 表示自动取最新")
    parser.add_argument("--gt_pcd_dir", type=Path, required=True, help="GT 点云目录（例如 2m_no_base/pointcloud）")
    parser.add_argument("--out_root", type=Path, required=True, help="输出根目录（会创建 gt/ 和 pred/ 子目录）")

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all", "zero"],
        help=(
            "导出哪一部分数据：train/val/test=按数据集 info_all_<split>.json；"
            "all=frame_ids.txt 全部帧；zero=info_zero_train.json（通常只有 1 帧）。"
            "\n注意：如果你生成数据集时没有设置 --val_frac/--test_frac，那么 train==all，"
            "同时 val/test 可能不存在。"
        ),
    )

    parser.add_argument("--opacity_thresh", type=float, default=0.005, help="过滤低 opacity 的高斯（sigmoid 后阈值）")

    # NOTE: max_points 是“预裁剪”pred canonical 点数（在变形前，且对所有帧一致）。
    # 如果你只关心最终保存点数，请优先用 --final_num_points。
    parser.add_argument(
        "--max_points",
        type=int,
        default=-1,
        help=(
            "pred 的预裁剪：在变形前对 canonical gaussians 随机子采样到该数量（-1=不限制）。"
            "该操作发生在 per-frame 变形之前。"
        ),
    )

    parser.add_argument(
        "--final_num_points",
        type=int,
        default=-1,
        help=(
            "最终保存的 GT 与 Pred 点数（全局 N；-1=不限制）。"
            "会在每帧 GT/PRED 点云生成完成后、写 ply 之前随机无放回下采样到 N。"
            "若任意帧 GT 点数或 Pred 点数 < N，则直接报错。"
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="随机下采样的种子（影响 --max_points 与 --final_num_points 的随机性）",
    )

    parser.add_argument(
        "--align",
        type=str,
        default="none",
        choices=["none", "scale", "sim3"],
        help="对 pred 做可选对齐：none=不处理；scale=仅统一缩放；sim3=统一缩放+平移（基于参考帧质心）",
    )
    parser.add_argument(
        "--align_ref_sid",
        type=int,
        default=None,
        help="用于估计缩放/平移的参考帧 sid（不填则优先用 info_zero_train.json 的 zero_sid，否则用 frame_ids[0]）",
    )
    parser.add_argument(
        "--align_sample",
        type=int,
        default=50000,
        help="估计对齐时用于计算尺度的采样点数（从 GT/pred 各随机采样）",
    )

    args = get_combined_args(parser)
    dataset = mp.extract(args)

    # dataset.source_path 形如 .../3dgs/2m_no_base.all
    source_path = Path(dataset.source_path)
    if "." not in source_path.name:
        raise ValueError(f"-s/--source_path 需要带 .all/.zero 后缀：收到 {source_path}")
    dataset_root = Path(str(source_path).rsplit(".", 1)[0])

    # 输出目录
    out_root = Path(args.out_root)
    out_gt = out_root / "gt"
    out_pred = out_root / "pred"
    _ensure_dir(out_gt)
    _ensure_dir(out_pred)

    # 读取“全量”帧列表 + joints（joint.txt/ frame_ids.txt 总是对应全量帧，不随 train/val/test 改变）
    all_frame_ids = read_frame_ids(dataset_root)
    joints_all = read_joint_txt(dataset_root)
    if joints_all.shape[0] != len(all_frame_ids):
        raise RuntimeError(
            f"joint.txt 行数({joints_all.shape[0]}) 与 frame_ids.txt 行数({len(all_frame_ids)}) 不一致。"
            "请确认数据集导出是否完整。"
        )

    # 根据 split 选择要导出的帧
    export_sids = read_frame_ids_by_split(dataset_root, str(args.split))
    if not export_sids:
        raise RuntimeError(
            f"split={args.split} 没有任何帧可导出。\n"
            "如果你想导出 test/val，请在生成数据集时设置 --test_frac/--val_frac。"
        )

    sid_to_idx = {sid: i for i, sid in enumerate(all_frame_ids)}
    export_joints: List[np.ndarray] = []
    for sid in export_sids:
        if sid not in sid_to_idx:
            raise RuntimeError(f"sid={sid} 不在 frame_ids.txt 里，数据集不一致？")
        export_joints.append(joints_all[sid_to_idx[sid]])

    print(f"[export] split={args.split} frames={len(export_sids)}")

    # RNG
    rng = np.random.default_rng(int(args.sample_seed))

    # 最终保存点数
    final_n = int(args.final_num_points)
    if final_n == 0:
        raise ValueError("--final_num_points 不能为 0；请使用 -1（不限制）或正整数")

    # 找 iteration
    model_path = Path(args.model_path)
    if args.iteration == -1:
        it = searchForMaxIteration(str(model_path / "point_cloud"))
    else:
        it = int(args.iteration)

    pc_ply = model_path / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    xyz_can, f_dc, opacity_param = load_gaussians_from_pointcloud_ply(pc_ply)

    # opacity filter
    opacity = _sigmoid_np(opacity_param).reshape(-1)
    keep = opacity >= float(args.opacity_thresh)
    xyz_can = xyz_can[keep]
    f_dc = f_dc[keep]

    # color from f_dc
    rgb_u8_all = rgb_from_fdc_uint8(f_dc)

    # optional pred pre-subsample (fixed across frames)
    if int(args.max_points) > 0 and xyz_can.shape[0] > int(args.max_points):
        idx = rng.choice(xyz_can.shape[0], size=int(args.max_points), replace=False)
        xyz_can = xyz_can[idx]
        rgb_u8_all = rgb_u8_all[idx]

    # 如果指定了最终点数，做一些更早的 sanity check
    if final_n > 0:
        if int(args.max_points) > 0 and int(args.max_points) < final_n:
            raise ValueError(
                f"--max_points({int(args.max_points)}) < --final_num_points({final_n})，"
                "会导致 Pred 预裁剪后点数不足，无法在最终保存阶段下采样到指定点数。"
            )
        if int(xyz_can.shape[0]) < final_n:
            raise RuntimeError(
                f"Pred 点数不足（opacity 过滤 + 可选 max_points 后）：{int(xyz_can.shape[0])} < final_num_points={final_n}"
            )

        # 提前检查第一帧 GT（更早给出错误）；仍会在 loop 内逐帧再次检查
        gt0 = Path(args.gt_pcd_dir) / f"{export_sids[0]:06d}.ply"
        gt0_xyz, _gt0_rgb = load_gt_xyzrgb(gt0)
        if int(gt0_xyz.shape[0]) < final_n:
            raise RuntimeError(
                f"GT 点数不足（示例帧 sid={export_sids[0]:06d}）：{int(gt0_xyz.shape[0])} < final_num_points={final_n}"
            )

    # load deformer if checkpoint exists
    ckpt_path = model_path / f"chkpnt_{it}.pth"
    deformer = None
    if ckpt_path.exists() and it > 7000:
        deformer = GaussianDeformer(num_joints=int(dataset.joints), use_mlp=False)
        deformer.cuda()
        deformer.eval()
        try:
            ckpt = torch.load(str(ckpt_path), weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path))
        # (model_params, transform_params, transform_opt_params, transform_sch_params, first_iter)
        transform_params = ckpt[1]
        # 训练后的 checkpoint 里通常包含 ellipsoid_* 参数；需要 add=True 把这些张量注册成 Parameter
        # 否则 load_state_dict 会报 unexpected key。
        deformer.load(transform_params, optimizer=None, add=True)
        deformer.eval()
    else:
        deformer = None

    # 预先把 canonical 点放到 GPU，避免每帧重复 CPU->GPU 拷贝
    pts_cuda = None
    if deformer is not None:
        pts_cuda = torch.from_numpy(xyz_can).cuda()

    # alignment (estimate once from reference frame)
    align_mode = str(args.align)
    s_align = 1.0
    t_align = np.zeros(3, dtype=np.float32)

    if align_mode != "none":
        ref_sid = args.align_ref_sid
        if ref_sid is None:
            # 默认优先用 zero 帧（如果也在导出列表里），否则用导出列表第一帧
            zsid = parse_zero_sid_from_info(dataset_root)
            if zsid is not None and zsid in export_sids:
                ref_sid = zsid
            else:
                ref_sid = export_sids[0]

        if ref_sid not in sid_to_idx:
            raise ValueError(f"align_ref_sid={ref_sid} 不在 frame_ids.txt 中")

        joints_ref = joints_all[sid_to_idx[ref_sid]]

        # pred(ref)
        if deformer is None:
            pred_ref = xyz_can
        else:
            with torch.no_grad():
                j = torch.from_numpy(joints_ref).cuda()
                pred_ref, *_ = deformer(pts_cuda, j, True)  # delta_mlp=True
                pred_ref = pred_ref.detach().cpu().numpy().astype(np.float32)

        # gt(ref)
        gt_ref_ply = Path(args.gt_pcd_dir) / f"{ref_sid:06d}.ply"
        gt_ref_xyz, _gt_ref_rgb = load_gt_xyzrgb(gt_ref_ply)

        # subsample for stable estimate
        def _subsample(x: np.ndarray, n: int) -> np.ndarray:
            if x.shape[0] <= n:
                return x
            ii = rng.choice(x.shape[0], size=n, replace=False)
            return x[ii]

        pred_s = _subsample(pred_ref, int(args.align_sample))
        gt_s = _subsample(gt_ref_xyz, int(args.align_sample))

        s_align, t_align = estimate_scale_and_shift(pred_s, gt_s)

        if align_mode == "scale":
            t_align = np.zeros(3, dtype=np.float32)

        print(f"[align] mode={align_mode} ref_sid={ref_sid:06d} scale={s_align:.6f} t={t_align.tolist()}")

    # export meta
    meta = {
        "source_path": str(source_path),
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "iteration": int(it),
        "split": str(args.split),
        "num_frames": int(len(export_sids)),
        "num_gaussians_after_opacity_filter": int(xyz_can.shape[0]),
        "opacity_thresh": float(args.opacity_thresh),
        "max_points": int(args.max_points),
        "final_num_points": int(final_n),
        "sample_seed": int(args.sample_seed),
        "align": align_mode,
        "align_scale": float(s_align),
        "align_translation": [float(x) for x in t_align.tolist()],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # main loop
    for sid, joints in zip(export_sids, export_joints):
        # --- GT ---
        gt_in = Path(args.gt_pcd_dir) / f"{sid:06d}.ply"
        gt_xyz, gt_rgb = load_gt_xyzrgb(gt_in)
        gt_xyz, gt_rgb = _subsample_xyzrgb(
            gt_xyz,
            gt_rgb,
            final_n,
            rng,
            err_prefix=f"[GT sid={sid:06d}]",
        )
        gt_out = out_gt / f"{sid:06d}.ply"
        write_ply_xyzrgb_ascii(gt_out, gt_xyz, gt_rgb)

        # --- Pred ---
        if deformer is None:
            pred_xyz = xyz_can.copy()
        else:
            with torch.no_grad():
                j = torch.from_numpy(joints).cuda()
                pred_xyz, *_ = deformer(pts_cuda, j, True)  # delta_mlp=True
                pred_xyz = pred_xyz.detach().cpu().numpy().astype(np.float32)

        if align_mode != "none":
            pred_xyz = pred_xyz * float(s_align) + t_align.reshape(1, 3)

        pred_rgb = rgb_u8_all
        pred_xyz, pred_rgb = _subsample_xyzrgb(
            pred_xyz,
            pred_rgb,
            final_n,
            rng,
            err_prefix=f"[Pred sid={sid:06d}]",
        )

        pred_out = out_pred / f"{sid:06d}.ply"
        write_ply_xyzrgb_ascii(pred_out, pred_xyz, pred_rgb)

    print("\nDONE.")
    print(f"Exported to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
'''
export CUDA_VISIBLE_DEVICES=1
python demo_export_pointclouds.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.all \
  -m out_tdcr2_no_base_stage2 \
  --iteration 30000 \
  --gt_pcd_dir /data/yxk/K-data/K/fllm-sm/sim/2m_no_base/pointcloud \
  --out_root demo_out/2m_no_base \
  --opacity_thresh 0.005 \
  --final_num_points 20000

export CUDA_VISIBLE_DEVICES=1
python demo_export_pointclouds.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_with_base.all \
  -m out_tdcr2_with_base_stage2 \
  --iteration 30000 \
  --gt_pcd_dir /data/yxk/K-data/K/fllm-sm/sim/2m_with_base/pointcloud \
  --out_root demo_out/2m_with_base \
  --opacity_thresh 0.005 \
  --final_num_points 20000

export CUDA_VISIBLE_DEVICES=4
python demo_export_pointclouds.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_no_base.all \
  -m out_tdcr3_no_base_stage2 \
  --iteration 30000 \
  --gt_pcd_dir /data/yxk/K-data/K/fllm-sm/sim/3m_no_base/pointcloud \
  --out_root demo_out/3m_no_base \
  --opacity_thresh 0.005 \
  --final_num_points 20000

export CUDA_VISIBLE_DEVICES=5
python demo_export_pointclouds.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_with_base.all \
  -m out_tdcr3_with_base_stage2 \
  --iteration 30000 \
  --gt_pcd_dir /data/yxk/K-data/K/fllm-sm/sim/3m_with_base/pointcloud \
  --out_root demo_out/3m_with_base \
  --opacity_thresh 0.005 \
  --final_num_points 20000

'''