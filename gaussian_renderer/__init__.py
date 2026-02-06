#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Modified to use AMD-optimized gsplat rasterization backend instead of
# the original diff_gaussian_rasterization (langsplat-rasterization).

import math

import torch
import gsplat
from scene.gaussian_model import GaussianModel


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    opt,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene using gsplat's AMD-optimized rasterization backend.

    Background tensor (bg_color) must be on GPU!
    """

    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # --- Camera parameter conversion ---
    # gsplat uses intrinsics matrix Ks [C, 3, 3] instead of tanfov scalars.
    focal_x = W / (2.0 * tanfovx)
    focal_y = H / (2.0 * tanfovy)
    Ks = torch.tensor(
        [[focal_x, 0.0, W / 2.0], [0.0, focal_y, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)  # [1, 3, 3]

    # gsplat uses world-to-camera viewmats [C, 4, 4].
    # The original 3DGS stores world_view_transform in column-major (OpenGL)
    # convention inside a row-major PyTorch tensor, so we transpose to get the
    # actual world-to-camera matrix that gsplat expects.
    viewmats = viewpoint_camera.world_view_transform.T.unsqueeze(0)  # [1, 4, 4]

    # --- Gaussian parameters ---
    means3D = pc.get_xyz  # [N, 3]
    opacity = pc.get_opacity.squeeze(-1)  # [N] -- gsplat expects 1-D

    # Scales/rotations or precomputed covariance
    scales = None
    rotations = None
    covars = None
    if pipe.compute_cov3D_python:
        covars = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # --- Colors (SH coefficients or precomputed) ---
    sh_degree = None
    if override_color is not None:
        colors = override_color  # [N, 3]
    elif pipe.convert_SHs_python:
        # Evaluate SH in Python (same path the original LangSplat supports)
        from utils.sh_utils import eval_sh

        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)  # [N, 3]
    else:
        # Let gsplat evaluate SH inside its optimised CUDA kernel
        colors = pc.get_features  # [N, K, 3] SH coefficients
        sh_degree = pc.active_sh_degree

    # --- Language features (optional, e.g. CLIP embeddings for LangSplat) ---
    language_features = None
    if opt.include_feature:
        lf = pc.get_language_feature
        language_features = lf / (lf.norm(dim=-1, keepdim=True) + 1e-9)  # [N, 3]

    # --- Backgrounds ---
    # gsplat expects [..., C, D] shaped backgrounds.
    backgrounds = bg_color.unsqueeze(0) if bg_color is not None else None  # [1, 3]

    # --- Call gsplat rasterization ---
    # packed=False so that outputs are [B, C, N, ...] shaped, which makes it
    # straightforward to map back to per-Gaussian densification stats.
    render_colors, render_alphas, meta = gsplat.rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        tile_size=8,  # AMD-optimised default (vs 16 in original)
        backgrounds=backgrounds,
        packed=False,
        covars=covars,
        language_features=language_features,
    )

    # --- Output format conversion ---
    # gsplat returns [1, H, W, D] (channels-last); LangSplat expects [D, H, W].
    rendered_image = render_colors[0].permute(2, 0, 1)  # [3, H, W]

    if language_features is not None and "render_language_features" in meta:
        language_feature_image = meta["render_language_features"][0].permute(
            2, 0, 1
        )  # [D_lang, H, W]
    else:
        language_feature_image = torch.zeros((1,), device="cuda")

    # --- Radii conversion ---
    # gsplat returns [1, N, 2] (radius_x, radius_y); LangSplat expects scalar [N].
    radii = meta["radii"].squeeze(0)  # [N, 2]
    radii_scalar = radii.max(dim=-1).values  # [N]

    # --- Viewspace points for densification ---
    # LangSplat's add_densification_stats() reads viewspace_point_tensor.grad[:, :2].
    # gsplat's means2d ([1, N, 2]) carries gradients through autograd.
    # We create a [N, 2] leaf proxy and, when means2d requires grad, use a
    # backward hook to copy gradients into the proxy during loss.backward().
    means2d_orig = meta["means2d"]  # [1, N, 2]
    viewspace_points = torch.zeros(means3D.shape[0], 2, device="cuda", requires_grad=True)
    viewspace_points.retain_grad()

    if means2d_orig.requires_grad:
        def _copy_grad(grad):
            # grad has shape [1, N, 2]; squeeze and write to the proxy
            viewspace_points.grad = grad.squeeze(0).detach()
            return grad

        means2d_orig.register_hook(_copy_grad)

    return {
        "render": rendered_image,
        "language_feature_image": language_feature_image,
        "viewspace_points": viewspace_points,
        "visibility_filter": radii_scalar > 0,
        "radii": radii_scalar,
    }
