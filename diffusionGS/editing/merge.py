"""
Merge two GaussianModel objects: geometry from geo_gs, color from color_gs.
"""

import copy

import torch


def merge_gaussians(geo_gs, color_gs):
    """
    Return a new GaussianModel whose spatial structure (xyz, scaling, rotation,
    opacity) comes from geo_gs and whose color (_features_dc / _features_rest)
    comes from color_gs.

    If the filtered Gaussian counts differ between the two runs, colors are
    transferred via nearest-neighbor matching in 3D space.
    """
    merged = copy.deepcopy(geo_gs)

    n_geo = geo_gs._xyz.shape[0]
    n_color = color_gs._xyz.shape[0]

    if n_geo == n_color:
        merged._features_dc = color_gs._features_dc.clone()
        if geo_gs._features_rest is not None and color_gs._features_rest is not None:
            merged._features_rest = color_gs._features_rest.clone()
        return merged

    # Nearest-neighbor color transfer (chunked to stay within VRAM)
    device = geo_gs._xyz.device
    geo_xyz = geo_gs._xyz                      # [N, 3]
    color_xyz = color_gs._xyz.to(device)       # [M, 3]
    color_dc = color_gs._features_dc.to(device)

    chunk = 4096
    nearest_idx = torch.zeros(n_geo, dtype=torch.long, device=device)
    for i in range(0, n_geo, chunk):
        dists = torch.cdist(
            geo_xyz[i : i + chunk].unsqueeze(0),
            color_xyz.unsqueeze(0),
        )[0]
        nearest_idx[i : i + chunk] = dists.argmin(dim=1)

    merged._features_dc = color_dc[nearest_idx].clone()
    if geo_gs._features_rest is not None and color_gs._features_rest is not None:
        color_rest = color_gs._features_rest.to(device)
        merged._features_rest = color_rest[nearest_idx].clone()

    return merged
