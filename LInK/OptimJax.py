import jax
import numpy as np
from .Solver import solve_rev_vectorized_batch_jax as solve_rev_vectorized_batch

import jax.numpy as jnp
import torch
import torch.nn as nn
from .BFGS_jax import Batch_BFGS

import matplotlib.pyplot as plt
import time

from tqdm.autonotebook import trange, tqdm
from .Visulization import draw_mechanism
import gradio as gr


# cosine_search和cosine_search_jax函数：在嵌入空间中通过余弦相似度检索与目标曲线最相似的候选曲线。
@torch.compile
def cosine_search(target_emb, atlas_emb, max_batch_size=1000000, ids=None):
    z = nn.functional.normalize(target_emb.unsqueeze(0).tile([max_batch_size, 1]))

    if ids is None:
        ids = torch.arange(atlas_emb.shape[0]).to(target_emb.device).long()

    sim = []
    for i in range(int(np.ceil(ids.shape[0] / max_batch_size))):
        z1 = atlas_emb[ids[i * max_batch_size:(i + 1) * max_batch_size]]
        sim.append(nn.functional.cosine_similarity(z1, z[0:z1.shape[0]]))
    sim = torch.cat(sim, 0)

    return ids[(-sim).argsort()], sim


@jax.jit
def cosine_search_jax(target_emb, atlas_emb, max_batch_size=1000000, ids=None):
    z = target_emb / jax.numpy.linalg.norm(target_emb)

    if ids is None:
        ids = jax.numpy.arange(atlas_emb.shape[0])

    sim = []
    for i in range(int(np.ceil(ids.shape[0] / max_batch_size))):
        z1 = atlas_emb[ids[i * max_batch_size:(i + 1) * max_batch_size]]
        sim.append(jax.numpy.sum(z1 * z[None], axis=-1) / jax.numpy.linalg.norm(z1, axis=-1) / jax.numpy.linalg.norm(z))
    sim = jax.numpy.concatenate(sim, 0)

    return ids[jax.numpy.argsort(-sim)], sim


# uniformize函数：对输入曲线进行均匀重采样，确保曲线点间隔一致。
def uniformize(curves, n):
    l = jax.numpy.cumsum(
        jax.numpy.pad(jax.numpy.linalg.norm(curves[:, 1:, :] - curves[:, :-1, :], axis=-1), ((0, 0), (1, 0))), axis=-1)
    l = l / l[:, -1].reshape(-1, 1)

    sampling = jax.numpy.linspace(-1e-6, 1 - 1e-6, n)
    end_is = jax.vmap(lambda a: jax.numpy.searchsorted(a.reshape(-1), sampling)[1:])(l)

    end_ids = end_is

    l_end = l[jax.numpy.arange(end_is.shape[0]).reshape(-1, 1), end_is]
    l_start = l[jax.numpy.arange(end_is.shape[0]).reshape(-1, 1), end_is - 1]
    ws = (l_end - sampling[1:].reshape(1, -1)) / (l_end - l_start)

    end_gather = curves[jax.numpy.arange(end_ids.shape[0]).reshape(-1, 1), end_ids]
    start_gather = curves[jax.numpy.arange(end_ids.shape[0]).reshape(-1, 1), end_ids - 1]

    uniform_curves = jax.numpy.concatenate(
        [curves[:, 0:1, :], (end_gather - (end_gather - start_gather) * ws[:, :, None])], 1)

    return uniform_curves

def uniformize_multi(curves: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    将 (B, N, T, 2) 的曲线 uniformize 为固定长度 n。
    """
    B, N, T, D = curves.shape

    # 计算累积距离
    diffs = curves[:, :, 1:, :] - curves[:, :, :-1, :]  # (B, N, T-1, 2)
    dists = jnp.linalg.norm(diffs, axis=-1)  # (B, N, T-1)
    dists = jnp.pad(dists, ((0, 0), (0, 0), (1, 0)))  # pad 前置 0
    cum_dists = jnp.cumsum(dists, axis=-1)  # (B, N, T)
    cum_dists /= cum_dists[:, :, -1:].clip(min=1e-8)  # 归一化到 [0, 1]

    # 目标采样点
    sampling = jnp.linspace(-1e-6, 1 - 1e-6, n)  # (n,)

    def interp_single(c, l):
        idxs = jnp.searchsorted(l, sampling)  # (n,)
        idxs = jnp.clip(idxs, 1, T - 1)
        l_start = l[idxs - 1]
        l_end = l[idxs]
        w = (l_end - sampling) / (l_end - l_start + 1e-8)

        pt_start = c[idxs - 1]
        pt_end = c[idxs]
        return pt_end - (pt_end - pt_start) * w[:, None]  # (n, 2)

    vmap_interp = jax.vmap(jax.vmap(interp_single, in_axes=(0, 0)), in_axes=(0, 0))
    return vmap_interp(curves, cum_dists)  # (B, N, n, 2)

# 计算两个点x和y之间的欧几里得距离。
@jax.jit
def _euclidean_distance(x, y) -> float:
    dist = jax.numpy.sqrt(jax.numpy.sum((x - y) ** 2))
    return dist


# 计算两个二维数组a和b中所有点对之间的欧几里得距离，生成一个距离矩阵。
@jax.jit
def cdist(a, b):
    """Jax implementation of :func:`scipy.spatial.distance.cdist`.

    Uses euclidean distance.

    Parameters
    ----------
    x
        Array of shape (n_cells_a, n_features)
    y
        Array of shape (n_cells_b, n_features)

    Returns
    -------
    dist
        Array of shape (n_cells_a, n_cells_b)
    """
    return jax.vmap(lambda x, y: jax.vmap(lambda x1: jax.vmap(lambda y1: _euclidean_distance(x1, y1))(y))(x))(a, b)


# 计算两个点集c1和c2之间的批量Chamfer距离。
@jax.jit
def batch_chamfer_distance(c1, c2):
    d = cdist(c1, c2)
    id1 = d.argmin(1)
    id2 = d.argmin(2)

    d1 = jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(id1.shape[0]).reshape(-1, 1), id1], axis=-1).mean(1)
    d2 = jax.numpy.linalg.norm(c1 - c2[jax.numpy.arange(id2.shape[0]).reshape(-1, 1), id2], axis=-1).mean(1)

    return d1 + d2

@jax.jit
def multi_batch_chamfer_distance(c1: jnp.ndarray, c2: jnp.ndarray) -> jnp.ndarray:
    """
    更高效地计算 Batched Chamfer 距离。
    输入:
        c1: (B, N, T, D)
        c2: (B, N, T, D)
    输出:
        (B, N)
    """
    # c1 → c2 最近点距离
    dists_1 = jnp.linalg.norm(c1[:, :, :, None, :] - c2[:, :, None, :, :], axis=-1)  # (B, N, T, T)
    min_dists_1 = dists_1.min(axis=-1).mean(axis=-1)  # (B, N)

    # c2 → c1 最近点距离
    dists_2 = jnp.linalg.norm(c2[:, :, :, None, :] - c1[:, :, None, :, :], axis=-1)  # (B, N, T, T)
    min_dists_2 = dists_2.min(axis=-1).mean(axis=-1)  # (B, N)

    return min_dists_1 + min_dists_2  # (B, N)

# 计算两个点集c1和c2之间的批量有序距离，考虑点的顺序（顺时针和逆时针）。
@jax.jit
def batch_ordered_distance(c1, c2):
    C = cdist(c2, c1)

    row_ind = jax.numpy.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:, None] - jax.numpy.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:, None] - row_ind].T

    col_inds_ccw = jax.numpy.copy(col_inds[:, ::-1])

    row_inds = row_inds
    col_inds = col_inds
    col_inds_ccw = col_inds_ccw

    argmin_cw = jax.numpy.argmin(C[:, row_inds, col_inds].sum(2), axis=1)
    argmin_ccw = jax.numpy.argmin(C[:, row_inds, col_inds_ccw].sum(2), axis=1)

    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]

    ds_cw = jax.numpy.square(
        jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_cw.shape[0]).reshape(-1, 1), col_ind_cw], axis=-1)).mean(
        1)
    ds_ccw = jax.numpy.square(
        jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_ccw.shape[0]).reshape(-1, 1), col_ind_ccw],
                              axis=-1)).mean(1)
    ds = jax.numpy.minimum(ds_cw, ds_ccw)

    return ds

@jax.jit
def multi_batch_ordered_distance(c1: jnp.ndarray, c2: jnp.ndarray) -> jnp.ndarray:
    """
    高效计算多 batch 的 ordered distance，考虑时间轴平移（顺/逆时针）。
    输入:
        c1, c2: (B, N, T, D)
    输出:
        (B, N)
    """
    B, N, T, D = c1.shape
    shifts = jnp.arange(T)

    def single_ordered(c1_curve, c2_curve):
        def mse_shift(shift):
            c2_roll_cw = jnp.roll(c2_curve, shift, axis=0)
            c2_roll_ccw = jnp.roll(c2_curve, -shift, axis=0)
            mse_cw = jnp.mean((c1_curve - c2_roll_cw) ** 2)
            mse_ccw = jnp.mean((c1_curve - c2_roll_ccw) ** 2)
            return jnp.minimum(mse_cw, mse_ccw)
        return jnp.min(jax.vmap(mse_shift)(shifts))

    return jax.vmap(  # over batch
        lambda c1_b, c2_b: jax.vmap(single_ordered)(c1_b, c2_b)
    )(c1, c2)  # -> (B, N)

# 计算两个点集（如曲线）之间的有序距离，考虑点的顺序（顺时针和逆时针），并返回最小的那个距离。
@jax.jit
def ordered_objective_batch(c1, c2):
    C = cdist(c2, c1)

    row_ind = jax.numpy.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:, None] - jax.numpy.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:, None] - row_ind].T

    col_inds_ccw = jax.numpy.copy(col_inds[:, ::-1])

    row_inds = row_inds
    col_inds = col_inds
    col_inds_ccw = col_inds_ccw

    argmin_cw = jax.numpy.argmin(C[:, row_inds, col_inds].sum(2), axis=1)
    argmin_ccw = jax.numpy.argmin(C[:, row_inds, col_inds_ccw].sum(2), axis=1)

    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]

    ds_cw = jax.numpy.square(
        jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_cw.shape[0]).reshape(-1, 1), col_ind_cw], axis=-1)).mean(
        1)
    ds_ccw = jax.numpy.square(
        jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_ccw.shape[0]).reshape(-1, 1), col_ind_ccw],
                              axis=-1)).mean(1)
    ds = jax.numpy.minimum(ds_cw, ds_ccw)

    return ds * 2 * jax.numpy.pi


# 找到将输入曲线in_curves变换到目标曲线target_curve的最佳平移、缩放和旋转变换。
@jax.jit
def find_transforms(in_curves, target_curve, n_angles=100):
    objective_fn = batch_ordered_distance
    translations = in_curves.mean(1)
    # center curves
    curves = in_curves - translations[:, None]

    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1) / in_curves.shape[1])
    curves = curves / s[:, None, None]

    # find best rotation for each curve
    test_angles = jax.numpy.linspace(0, 2 * jax.numpy.pi, n_angles)
    R = jax.numpy.zeros([n_angles, 2, 2])
    R = jax.numpy.hstack(
        [jax.numpy.cos(test_angles)[:, None], -jax.numpy.sin(test_angles)[:, None], jax.numpy.sin(test_angles)[:, None],
         jax.numpy.cos(test_angles)[:, None]]).reshape(-1, 2, 2)

    R = R[None, :, :, :].repeat(curves.shape[0], 0)
    R = R.reshape(-1, 2, 2)

    # rotate curves
    curves = curves[:, None, :, :].repeat(n_angles, 1)
    curves = curves.reshape(-1, curves.shape[-2], curves.shape[-1])

    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves, (0, 2, 1))), (0, 2, 1))

    # find best rotation by measuring cdist to target curve
    target_curve = target_curve[None, :, :].repeat(curves.shape[0], 0)
    # cdist = torch.cdist(curves, target_curve)

    # # chamfer distance
    cdist = objective_fn(curves, target_curve)
    cdist = cdist.reshape(-1, n_angles)
    best_rot_idx = cdist.argmin(-1)
    best_rot = test_angles[best_rot_idx]

    return translations, s, best_rot


# 将平移、缩放和旋转变换应用于曲线。
@jax.jit
def apply_transforms(curves, translations, scales, rotations):
    curves = curves * scales[:, None, None]
    R = jax.numpy.zeros([rotations.shape[0], 2, 2])
    R = jax.numpy.hstack(
        [jax.numpy.cos(rotations)[:, None], -jax.numpy.sin(rotations)[:, None], jax.numpy.sin(rotations)[:, None],
         jax.numpy.cos(rotations)[:, None]]).reshape(-1, 2, 2)
    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves, (0, 2, 1))), (0, 2, 1))
    curves = curves + translations[:, None]
    return curves

def apply_transforms_multi(curves, translations, scales, rotations):
    """
    curves: shape (B, N, T, 2)
    translations: (B, 2)
    scales: (B,)
    rotations: (B,)
    """
    B, N, T, _ = curves.shape

    # 缩放
    curves = curves * scales[:, None, None, None]  # (B, N, T, 2)

    # 旋转矩阵 (B, 2, 2)
    cos_theta = jax.numpy.cos(rotations)
    sin_theta = jax.numpy.sin(rotations)
    R = jax.numpy.stack([
        jax.numpy.stack([cos_theta, -sin_theta], axis=-1),
        jax.numpy.stack([sin_theta,  cos_theta], axis=-1)
    ], axis=-2)  # shape (B, 2, 2)

    # 旋转：需要 reshape -> matmul -> reshape back
    curves = jax.numpy.einsum('bij,bntj->bnti', R, curves)  # (B, N, T, 2)

    # 平移
    curves = curves + translations[:, None, None, :]  # (B, N, T, 2)

    return curves

# 对曲线进行标准化预处理，包括均匀采样、中心化、缩放标准化和方向对齐。
def preprocess_curves(curves, n=200):
    # equidistant sampling (Remove Timing) 均匀化采样
    curves = uniformize(curves, n)

    # center curves 中心化曲线
    curves = curves - curves.mean(1)[:, None]

    # apply uniform scaling 统一缩放
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1) / n)[:, None, None]
    curves = curves / s

    # find the furthest point on the curve 找到曲线上的最远点
    max_idx = jax.numpy.square(curves).sum(-1).argmax(axis=1)

    # rotate curves so that the furthest point is horizontal 旋转曲线
    theta = -jax.numpy.arctan2(curves[jax.numpy.arange(curves.shape[0]), max_idx, 1],
                               curves[jax.numpy.arange(curves.shape[0]), max_idx, 0])

    # normalize the rotation
    R = jax.numpy.hstack([jax.numpy.cos(theta)[:, None], -jax.numpy.sin(theta)[:, None], jax.numpy.sin(theta)[:, None],
                          jax.numpy.cos(theta)[:, None]])
    R = R.reshape(-1, 2, 2)

    # curves = torch.bmm(R,curves.transpose(1,2)).transpose(1,2)
    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves, (0, 2, 1))), (0, 2, 1))

    return curves


def preprocess_multi_curves_as_whole(curves):
    n_curve, n_points, _ = curves.shape
    all_points = curves.reshape(-1, 2)  # (n_curve * n_points, 2)

    # center
    all_points -= all_points.mean(0)

    # scale
    s = jax.numpy.sqrt(jax.numpy.square(all_points).sum() / all_points.shape[0])
    all_points /= s

    # find furthest point
    dists = jax.numpy.square(all_points).sum(-1)
    max_idx = dists.argmax()
    furthest = all_points[max_idx]

    # rotate
    theta = -jax.numpy.arctan2(furthest[1], furthest[0])
    R = jax.numpy.array([
        [jax.numpy.cos(theta), -jax.numpy.sin(theta)],
        [jax.numpy.sin(theta), jax.numpy.cos(theta)],
    ])
    all_points = jax.numpy.dot(all_points, R.T)

    return all_points.reshape(n_curve, n_points, 2)


@jax.jit
def get_scales(curves):
    return jax.numpy.sqrt(jax.numpy.square(curves - curves.mean(1)).sum(-1).sum(-1) / curves.shape[1])


@jax.jit
def get_multi_scales(curves):
    # curves: (n, 200, 2)
    mean = curves.mean(axis=1, keepdims=True)  # 每条曲线的中心点 (n, 1, 2)
    centered = curves - mean  # 平移到原点 (n, 200, 2)
    scale = jax.numpy.sqrt(jax.numpy.square(centered).sum(axis=-1).sum(axis=-1) / curves.shape[1])  # (n,)
    return scale


# 计算目标曲线与由机构参数生成的曲线之间的目标函数值，结合了Chamfer距离和有序距离。
@jax.jit
def blind_objective_batch(curve, As, x0s, node_types, curve_size=200,
                          thetas=jax.numpy.linspace(0.0, 2 * jax.numpy.pi, 2000), CD_weight=1.0, OD_weight=1.0,
                          idxs=None):
    curve = preprocess_curves(curve[None], curve_size)[0]

    sol = solve_rev_vectorized_batch(As, x0s, node_types, thetas)

    if idxs is None:
        idxs = (As.sum(-1) > 0).sum(-1) - 1
    current_sol = sol[jax.numpy.arange(sol.shape[0]), idxs]

    # find nans at axis 0 level
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))
    best_matches_masked = current_sol * good_idx[:, None, None]
    current_sol_r_masked = current_sol * ~good_idx[:, None, None]
    current_sol = uniformize(current_sol, current_sol.shape[1])
    current_sol = current_sol * good_idx[:, None, None] + current_sol_r_masked

    dummy = uniformize(curve[None], thetas.shape[0])[0]
    best_matches_r_masked = dummy[None].repeat(best_matches_masked.shape[0], 0) * ~good_idx[:, None, None]
    best_matches = best_matches_masked + best_matches_r_masked
    best_matches = uniformize(best_matches, curve.shape[0])

    tr, sc, an = find_transforms(best_matches, curve)
    tiled_curves = curve[None, :, :].repeat(best_matches.shape[0], 0)
    tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)

    OD = batch_ordered_distance(
        current_sol[:, jax.numpy.linspace(0, current_sol.shape[1] - 1, tiled_curves.shape[1]).astype(int), :] / sc[:,
                                                                                                                None,
                                                                                                                None],
        tiled_curves / sc[:, None, None])
    CD = batch_chamfer_distance(current_sol / sc[:, None, None], tiled_curves / sc[:, None, None])

    objective_function = CD_weight * CD + OD_weight * OD

    return objective_function


# 创建一个批量优化目标函数，用于优化机械机构的参数。
def make_batch_optim_obj(curve, As, x0s, node_types, timesteps=2000, CD_weight=1.0, OD_weight=1.0, start_theta=0.0,
                         end_theta=2 * jax.numpy.pi):
    thetas = jax.numpy.linspace(start_theta, end_theta, timesteps)
    # 计算机械机构在各个时间步长下的位置轨迹
    sol = solve_rev_vectorized_batch(As, x0s, node_types, thetas)

    # 获取终点轨迹
    idxs = (As.sum(-1) > 0).sum(-1) - 1
    best_matches = sol[jax.numpy.arange(sol.shape[0]), idxs]

    # 掩膜无效的匹配
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(best_matches.sum(-1).sum(-1)))
    best_matches_masked = best_matches * good_idx[:, None, None]
    best_matches_r_masked = best_matches[good_idx][0][None].repeat(best_matches.shape[0], 0) * ~good_idx[:, None, None]
    best_matches = best_matches_masked + best_matches_r_masked
    best_matches = uniformize(best_matches, curve.shape[0])

    # 计算变换（平移、缩放和旋转）
    tr, sc, an = find_transforms(best_matches, curve, )
    tiled_curves = curve[None, :, :].repeat(best_matches.shape[0], 0)
    tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)

    # def objective(x0s_current):
    #     current_x0 = x0s_current
    #     sol = solve_rev_vectorized_batch(As,current_x0,node_types,thetas)
    #     current_sol = sol[jax.numpy.arange(sol.shape[0]),idxs]

    #     #find nans at axis 0 level
    #     good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))

    #     current_sol_r_masked = current_sol * ~good_idx[:,None,None]
    #     current_sol = uniformize(current_sol, current_sol.shape[1])
    #     current_sol = current_sol * good_idx[:,None,None] + current_sol_r_masked

    #     OD = batch_ordered_distance(current_sol[:,jax.numpy.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1]).astype(int),:]/sc[:,None,None],tiled_curves/sc[:,None,None])
    #     CD = batch_chamfer_distance(current_sol/sc[:,None,None],tiled_curves/sc[:,None,None])
    #     objective_function = CD_weight* CD + OD_weight * OD

    #     objective_function = jax.numpy.where(jax.numpy.isnan(objective_function),1e6,objective_function)

    #     return objective_function

    # def get_sum(x0s_current):
    #     obj = objective(x0s_current)
    #     return obj.sum(), obj

    # def final(x0s_current):
    #     fn = jax.jit(jax.value_and_grad(get_sum,has_aux=True))

    #     val,grad = fn(x0s_current)

    #     val = jax.numpy.nan_to_num(val[1],nan=1e6)
    #     grad = jax.numpy.nan_to_num(grad,nan=0)

    #     return val,grad

    fn = lambda x0s_current: final(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight,
                                   OD_weight)

    return fn

def make_multi_batch_optim_obj(curve, As, x0s, node_types, timesteps=2000, CD_weight=1.0, OD_weight=1.0, start_theta=0.0,
                         end_theta=2 * jax.numpy.pi):
    thetas = jax.numpy.linspace(start_theta, end_theta, timesteps)
    # 计算机械机构在各个时间步长下的位置轨迹
    sol = solve_rev_vectorized_batch(As, x0s, node_types, thetas)

    # 提取对应节点的轨迹
    idxs_end = (As.sum(-1) > 0).sum(-1) - 1  # 终点索引
    idxs_neighbors = idxs_end[:, None] - jax.numpy.array([1, 2])  # 相邻点的索引（往前数1和2）

    # 针对不同batch做索引
    batch_indices = jax.numpy.arange(sol.shape[0])

    # 取终点轨迹
    traj_end = sol[batch_indices[:, None], idxs_end[:, None]]  # shape (batch_size, 1, 3)
    # 取相邻节点轨迹（注意要防止idx越界）
    traj_neighbors = sol[batch_indices[:, None, None], idxs_neighbors[:, :, None]]  # shape (batch_size, 2, 3)
    # 扩展 traj_neighbors 以匹配 traj_end 的维度
    traj_neighbors_expanded = jax.numpy.squeeze(traj_neighbors, axis=2)  # 去掉第3个维度，变为 (300, 2, 2000, 2)

    # 现在可以进行拼接
    all_trajs = jax.numpy.concatenate([traj_end, traj_neighbors_expanded], axis=1)  # 形状 (300, 3, 2000, 2)

    # 处理NaN（掩膜无效轨迹）
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(all_trajs.sum(-1).sum(-1).sum(-1)))
    all_trajs_masked = all_trajs * good_idx[:, None, None, None]
    all_trajs_r_masked = all_trajs[good_idx][0][None].repeat(all_trajs.shape[0], 0) * ~good_idx[:, None, None, None]
    all_trajs = all_trajs_masked + all_trajs_r_masked
    # all_trajs.shape == (B, 3, 2000, 2)
    N = 200  # 统一为200点
    B = all_trajs.shape[0]
    new_all_trajs = []

    for b in range(B):
        trajs = []
        for k in range(3):  # 3条轨迹：终点+邻居
            traj_uniform = uniformize(all_trajs[b, k][None, :, :], N)[0]  # (200, 2)
            trajs.append(traj_uniform)
        new_all_trajs.append(jax.numpy.stack(trajs, axis=0))  # (3, 200, 2)

    all_trajs_uniform = jax.numpy.stack(new_all_trajs, axis=0)  # (B, 3, 200, 2)

    # 计算变换，让轨迹和目标curve对齐（分别对终点轨迹和相邻点轨迹）
    tr, sc, an = find_transforms(all_trajs_uniform[:, 0], curve[0])  # 用终点轨迹对齐
    tiled_curves = curve[None, :, :, :].repeat(all_trajs_uniform.shape[0], 0)  # shape: (batch_size, n, 200, 3)
    tiled_curves = apply_transforms_multi(tiled_curves, tr, sc, -an)

    # 定义优化目标
    def fn(x0s_current):
        return final_multi(x0s_current, As, node_types, curve, thetas, idxs_end, idxs_neighbors, sc, tiled_curves,
                           CD_weight, OD_weight)

    return fn


# 计算目标函数值，用于评估当前机构状态生成的曲线与目标曲线之间的匹配程度。
def objective(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    current_x0 = x0s_current
    sol = solve_rev_vectorized_batch(As, current_x0, node_types, thetas)
    current_sol = sol[jax.numpy.arange(sol.shape[0]), idxs]

    # find nans at axis 0 level
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))

    current_sol_r_masked = current_sol * ~good_idx[:, None, None]
    current_sol = uniformize(current_sol, current_sol.shape[1])
    current_sol = current_sol * good_idx[:, None, None] + current_sol_r_masked

    OD = batch_ordered_distance(
        current_sol[:, jax.numpy.linspace(0, current_sol.shape[1] - 1, tiled_curves.shape[1]).astype(int), :] / sc[:,
                                                                                                                None,
                                                                                                                None],
        tiled_curves / sc[:, None, None])
    CD = batch_chamfer_distance(current_sol / sc[:, None, None], tiled_curves / sc[:, None, None])
    objective_function = CD_weight * CD + OD_weight * OD

    objective_function = jax.numpy.where(jax.numpy.isnan(objective_function), 1e6, objective_function)

    return objective_function

def objective_multi(x0s_current, As, node_types, curve, thetas,
                    idxs_end, idxs_neighbors, sc, tiled_curves,
                    CD_weight, OD_weight):

    current_x0 = x0s_current
    sol = solve_rev_vectorized_batch(As, current_x0, node_types, thetas)  # (B, n_nodes, T, 3)
    B = sol.shape[0]

    # === 提取终点及邻居轨迹 ===
    sol_end = sol[jnp.arange(B), idxs_end]  # (B, T, 3)
    sol_neighbors = sol[jnp.arange(B)[:, None], idxs_neighbors]  # (B, N-1, T, 3)
    sol_all = jnp.concatenate([sol_end[:, None, :, :], sol_neighbors], axis=1)  # (B, N, T, 3)

    # === mask 掉 NaN 批次 ===
    valid_mask = ~jnp.isnan(sol_all).sum(axis=(1, 2, 3)).astype(bool)  # (B,)
    sol_all = jnp.where(valid_mask[:, None, None, None], sol_all, 0.0)

    # === 向量化 uniformize ===
    # 将 (B, N, T, 3) → (B*N, T, 3)，只对前两维拼接处理
    BN = B * sol_all.shape[1]
    flat_curves = sol_all.reshape((BN, -1, 2))  # (B*N, T, 2)
    flat_curves_2d = flat_curves[:, :, :2]

    uniform_2d = uniformize(flat_curves_2d, 200)  # (B*N, 200, 2)
    uniform_z = flat_curves[:, :200, 2:]  # 截断 Z，(B*N, 200, 1)
    uniform_3d = jnp.concatenate([uniform_2d, uniform_z], axis=-1)  # (B*N, 200, 2)

    # reshape 回 (B, N, T, 2)
    sol_all = uniform_3d.reshape((B, -1, 200, 2))
    tiled_curves = tiled_curves[:, :, :200, :]

    # === Normalize ===
    sol_all = sol_all / sc[:, None, None, None]
    tiled_curves = tiled_curves / sc[:, None, None, None]

    # === 计算 CD 和 OD ===
    CD = multi_batch_chamfer_distance(sol_all, tiled_curves)
    OD = multi_batch_ordered_distance(sol_all, tiled_curves)

    # === 聚合 ===
    loss = CD_weight * CD + OD_weight * OD  # (B, N)
    loss = jnp.where(valid_mask[:, None], loss, 1e6)
    loss = jnp.mean(loss, axis=-1)  # (B,)

    return loss


# 计算目标函数值的总和及其辅助值。
def get_sum(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    obj = objective(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight)
    return obj.sum(), obj

def get_sum_multi(x0s_current, As, node_types, curve, thetas, idxs_end, idxs_neighbors, sc, tiled_curves, CD_weight, OD_weight):
    obj = objective_multi(x0s_current, As, node_types, curve, thetas, idxs_end, idxs_neighbors, sc, tiled_curves, CD_weight, OD_weight)
    return obj.sum(), obj

fn = jax.jit(jax.value_and_grad(get_sum, has_aux=True))
fn_multi = jax.jit(jax.value_and_grad(get_sum_multi, has_aux=True))

# 计算目标函数值及其梯度。
def final(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    val, grad = fn(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight)

    val = jax.numpy.nan_to_num(val[1], nan=1e6)
    grad = jax.numpy.nan_to_num(grad, nan=0)

    return val, grad

def final_multi(x0s_current, As, node_types, curve, thetas, idxs_end, idxs_neighbors, sc, tiled_curves, CD_weight, OD_weight):
    val, grad = fn_multi(x0s_current, As, node_types, curve, thetas, idxs_end, idxs_neighbors, sc, tiled_curves, CD_weight, OD_weight)

    val = jax.numpy.nan_to_num(val[1], nan=1e6)
    grad = jax.numpy.nan_to_num(grad, nan=0)

    return val, grad

# 对曲线进行平滑处理，减少噪声。
def smooth_hand_drawn_curves(curves, n=200, n_freq=5):
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves, n)

    # center curves
    curves = curves - curves.mean(1)[:, None]

    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1) / n)[:, None, None]
    curves = curves / s

    # reduce with fft
    curves = jax.numpy.concatenate([jax.numpy.real(
        jax.numpy.fft.ifft(jax.numpy.fft.fft(curves[:, :, 0], axis=1)[:, 0:n_freq], n=n, axis=1))[:, :, None],
                                    jax.numpy.real(
                                        jax.numpy.fft.ifft(jax.numpy.fft.fft(curves[:, :, 1], axis=1)[:, 0:n_freq], n=n,
                                                           axis=1))[:, :, None]], axis=2)

    return preprocess_curves(curves, n)


def smooth_hand_drawn_multi_curves(curves, n=200, n_freq=5):
    smoothed_curves = []

    for i in range(curves.shape[0]):
        curve = curves[i][None, ...]  # 取出一条曲线，保持 batch_dim 方便 uniformize

        # Step 1: 均匀采样
        curve = uniformize(curve, n)

        # Step 2: 居中
        curve = curve - curve.mean(1)[:, None]

        # Step 3: 缩放
        s = jax.numpy.sqrt(jax.numpy.square(curve).sum(-1).sum(-1) / n)[:, None, None]
        curve = curve / s

        # Step 4: 低频滤波
        curve = jax.numpy.concatenate([
            jax.numpy.real(jax.numpy.fft.ifft(jax.numpy.fft.fft(curve[:, :, 0], axis=1)[:, 0:n_freq], n=n, axis=1))[:,
            :, None],
            jax.numpy.real(jax.numpy.fft.ifft(jax.numpy.fft.fft(curve[:, :, 1], axis=1)[:, 0:n_freq], n=n, axis=1))[:,
            :, None]
        ], axis=2)

        # Step 5: 预处理
        curve = preprocess_curves(curve)

        smoothed_curves.append(curve[0])  # squeeze batch_dim

    curves = jax.numpy.stack(smoothed_curves, axis=0)
    return curves


# 更新优化过程中的进度信息。
def progerss_uppdater(x, prog=None):
    if prog is not None:
        prog.update(1)
        prog.set_postfix_str(f'Current Best CD: {x[1]:.7f}')


def demo_progress_updater(x, prog=None, desc=''):
    if prog is not None:
        prog(x[0], desc=desc + f'Current Best CD: {x[1]:.7f}')


class PathSynthesis:
    def __init__(self, trainer_instance, curves, As, x0s, node_types, precomputed_emb=None, optim_timesteps=2000,
                 top_n=300, init_optim_iters=10, top_n_level2=30, CD_weight=1.0, OD_weight=0.25, BFGS_max_iter=100,
                 n_repos=0, BFGS_lineserach_max_iter=10, BFGS_line_search_mult=0.5, butterfly_gen=200,
                 butterfly_pop=200, curve_size=200, smoothing=True, n_freq=5, device=None, sizes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if precomputed_emb is None:
            self.precomputed_emb = trainer_instance.compute_embeddings_base(curves, 1000)
        else:
            self.precomputed_emb = precomputed_emb

        self.curve_size = curve_size
        self.models = trainer_instance
        self.BFGS_max_iter = BFGS_max_iter
        self.BFGS_lineserach_max_iter = BFGS_lineserach_max_iter
        self.BFGS_line_search_mult = BFGS_line_search_mult
        self.butterfly_gen = butterfly_gen
        self.butterfly_pop = butterfly_pop
        self.smoothing = smoothing
        self.n_freq = n_freq
        self.curves = curves
        self.As = As
        self.x0s = x0s
        self.node_types = node_types
        self.top_n = top_n
        self.optim_timesteps = optim_timesteps
        self.init_optim_iters = init_optim_iters
        self.top_n_level2 = top_n_level2
        self.CD_weight = CD_weight
        self.OD_weight = OD_weight
        self.n_repos = n_repos

        if sizes is not None:
            self.sizes = sizes
        else:
            self.sizes = (As.sum(-1) > 0).sum(-1)

    def synthesize(self, target_curve, verbose=True, visualize=True, partial=False, max_size=20, save_figs=None):

        start_time = time.time()

        # target_curve = torch.tensor(target_curve).float().to(self.device)

        og_scale = get_scales(target_curve[None])[0]

        # 如果是部分曲线，扩展为闭合曲线
        if partial:
            size = target_curve.shape[0]
            # fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0]) / 2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = jax.numpy.linalg.norm(start_point - center)
            b = jax.numpy.linalg.norm(end_point - center)
            start_angle = jax.numpy.arctan2(start_point[1] - center[1], start_point[0] - center[0])
            end_angle = jax.numpy.arctan2(end_point[1] - center[1], end_point[0] - center[0])

            angles = jax.numpy.linspace(start_angle, end_angle, self.curve_size)
            ellipse = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)], 1)

            angles = jax.numpy.linspace(start_angle + 2 * np.pi, end_angle, self.curve_size)
            ellipse_2 = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)],
                                        1)

            # ellipse 1 length
            l_1 = jax.numpy.linalg.norm(ellipse - target_curve.mean(0), axis=-1).sum()
            # ellipse 2 length
            l_2 = jax.numpy.linalg.norm(ellipse_2 - target_curve.mean(0), axis=-1).sum()

            if l_1 > l_2:
                target_curve = jax.numpy.concatenate([target_curve, ellipse], 0)
            else:
                target_curve = jax.numpy.concatenate([target_curve, ellipse_2], 0)

        target_curve_copy = preprocess_curves(target_curve[None], self.curve_size)[0]
        target_curve_ = jax.numpy.copy(target_curve)

        # 平滑处理
        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve[None], n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve[None], self.curve_size)[0]

        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size][None], self.curve_size)[0]
            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), target_curve, )
            transformed_curve = apply_transforms(target_curve[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx + 1][None], self.curve_size)[0]

            target_uni = jax.numpy.copy(target_curve_copy)

            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), target_uni, )
            transformed_curve = apply_transforms(target_uni[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx + 1][None], self.curve_size)[0]

        # 可视化预处理结果
        if verbose:
            print('Curve preprocessing done')
            if visualize:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                if partial:
                    axs[0].plot(target_curve_[:size][:, 0], target_curve_[:size][:, 1], color="indigo")
                else:
                    axs[0].plot(target_curve_copy[:, 0], target_curve_copy[:, 1], color="indigo")
                axs[0].set_title('Original Curve')
                axs[0].axis('equal')
                axs[0].axis('off')

                axs[1].plot(target_curve[:, 0], target_curve[:, 1], color="indigo")
                axs[1].set_title('Preprocessed Curve')
                axs[1].axis('equal')
                axs[1].axis('off')

                if save_figs is not None:
                    fig.savefig(save_figs + '_preprocessing.png')
                else:
                    plt.show()

        input_tensor = target_curve[None]
        batch_padd = preprocess_curves(self.curves[np.random.choice(self.curves.shape[0], 255)])

        input_tensor = torch.tensor(np.concatenate([input_tensor, batch_padd], 0)).float().to(self.device)
        with torch.cuda.amp.autocast():
            target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        # target_emb = torch.tensor(target_emb).float().to(self.device)

        # 检索相似曲线
        ids = np.where(self.sizes <= max_size)[0]
        idxs, sim = cosine_search_jax(target_emb, self.precomputed_emb, ids=ids)

        # max batch size is 250
        tr, sc, an = [], [], []
        for i in range(int(np.ceil(self.top_n * 5 / 250))):
            tr_, sc_, an_ = find_transforms(self.curves[idxs[i * 250:(i + 1) * 250]], target_curve_copy, )
            tr.append(tr_)
            sc.append(sc_)
            an.append(an_)
        tr = jax.numpy.concatenate(tr, 0)
        sc = jax.numpy.concatenate(sc, 0)
        an = jax.numpy.concatenate(an, 0)

        # 计算距离矩阵
        tiled_curves = target_curve_copy[None].repeat(self.top_n * 5, 0)
        tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
        CD = batch_ordered_distance(tiled_curves / sc[:, None, None],
                                    self.curves[idxs[:self.top_n * 5]] / sc[:, None, None])

        # get best matches index
        tid = jax.numpy.argsort(CD)[:self.top_n]

        # 可视化最佳匹配曲线
        if verbose:
            print(f'Best ordered distance found in top {self.top_n} is {CD.min()}')
            if visualize:
                grid_size = int(np.ceil(self.top_n ** 0.5))
                fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
                for i in range(grid_size):
                    for j in range(grid_size):
                        if i * grid_size + j < self.top_n:
                            axs[i, j].plot(self.curves[idxs[tid][i * grid_size + j]][:, 0],
                                           self.curves[idxs[tid][i * grid_size + j]][:, 1], color='indigo')
                            axs[i, j].plot(tiled_curves[tid][i * grid_size + j][:, 0],
                                           tiled_curves[tid][i * grid_size + j][:, 1], color="darkorange", alpha=0.7)
                        axs[i, j].axis('off')
                        axs[i, j].axis('equal')

                if save_figs is not None:
                    fig.savefig(save_figs + '_retrieved.png')
                else:
                    plt.show()

        As = self.As[idxs[tid]]
        x0s = self.x0s[idxs[tid]]
        node_types = self.node_types[idxs[tid]]

        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        # 初始化优化
        # 这段代码通过两阶段的 BFGS 优化，逐步缩小候选解的范围并细化优化结果，最终找到与目标曲线最匹配的机械机构参数。
        if verbose:
            print('Starting initial optimization')
            prog = trange(self.init_optim_iters)
        else:
            prog = None

        x, f = Batch_BFGS(x0s, obj, max_iter=self.init_optim_iters, line_search_max_iter=self.BFGS_lineserach_max_iter,
                          tau=self.BFGS_line_search_mult, progress=lambda x: progerss_uppdater(x, prog),
                          threshhold=0.001)

        # 筛选第二级优化的顶级候选解：
        # top n level 2
        top_n_2 = f.argsort()[:self.top_n_level2]
        As = As[top_n_2]
        x0s = x[top_n_2]
        node_types = node_types[top_n_2]

        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        # 第二级优化
        if verbose:
            print('Starting second optimization stage')
            prog2 = trange(self.BFGS_max_iter)
        else:
            prog2 = None

        for i in range(self.n_repos):
            x, f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter // (self.n_repos + 1),
                              line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                              progress=lambda x: progerss_uppdater(x, prog2))
            if verbose:
                print('Re-Positioning')
            x0s = x

        # 找到最佳解
        x, f = Batch_BFGS(x0s, obj,
                          max_iter=self.BFGS_max_iter - self.n_repos * self.BFGS_max_iter // (self.n_repos + 1),
                          line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                          progress=lambda x: progerss_uppdater(x, prog2))
        best_idx = f.argmin()

        end_time = time.time()

        if verbose:
            print('Total time taken(s):', end_time - start_time)

        if verbose:
            print('Best chamfer distance found is', f.min())

        # 处理部分曲线
        if partial:
            target_uni = uniformize(target_curve_copy[None], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.optim_timesteps), target_uni, )
            transformed_curve = apply_transforms(target_uni[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))

            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            original_match = jax.numpy.copy(best_matches)
            best_matches = uniformize(best_matches, self.optim_timesteps)

            tr, sc, an = find_transforms(best_matches, target_uni, )
            tiled_curves = uniformize(target_uni[:matched_point_idx][None], self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
            transformed_curve = tiled_curves[0]

            best_matches = get_partial_matches(best_matches, tiled_curves[0], )

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])

            st_id, en_id = get_partial_index(original_match, tiled_curves[0], )

            st_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[st_id].squeeze()
            en_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[en_id].squeeze()

            st_theta[st_theta > en_theta] = st_theta[st_theta > en_theta] - 2 * np.pi

        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            best_matches = uniformize(best_matches, self.optim_timesteps)
            target_uni = uniformize(target_curve_copy[None], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(best_matches, target_uni, )
            tiled_curves = uniformize(target_curve_copy[None], self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
            transformed_curve = tiled_curves[0]

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])

            st_theta = 0.
            en_theta = np.pi * 2

        if visualize:
            ax = draw_mechanism(As[best_idx], x[best_idx], np.where(node_types[best_idx])[0], [0, 1], highlight=tid[0],
                                solve=True, thetas=np.linspace(st_theta, en_theta, self.optim_timesteps))
            # ax.plot(best_matches[0].detach().cpu().numpy()[:,0],best_matches[0].detach().cpu().numpy()[:,1],color="darkorange")
            ax.plot(transformed_curve[:, 0], transformed_curve[:, 1], color="indigo", alpha=0.7, linewidth=2)

            if save_figs is not None:
                fig.savefig(save_figs + '_final_candidate.png')
            else:
                plt.show()

        if verbose:
            print(f'Final Chamfer Distance: {CD[0] * og_scale:.7f}, Ordered Distance: {OD[0] * (og_scale ** 2):.7f}')

        A = As[best_idx]
        x = x[best_idx]
        node_types = node_types[best_idx]

        n_joints = (A.sum(-1) > 0).sum()

        A = A[:n_joints, :][:, :n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]

        transformation = [tr, sc, an]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD * og_scale, OD * (og_scale ** 2), og_scale]

        return [A, x, node_types, start_theta, end_theta, transformation], performance, transformed_curve
        # return As[best_idx].cpu().numpy(), x[best_idx].cpu().numpy(), node_types[best_idx].cpu().numpy(), [tr,sc,an], transformed_curve, best_matches[0].detach().cpu().numpy(), [CD.item()*og_scale,OD.item()*og_scale**2]

    # 曲线预处理和可视化
    def demo_sythesize_step_1(self, target_curve, partial=False):
        torch.cuda.empty_cache()
        start_time = time.time()

        # 预处理曲线，把 target_curve 先变成 (1, N, 2) 的形式
        target_curve = preprocess_curves(target_curve[None], self.curve_size)[0]

        # 获取原始尺度信息
        og_scale = get_scales(target_curve[None])[0]

        size = target_curve.shape[0]
        if partial:
            # fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0]) / 2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = jax.numpy.linalg.norm(start_point - center)
            b = jax.numpy.linalg.norm(end_point - center)
            start_angle = jax.numpy.arctan2(start_point[1] - center[1], start_point[0] - center[0])
            end_angle = jax.numpy.arctan2(end_point[1] - center[1], end_point[0] - center[0])

            angles = jax.numpy.linspace(start_angle, end_angle, self.curve_size)
            ellipse = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)], 1)

            angles = jax.numpy.linspace(start_angle + 2 * np.pi, end_angle, self.curve_size)
            ellipse_2 = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)],
                                        1)

            # ellipse 1 length
            l_1 = jax.numpy.linalg.norm(ellipse - target_curve.mean(0), axis=-1).sum()
            # ellipse 2 length
            l_2 = jax.numpy.linalg.norm(ellipse_2 - target_curve.mean(0), axis=-1).sum()

            if l_1 > l_2:
                target_curve = jax.numpy.concatenate([target_curve, ellipse], 0)
            else:
                target_curve = jax.numpy.concatenate([target_curve, ellipse_2], 0)

        target_curve_copy = preprocess_curves(target_curve[None], self.curve_size)[0]  # target_curve_copy
        target_curve_ = jax.numpy.copy(target_curve)  # target_curve_

        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve[None], n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve[None], self.curve_size)[0]  # target_curve

        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size][None], self.curve_size)[0]
            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), target_curve, )
            transformed_curve = apply_transforms(target_curve[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx + 1][None], self.curve_size)[0]

            target_uni = jax.numpy.copy(target_curve_copy)

            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), target_uni, )
            transformed_curve = apply_transforms(target_uni[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx + 1][None], self.curve_size)[0]
        else:
            target_curve_copy_ = jax.numpy.copy(target_curve_copy)  # target_curve_copy_

        fig1 = plt.figure(figsize=(5, 5))
        if partial:
            plt.plot(target_curve_[:size][:, 0], target_curve_[:size][:, 1], color="indigo")
        else:
            plt.plot(target_curve_copy[:, 0], target_curve_copy[:, 1], color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Original Curve')

        fig2 = plt.figure(figsize=(5, 5))
        plt.plot(target_curve[:, 0], target_curve[:, 1], color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Preprocessed Curve')

        # save all variables which will be used in the next step
        payload = [target_curve_copy,  # 补全了椭圆之后得到的
                   target_curve_copy_,  # 修剪成 partial 部分后的 target_curve_copy（用于后续 uniformize 等操作）
                   target_curve_,  # partial补齐后的曲线，smoothing前
                   target_curve,  # partial补齐smoothing后的曲线
                   og_scale,
                   partial,
                   size]

        return payload, fig1, fig2

    def demo_multi_sythesize_step_1(self, target_curve, partial=False):
        torch.cuda.empty_cache()
        start_time = time.time()

        # 记录原始曲线数量、原始第0条曲线的长度
        size = target_curve[0].shape[0]

        # ---- Step 1: 预处理标准化 ----
        target_curve = preprocess_multi_curves_as_whole(target_curve)  # shape: (n, 200, 2)
        og_scale = get_multi_scales(target_curve.reshape(-1, 200, 2))[0]  # 整体缩放比例

        # 注意这里改成 first_curve
        first_curve = target_curve[0]
        if partial:
            # fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (first_curve[-1] + first_curve[0]) / 2
            start_point = first_curve[-1]
            end_point = first_curve[0]
            a = jax.numpy.linalg.norm(start_point - center)
            b = jax.numpy.linalg.norm(end_point - center)
            start_angle = jax.numpy.arctan2(start_point[1] - center[1], start_point[0] - center[0])
            end_angle = jax.numpy.arctan2(end_point[1] - center[1], end_point[0] - center[0])

            angles = jax.numpy.linspace(start_angle, end_angle, self.curve_size)
            ellipse = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)],
                                      axis=1)

            angles = jax.numpy.linspace(start_angle + 2 * np.pi, end_angle, self.curve_size)
            ellipse_2 = jax.numpy.stack([center[0] + a * jax.numpy.cos(angles), center[1] + b * jax.numpy.sin(angles)],
                                        axis=1)

            # ellipse 1 length
            l_1 = jax.numpy.linalg.norm(ellipse - first_curve.mean(0), axis=-1).sum()
            # ellipse 2 length
            l_2 = jax.numpy.linalg.norm(ellipse_2 - first_curve.mean(0), axis=-1).sum()

            if l_1 > l_2:
                first_curve = jax.numpy.concatenate([first_curve, ellipse], axis=0)
            else:
                first_curve = jax.numpy.concatenate([first_curve, ellipse_2], axis=0)

        # 预处理曲线
        target_curve_copy = preprocess_curves(first_curve[None], self.curve_size)[0]  # target_curve_copy
        target_curve_copy_n = target_curve.at[0].set(target_curve_copy)
        target_curve_ = jax.numpy.copy(first_curve)  # target_curve_
        target_curve_n = target_curve.at[0].set(target_curve_[:size])

        # 平滑或预处理
        if self.smoothing:
            first_curve = smooth_hand_drawn_curves(first_curve[None], n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            first_curve = preprocess_curves(first_curve[None], self.curve_size)[0]  # target_curve

        # partial 模式裁剪部分
        if partial:
            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), first_curve)
            transformed_curve = apply_transforms(first_curve[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            first_curve = preprocess_curves(first_curve[:matched_point_idx + 1][None], self.curve_size)[0]

            target_uni = jax.numpy.copy(target_curve_copy)

            # 同样的裁剪操作，这里可以优化
            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.curve_size), target_uni)
            transformed_curve = apply_transforms(target_uni[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx + 1][None], self.curve_size)[0]
        else:
            target_curve_copy_ = jax.numpy.copy(target_curve_copy)  # target_curve_copy_

        # ---- Step 3: 可视化 ----
        fig1 = plt.figure(figsize=(5, 5))
        for c in target_curve_n:
            plt.plot(c[:, 0], c[:, 1], color='indigo')
        plt.axis('equal')
        plt.axis('off')
        plt.title('Original Curve')

        fig2 = plt.figure(figsize=(5, 5))
        plt.plot(first_curve[:, 0], first_curve[:, 1], color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Preprocessed Curve')

        # ---- Step 4: 封装输出 ----
        payload = [
            target_curve_copy_n,  # 初步处理后的完整曲线（无smoothing/无partial剪裁，用于显示）(3,200,2)
            target_curve_copy_,  # 修剪成 partial 部分后的 target_curve_copy（用于后续 uniformize 等操作）
            target_curve_,  # partial补齐后的曲线，smoothing前
            first_curve,  # partial补齐smoothing后切分的曲线(200,2)
            og_scale,
            partial,
            size
        ]

        return payload, fig1, fig2

    # 嵌入空间检索和候选生成
    def demo_sythesize_step_2(self, payload, max_size=20):
        target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload

        # 构造输入 tensor：把目标曲线放在 batch 第一位，其它位置随机填充 255 条曲线
        input_tensor = target_curve[None]
        batch_padd = preprocess_curves(self.curves[np.random.choice(self.curves.shape[0], 255)])

        input_tensor = torch.tensor(np.concatenate([input_tensor, batch_padd], 0)).float().to(self.device)
        # 在 mixed-precision 下计算 embedding
        with torch.cuda.amp.autocast():
            target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        # target_emb = torch.tensor(target_emb).float().to(self.device)

        # 从预先算好的 embedding 库里筛出长度 <= max_size 的曲线
        ids = np.where(self.sizes <= max_size)[0]
        # 用 jax 实现的余弦相似度搜索，返回按相似度排好序的候选下标 idxs 和相似度 sim
        idxs, sim = cosine_search_jax(target_emb, self.precomputed_emb, ids=ids)
        # idxs = idxs.detach().cpu().numpy()

        # max batch size is 250
        tr, sc, an = [], [], []
        for i in range(int(np.ceil(self.top_n * 5 / 250))):
            tr_, sc_, an_ = find_transforms(self.curves[idxs[i * 250:(i + 1) * 250]], target_curve_copy, )
            tr.append(tr_)
            sc.append(sc_)
            an.append(an_)
        tr = jax.numpy.concatenate(tr, 0)
        sc = jax.numpy.concatenate(sc, 0)
        an = jax.numpy.concatenate(an, 0)
        tiled_curves = target_curve_copy[None].repeat(self.top_n * 5, 0)
        tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
        CD = batch_ordered_distance(tiled_curves / sc[:, None, None],
                                    self.curves[idxs[:self.top_n * 5]] / sc[:, None, None])

        # get best matches index
        tid = jax.numpy.argsort(CD)[:self.top_n]

        grid_size = int(np.ceil(self.top_n ** 0.5))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < self.top_n:
                    axs[i, j].plot(self.curves[idxs[tid][i * grid_size + j]][:, 0],
                                   self.curves[idxs[tid][i * grid_size + j]][:, 1], color='indigo')
                    axs[i, j].plot(tiled_curves[tid][i * grid_size + j][:, 0],
                                   tiled_curves[tid][i * grid_size + j][:, 1], color="darkorange", alpha=0.7)
                axs[i, j].axis('off')
                axs[i, j].axis('equal')

        payload = [idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial,
                   size]

        return payload, fig

    def demo_multi_sythesize_step_2(self, payload, max_size=20):
        target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload

        # 构造输入 tensor：把目标曲线放在 batch 第一位，其它位置随机填充 255 条曲线
        input_tensor = target_curve[None]
        batch_padd = preprocess_curves(self.curves[np.random.choice(self.curves.shape[0], 255)])

        input_tensor = torch.tensor(np.concatenate([input_tensor, batch_padd], 0)).float().to(self.device)
        # 在 mixed-precision 下计算 embedding
        with torch.cuda.amp.autocast():
            target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        # target_emb = torch.tensor(target_emb).float().to(self.device)

        # 从预先算好的 embedding 库里筛出长度 <= max_size 的曲线
        ids = np.where(self.sizes <= max_size)[0]
        # 用 jax 实现的余弦相似度搜索，返回按相似度排好序的候选下标 idxs 和相似度 sim
        idxs, sim = cosine_search_jax(target_emb, self.precomputed_emb, ids=ids)
        # idxs = idxs.detach().cpu().numpy()

        # max batch size is 250
        tr, sc, an = [], [], []
        for i in range(int(np.ceil(self.top_n * 5 / 250))):
            tr_, sc_, an_ = find_transforms(self.curves[idxs[i * 250:(i + 1) * 250]], target_curve_copy[0], )
            tr.append(tr_)
            sc.append(sc_)
            an.append(an_)
        tr = jax.numpy.concatenate(tr, 0)
        sc = jax.numpy.concatenate(sc, 0)
        an = jax.numpy.concatenate(an, 0)
        tiled_curves = target_curve_copy[0:1].repeat(self.top_n * 5, 0)
        tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
        CD = batch_ordered_distance(tiled_curves / sc[:, None, None],
                                    self.curves[idxs[:self.top_n * 5]] / sc[:, None, None])

        # get best matches index
        tid = jax.numpy.argsort(CD)[:self.top_n]

        grid_size = int(np.ceil(self.top_n ** 0.5))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < self.top_n:
                    axs[i, j].plot(self.curves[idxs[tid][i * grid_size + j]][:, 0],
                                   self.curves[idxs[tid][i * grid_size + j]][:, 1], color='indigo')
                    axs[i, j].plot(tiled_curves[tid][i * grid_size + j][:, 0],
                                   tiled_curves[tid][i * grid_size + j][:, 1], color="darkorange", alpha=0.7)
                axs[i, j].axis('off')
                axs[i, j].axis('equal')

        payload = [idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial,
                   size]

        return payload, fig

    # 优化求解和结果可视化
    def demo_sythesize_step_3(self, payload, progress=None):
        idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload

        # 取出初始设计参数
        As = self.As[idxs[tid]]                         # 机构的连接矩阵（节点连接关系）
        x0s = self.x0s[idxs[tid]]                       # 初始节点位置，表示每个关节的初始位置
        node_types = self.node_types[idxs[tid]]         # 表示每个节点是否为固定节点

        # 创建优化目标函数
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        prog = None

        # 使用Batch_BFGS算法对初始参数x0s进行优化
        x, f = Batch_BFGS(x0s, obj, max_iter=self.init_optim_iters, line_search_max_iter=self.BFGS_lineserach_max_iter,
                          tau=self.BFGS_line_search_mult,
                          progress=lambda x: demo_progress_updater(x, progress, desc='Stage 1: '), threshhold=0.001)

        # top n level 2
        top_n_2 = f.argsort()[:self.top_n_level2]
        As = As[top_n_2]
        x0s = x[top_n_2]
        node_types = node_types[top_n_2]

        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)
        prog2 = None

        for i in range(self.n_repos):
            x, f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter // (self.n_repos + 1),
                              line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                              threshhold=0.04, progress=lambda x: demo_progress_updater(
                    [x[0] / (self.n_repos + 1) + i / (self.n_repos + 1), x[1]], progress, desc='Stage 2: '))
            x0s = x

        x, f = Batch_BFGS(x0s, obj,
                          max_iter=self.BFGS_max_iter - self.n_repos * self.BFGS_max_iter // (self.n_repos + 1),
                          line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                          threshhold=0.04, progress=lambda x: demo_progress_updater(
                [x[0] / (self.n_repos + 1) + self.n_repos / (self.n_repos + 1), x[1]], progress, desc='Stage 2: '))

        best_idx = f.argmin()

        end_time = time.time()

        if partial:
            target_uni = uniformize(target_curve_copy[None], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.optim_timesteps), target_uni, )
            transformed_curve = apply_transforms(target_uni[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))

            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            original_match = jax.numpy.copy(best_matches)
            best_matches = uniformize(best_matches, self.optim_timesteps)

            tr, sc, an = find_transforms(best_matches, target_uni, )
            tiled_curves = uniformize(target_uni[:matched_point_idx][None], self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
            transformed_curve = tiled_curves[0]

            best_matches = get_partial_matches(best_matches, tiled_curves[0], )

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])

            st_id, en_id = get_partial_index(original_match, tiled_curves[0], )

            st_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[st_id].squeeze()
            en_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[en_id].squeeze()

            st_theta[st_theta > en_theta] = st_theta[st_theta > en_theta] - 2 * np.pi

        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            best_matches = uniformize(best_matches, self.optim_timesteps)
            target_uni = uniformize(target_curve_copy[None], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(best_matches, target_uni, )
            tiled_curves = uniformize(target_curve_copy[None], self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves, tr, sc, -an)
            transformed_curve = tiled_curves[0]

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves / sc[:, None, None])

            st_theta = 0.
            en_theta = np.pi * 2

        # 提取最优机构（best_idx）相关信息，去掉孤立点
        n_joints = (As[best_idx].sum(-1) > 0).sum()
        # 画出机构运动轨迹
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax = draw_mechanism(As[best_idx][:n_joints, :][:, :n_joints], x[best_idx][0:n_joints],
                            np.where(node_types[best_idx][0:n_joints])[0], [0, 1], highlight=tid[0].item(), solve=True,
                            thetas=np.linspace(st_theta, en_theta, self.optim_timesteps), ax=ax)
        ax.plot(transformed_curve[:, 0], transformed_curve[:, 1], color="indigo", alpha=0.7, linewidth=2)

        # 再清洗一遍最优机构的数据
        A = As[best_idx]
        x = x[best_idx]
        node_types = node_types[best_idx]

        n_joints = (A.sum(-1) > 0).sum()

        A = A[:n_joints, :][:, :n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]

        transformation = [tr, sc, an]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD.item() * og_scale, OD.item() * (og_scale ** 2), og_scale]
        torch.cuda.empty_cache()
        return fig, [[A, x, node_types, start_theta, end_theta, transformation], performance,
                     transformed_curve], gr.update(value={"Progress": 1.0})

    def demo_multi_sythesize_step_3(self, payload, progress=None):
        idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload

        # 取出初始设计参数
        As = self.As[idxs[tid]]                         # 机构的连接矩阵（节点连接关系）
        x0s = self.x0s[idxs[tid]]                       # 初始节点位置，表示每个关节的初始位置
        node_types = self.node_types[idxs[tid]]         # 表示每个节点是否为固定节点

        # 创建优化目标函数
        obj = make_multi_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        prog = None

        # 使用Batch_BFGS算法对初始参数x0s进行优化
        x, f = Batch_BFGS(x0s, obj, max_iter=self.init_optim_iters, line_search_max_iter=self.BFGS_lineserach_max_iter,
                          tau=self.BFGS_line_search_mult,
                          progress=lambda x: demo_progress_updater(x, progress, desc='Stage 1: '), threshhold=0.001)

        # top n level 2
        top_n_2 = f.argsort()[:self.top_n_level2]
        As = As[top_n_2]
        x0s = x[top_n_2]
        node_types = node_types[top_n_2]

        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_multi_batch_optim_obj(target_curve_copy, As, x0s, node_types, timesteps=self.optim_timesteps,
                                   OD_weight=self.OD_weight, CD_weight=self.CD_weight)
        prog2 = None

        for i in range(self.n_repos):
            x, f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter // (self.n_repos + 1),
                              line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                              threshhold=0.04, progress=lambda x: demo_progress_updater(
                    [x[0] / (self.n_repos + 1) + i / (self.n_repos + 1), x[1]], progress, desc='Stage 2: '))
            x0s = x

        x, f = Batch_BFGS(x0s, obj,
                          max_iter=self.BFGS_max_iter - self.n_repos * self.BFGS_max_iter // (self.n_repos + 1),
                          line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult,
                          threshhold=0.04, progress=lambda x: demo_progress_updater(
                [x[0] / (self.n_repos + 1) + self.n_repos / (self.n_repos + 1), x[1]], progress, desc='Stage 2: '))

        best_idx = f.argmin()

        end_time = time.time()

        if partial:
            target_uni_1 = uniformize(target_curve_copy[0:1], self.optim_timesteps)[0]
            target_uni_2 = uniformize(target_curve_copy[1:2], self.optim_timesteps)[0]
            target_uni_3 = uniformize(target_curve_copy[2:3], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(uniformize(target_curve_[None], self.optim_timesteps), target_uni_1, )
            transformed_curve = apply_transforms(target_uni_1[None], tr, sc, -an)[0]
            end_point = target_curve_[size - 1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve - end_point, axis=-1))

            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            original_match = jax.numpy.copy(best_matches)
            best_matches = uniformize(best_matches, self.optim_timesteps)

            tr, sc, an = find_transforms(best_matches, target_uni_1, )
            tiled_curves_1 = uniformize(target_uni_1[:matched_point_idx][None], self.optim_timesteps)
            tiled_curves_1 = apply_transforms(tiled_curves_1, tr, sc, -an)
            transformed_curve_1 = tiled_curves_1[0]

            tiled_curves_2 = uniformize(target_uni_2[None], self.optim_timesteps)
            tiled_curves_2 = apply_transforms(tiled_curves_2, tr, sc, -an)
            transformed_curve_2 = tiled_curves_2[0]

            tiled_curves_3 = uniformize(target_uni_3[None], self.optim_timesteps)
            tiled_curves_3 = apply_transforms(tiled_curves_3, tr, sc, -an)
            transformed_curve_3 = tiled_curves_3[0]

            best_matches = get_partial_matches(best_matches, tiled_curves_1[0], )

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves_1 / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves_1 / sc[:, None, None])

            st_id, en_id = get_partial_index(original_match, tiled_curves_1[0], )

            st_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[st_id].squeeze()
            en_theta = np.linspace(0, 2 * np.pi, self.optim_timesteps)[en_id].squeeze()

            st_theta[st_theta > en_theta] = st_theta[st_theta > en_theta] - 2 * np.pi

        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx + 1], x[best_idx:best_idx + 1],
                                             node_types[best_idx:best_idx + 1],
                                             jax.numpy.linspace(0, jax.numpy.pi * 2, self.optim_timesteps))
            tid = (As[best_idx:best_idx + 1].sum(-1) > 0).sum(-1) - 1
            best_matches = sol[np.arange(sol.shape[0]), tid]
            best_matches = uniformize(best_matches, self.optim_timesteps)
            target_uni_1 = uniformize(target_curve_copy[0:1], self.optim_timesteps)[0]

            tr, sc, an = find_transforms(best_matches, target_uni_1, )
            tiled_curves_1 = uniformize(target_curve_copy[0:1], self.optim_timesteps)
            tiled_curves_1 = apply_transforms(tiled_curves_1, tr, sc, -an)
            transformed_curve_1 = tiled_curves_1[0]

            tiled_curves_2 = uniformize(target_curve_copy[1:2], self.optim_timesteps)
            tiled_curves_2 = apply_transforms(tiled_curves_2, tr, sc, -an)
            transformed_curve_2 = tiled_curves_2[0]

            tiled_curves_3 = uniformize(target_curve_copy[2:3], self.optim_timesteps)
            tiled_curves_3 = apply_transforms(tiled_curves_3, tr, sc, -an)
            transformed_curve_3 = tiled_curves_3[0]

            CD = batch_chamfer_distance(best_matches / sc[:, None, None], tiled_curves_1 / sc[:, None, None])
            OD = ordered_objective_batch(best_matches / sc[:, None, None], tiled_curves_1 / sc[:, None, None])

            st_theta = 0.
            en_theta = np.pi * 2

        n_joints = (As[best_idx].sum(-1) > 0).sum()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax = draw_mechanism(As[best_idx][:n_joints, :][:, :n_joints], x[best_idx][0:n_joints],
                            np.where(node_types[best_idx][0:n_joints])[0], [0, 1], highlight=tid[0].item(), solve=True,
                            thetas=np.linspace(st_theta, en_theta, self.optim_timesteps), ax=ax)
        ax.plot(transformed_curve_1[:, 0], transformed_curve_1[:, 1], color="indigo", alpha=0.7, linewidth=2)
        ax.plot(transformed_curve_2[:, 0], transformed_curve_2[:, 1], color="indigo", alpha=0.7, linewidth=2)
        ax.plot(transformed_curve_3[:, 0], transformed_curve_3[:, 1], color="indigo", alpha=0.7, linewidth=2)

        A = As[best_idx]
        x = x[best_idx]
        node_types = node_types[best_idx]

        n_joints = (A.sum(-1) > 0).sum()

        A = A[:n_joints, :][:, :n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]

        transformation = [tr, sc, an]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD.item() * og_scale, OD.item() * (og_scale ** 2), og_scale]
        torch.cuda.empty_cache()
        return fig, [[A, x, node_types, start_theta, end_theta, transformation], performance,
                     transformed_curve_1], gr.update(value={"Progress": 1.0})


# 用于处理部分曲线匹配，找到曲线的起始和结束点的索引，并生成匹配的曲线部分。
def get_partial_matches(curves, target_curve):
    objective_fn = batch_ordered_distance
    start_point = target_curve[0]
    end_point = target_curve[-1]

    start_match_idx = np.linalg.norm(start_point - curves, axis=-1).argmin(-1)
    end_match_idx = np.linalg.norm(end_point - curves, axis=-1).argmin(-1)

    test_target = jax.numpy.concatenate([target_curve[None], target_curve[None]], 0)

    curves_out = []

    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i] + 1][None], target_curve.shape[0])[0]
            partial_2 = uniformize(
                jax.numpy.concatenate([curves[i][end_match_idx[i]:], curves[i][:start_match_idx[i] + 1]], 0)[None],
                target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            idx = f.argmin()
            curves_out.append(uniformize(partials[idx].squeeze()[None], target_curve.shape[0])[0])
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i] + 1][None], target_curve.shape[0])[0]
            partial_2 = uniformize(
                jax.numpy.concatenate([curves[i][start_match_idx[i]:], curves[i][:end_match_idx[i] + 1]], 0)[None],
                target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            idx = f.argmin()
            curves_out.append(uniformize(partials[idx].squeeze()[None], target_curve.shape[0])[0])

    return np.array(curves_out)


def get_partial_index(curves, target_curve):
    objective_fn = batch_ordered_distance
    start_point = target_curve[0]
    end_point = target_curve[-1]

    start_match_idx = np.linalg.norm(start_point - curves, axis=-1).argmin(-1)
    end_match_idx = np.linalg.norm(end_point - curves, axis=-1).argmin(-1)

    actual_start = np.copy(start_match_idx)
    actual_end = np.copy(end_match_idx)

    test_target = jax.numpy.concatenate([target_curve[None], target_curve[None]], 0)

    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i] + 1][None], target_curve.shape[0])[0]
            partial_2 = uniformize(
                jax.numpy.concatenate([curves[i][end_match_idx[i]:], curves[i][:start_match_idx[i] + 1]], 0)[None],
                target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            idx = f.argmin()
            if idx == 1:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i] + 1][None], target_curve.shape[0])[0]
            partial_2 = uniformize(
                jax.numpy.concatenate([curves[i][start_match_idx[i]:], curves[i][:end_match_idx[i] + 1]], 0)[None],
                target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            idx = f.argmin()

            if idx == 0:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]

    return actual_start, actual_end


# 处理曲线的部分匹配问题
def get_partial_matches_oto(curves, target_curves):
    objective_fn = batch_ordered_distance
    for i in range(curves.shape[0]):
        start_point = target_curves[i][0]
        end_point = target_curves[i][-1]

        start_match_idx = np.linalg.norm(start_point - curves[i:i + 1], axis=-1).argmin(-1).squeeze()
        end_match_idx = np.linalg.norm(end_point - curves[i:i + 1], axis=-1).argmin(-1).squeeze()

        test_target = jax.numpy.concatenate([target_curves[i][None], target_curves[i][None]], 0)

        if start_match_idx < end_match_idx:
            partial_1 = uniformize(curves[i][start_match_idx:end_match_idx + 1][None], curves.shape[1])[0]
            partial_2 = \
            uniformize(jax.numpy.concatenate([curves[i][end_match_idx:], curves[i][:start_match_idx + 1]], 0)[None],
                       curves.shape[1])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][start_match_idx:end_match_idx + 1][None], curves.shape[1])[0]
            else:
                curves[i] = \
                    uniformize(
                        jax.numpy.concatenate([curves[i][end_match_idx:], curves[i][:start_match_idx + 1]], 0)[None],
                        curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze()[None],curves.shape[1])[0]
        else:
            partial_1 = uniformize(curves[i][end_match_idx:start_match_idx + 1][None], curves.shape[1])[0]
            partial_2 = \
                uniformize(jax.numpy.concatenate([curves[i][start_match_idx:], curves[i][:end_match_idx + 1]], 0)[None],
                           curves.shape[1])[0]
            partials = jax.numpy.concatenate([partial_1[None], partial_2[None]], 0)

            f = objective_fn(partials, test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][end_match_idx:start_match_idx + 1][None], curves.shape[1])[0]
            else:
                curves[i] = \
                    uniformize(
                        jax.numpy.concatenate([curves[i][start_match_idx:], curves[i][:end_match_idx + 1]], 0)[None],
                        curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze()[None],curves.shape[1])[0]

    return curves
