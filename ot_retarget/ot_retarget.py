import torch
from torch import cdist
from torch.nn.functional import normalize

from einops import rearrange
from torch_einops_utils import batched_index_select
from torch_einops_utils.shape import assert_shape, size, exists

# rms radius normalization

@assert_shape('... num_pts dim', dim = 3)
def normalize_points(
    pts,                 # (b n 3)
    eps = 1e-6
):
    center = pts.mean(dim = -2, keepdim = True)
    diff = pts - center
    sq_diff = diff.square().sum(dim = -1)
    variance = sq_diff.mean(dim = -1, keepdim = True)
    radius = rearrange(variance.sqrt().clamp(min = eps), '... -> ... 1')

    return diff / radius

# sinkhorn ot (eq. 3), uniform marginals

@assert_shape('... num_rows num_cols')
def sinkhorn(
    cost_mat,            # (b num_rows num_cols)
    temp = 0.1,
    iters = 50,
    eps = 1e-12
):
    dtype = cost_mat.dtype
    num_cols = size(cost_mat, '... num_cols -> num_cols')
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    matrix = (-cost_mat / temp).to(compute_dtype).exp()

    for _ in range(iters):
        matrix = normalize(matrix, p = 1, dim = -1, eps = eps)
        matrix = normalize(matrix, p = 1, dim = -2, eps = eps)

    plan = matrix / num_cols
    return plan.to(dtype)

# proximity triple (eq. 2); channel frame, normals: witness -> probe

@assert_shape('... num_probes dim', dim = 3)
def proximity_triple(
    probes,              # (b n 3)
    dist_fn,
    closest_fn,
    eps = 1e-12
):
    witness = closest_fn(probes)
    dist = dist_fn(probes)
    diff = probes - witness
    normals = normalize(diff, dim = -1, eps = eps)

    return dist, witness, normals

# interaction residuals (eq. 5, 6); channel frame, demo_normals: witness -> probe
# pass geodesic_fn for on-manifold geodesic dist, else euclidean

@assert_shape({
    'witness': '... num_probes dim',
    'normals': '... num_probes dim',
    'demo_witness': '... num_probes dim',
    'demo_normals': '... num_probes dim',
    'dist': '... num_probes',
    'demo_dist': '... num_probes',
    'probes': '... num_probes dim',
    'probe_weights': '... num_probes'
}, dim = 3)
def interaction_residuals(
    dist,                # (b n)
    witness,             # (b n 3)
    normals,             # (b n 3)
    demo_dist,           # (b n)
    demo_witness,        # (b n 3)
    demo_normals,        # (b n 3)
    probes = None,       # (b n 3)
    probe_weights = None, # (b n)
    geodesic_fn = None,
    kernel_width = 0.13,
    dist_weight = 256.,
    witness_weight = 625.,
    normal_weight = 2500.,
    beta_max = 5.,
    eps = 1e-12
):
    clamped_demo_dist = demo_dist.relu()
    kernel_denom = max(kernel_width, eps)

    demo_kernel = (1. - clamped_demo_dist / kernel_denom).relu().square()

    active_dist = torch.minimum(clamped_demo_dist, dist)
    dist_kernel = (1. - active_dist / kernel_denom).relu().square().clamp(max = beta_max)

    dist_res = dist - clamped_demo_dist

    if exists(geodesic_fn):
        witness_res = geodesic_fn(witness, demo_witness).square()
    else:
        witness_diff = witness - demo_witness
        witness_res = witness_diff.square().sum(dim = -1)

    if not exists(probes):
        scaled_dist = rearrange(dist.abs(), '... -> ... 1')
        probes = witness + scaled_dist * normals

    rel_vec = probes - demo_witness
    demo_sign = torch.where(demo_dist >= 0., 1., -1.)
    normal_proj = (demo_normals * rel_vec).sum(dim = -1) * demo_sign
    normal_res = (-normal_proj).relu().square()

    dist_term = dist_weight * dist_kernel.square() * dist_res.square()
    witness_term = witness_weight * demo_kernel.square() * witness_res
    normal_term = normal_weight * demo_kernel.square() * normal_res

    cost = dist_term + witness_term + normal_term

    if exists(probe_weights):
        cost = probe_weights * cost

    return cost

# surface correspondence (eq. 4), precomputed at rest pose

@assert_shape({
    'human_pts': '... num_human dim',
    'robot_pts': '... num_robot dim',
    'plan': '... num_human num_robot'
}, dim = 3)
def surface_correspondence(
    human_pts,           # (b num_human 3)
    robot_pts,           # (b num_robot 3)
    plan = None,         # (b num_human num_robot)
    temp = 0.1,
    iters = 50,
    eps = 1e-12,
    return_plan = False
):
    if not exists(plan):
        normed_human_pts = normalize_points(human_pts, eps = eps)
        normed_robot_pts = normalize_points(robot_pts, eps = eps)

        cost_mat = cdist(normed_human_pts, normed_robot_pts).square()
        plan = sinkhorn(cost_mat, temp = temp, iters = iters, eps = eps)

    # src(p) = argmax_i P*[i, p] (eq. 4)
    source_indices = plan.argmax(dim = -2)

    if return_plan:
        return source_indices, plan

    return source_indices

# transfer across object geometries (eq. 3, 4); nearest demo point -> row argmax

@assert_shape({
    'demo_witness': '... num_witness dim',
    'demo_pts': '... num_demo dim',
    'sub_pts': '... num_sub dim',
    'sub_normals': '... num_sub dim',
    'plan': '... num_demo num_sub'
}, dim = 3)
def transfer_object_targets(
    demo_witness,        # (b num_witness 3)
    demo_pts,            # (b num_demo 3)
    sub_pts,             # (b num_sub 3)
    sub_normals = None,  # (b num_sub 3)
    plan = None,         # (b num_demo num_sub)
    temp = 0.1,
    iters = 50,
    eps = 1e-12,
    return_plan = False
):
    if not exists(plan):
        normed_demo_pts = normalize_points(demo_pts, eps = eps)
        normed_sub_pts = normalize_points(sub_pts, eps = eps)

        cost_mat = cdist(normed_demo_pts, normed_sub_pts).square()
        plan = sinkhorn(cost_mat, temp = temp, iters = iters, eps = eps)

    closest_dist = cdist(demo_witness, demo_pts)
    closest_indices = closest_dist.argmin(dim = -1)
    matched_plan = batched_index_select(plan, closest_indices, dim = -2)
    target_indices = matched_plan.argmax(dim = -1)

    targets = batched_index_select(sub_pts, target_indices, dim = -2)

    if exists(sub_normals):
        target_normals = batched_index_select(sub_normals, target_indices, dim = -2)

        if return_plan:
            return targets, target_normals, plan

        return targets, target_normals

    if return_plan:
        return targets, plan

    return targets

# joint retargeting (eq. 1, 8a); pass diff_fn for SE(3)-aware ⊖

def joint_retarget(
    robot_config,        # (b num_joints)
    object_poses,        # (b num_objects 7)
    style_cost_fn,
    interaction_cost_fn,
    prev_robot_config = None, # (b num_joints)
    smooth_weight = 1.,
    diff_fn = None,
    constraints = None,
    iters = 6,
    solver_fn = None
):
    def objective(config = None, poses = None):
        curr_config = config if exists(config) else robot_config
        curr_poses = poses if exists(poses) else object_poses
        cost = style_cost_fn(curr_config, curr_poses) + interaction_cost_fn(curr_config, curr_poses)

        if exists(prev_robot_config):
            diff = diff_fn(curr_config, prev_robot_config) if exists(diff_fn) else (curr_config - prev_robot_config)
            smooth_cost = diff.square().sum()
            cost = cost + smooth_weight * smooth_cost

        return cost

    if not exists(solver_fn):
        return objective()

    return solver_fn(
        objective,
        robot_config,
        object_poses,
        constraints,
        iters = iters
    )
