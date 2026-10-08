import pytest
import torch
from torch.nn.functional import normalize

from ot_retarget import (
    proximity_triple,
    interaction_residuals,
    transfer_object_targets,
    joint_retarget
)

@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('use_geodesic_fn', (False, True))
def test_e2e_joint_retarget(batch_size, use_geodesic_fn):
    torch.manual_seed(0)
    demo_dir = normalize(torch.randn(batch_size, 4, 3), dim = -1)
    sub_pts = normalize(torch.randn(batch_size, 32, 3), dim = -1)
    demo_dist = torch.full((batch_size, 4), 0.05)

    # eq. 3, 4
    target_witness, target_normals = transfer_object_targets(demo_dir, demo_dir, sub_pts, sub_normals = sub_pts)

    # eq. 2, 5, 6
    def interaction_cost_fn(q, poses):
        pts = q[:, None] + demo_dir - poses[:, None]
        dist, witness, normals = proximity_triple(
            pts,
            lambda x: x.norm(dim = -1) - 1.,
            lambda x: normalize(x, dim = -1)
        )
        geodesic_fn = (lambda a, b: (a - b).norm(dim = -1) * 1.5) if use_geodesic_fn else None
        return interaction_residuals(dist, witness, normals, demo_dist, target_witness, target_normals, geodesic_fn = geodesic_fn).sum()

    style_cost_fn = lambda q, poses: (q ** 2).sum() + (poses ** 2).sum()
    robot_config = torch.full((batch_size, 3), 0.5, requires_grad = True)
    object_poses = torch.zeros(batch_size, 3, requires_grad = True)
    initial = joint_retarget(robot_config, object_poses, style_cost_fn, interaction_cost_fn)

    # eq. 1, 8a
    def solver(objective, config, poses, constraints, iters):
        optimizer = torch.optim.Adam([config, poses], lr = 0.05)
        for _ in range(iters):
            optimizer.zero_grad()
            objective().backward()
            optimizer.step()
        return objective()

    final = joint_retarget(robot_config, object_poses, style_cost_fn, interaction_cost_fn, solver_fn = solver, iters = 100)
    assert final < initial
