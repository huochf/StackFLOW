import torch


def perspective_projection(points, trans=None, rotmat=None, focal_length=None, optical_center=None):
    # points: [b, n, 3], trans: [b, 3], rotmat: [b, 3, 3], focal_length: [b, 2], optical_center: [b, 2]
    if rotmat is not None:
        points = torch.matmul(points, rotmat.permute(0, 2, 1))
    if trans is not None:
        points = points + trans[:, None]

    if focal_length is not None:
        u = points[:, :, 0] / points[:, :, 2] * focal_length[:, 0:1]
        v = points[:, :, 1] / points[:, :, 2] * focal_length[:, 1:]
        points_2d = torch.stack([u, v], dim=2)
    else:
        u = points[:, :, 0] / points[:, :, 2]
        v = points[:, :, 1] / points[:, :, 2]
        points_2d = torch.stack([u, v], dim=2)

    if optical_center is not None:
        points_2d = points_2d + optical_center[:, None]

    return points_2d