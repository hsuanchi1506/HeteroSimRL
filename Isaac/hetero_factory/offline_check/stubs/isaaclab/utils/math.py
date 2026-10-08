import torch


def axis_angle_from_quat(quat, eps=1.0e-6):
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_over = torch.where(angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48)
    return quat[..., 1:4] / sin_half_over.unsqueeze(-1)
