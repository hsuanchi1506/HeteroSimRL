"""Quaternion helpers used by the official Factory code, (w, x, y, z) order as in Isaac Sim."""
import math as _m

import torch


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_apply(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def tf_inverse(q, t):
    qi = quat_conjugate(q)
    return qi, -quat_apply(qi, t)


def quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                        cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp], -1)


def get_euler_xyz(q):
    """Roll, pitch, yaw wrapped to [0, 2 pi), as Isaac Sim returns them."""
    qw, qx, qy, qz = q.unbind(-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), qw * qw - qx * qx - qy * qy + qz * qz)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1, _m.pi / 2.0 * torch.sign(sinp), torch.asin(sinp.clamp(-1, 1)))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)
    return roll % (2 * _m.pi), pitch % (2 * _m.pi), yaw % (2 * _m.pi)


def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = torch.nn.functional.normalize(axis, dim=-1) * theta.sin()
    return torch.cat([theta.cos(), xyz], -1)
