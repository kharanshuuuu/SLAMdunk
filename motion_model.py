"""
SynPF motion model: TUM improvement to the differential-drive odometry model.

Odometry is given as [dx, dy, dtheta] in the car frame (as in the lab skeleton)
and converted to rotate-translate-rotate (rot1, trans, rot2). Each particle is
moved with Gaussian-perturbed copies of these three numbers (std devs below).

  TUM:   s_rot1  = a1*|rot1| + a2 / max(trans, lam)      <- shrinks with speed
         s_rot2  = a1*|rot2| + a2 / max(trans, lam)
         s_trans = a3*trans  + a4*(|rot1| + |rot2|)
  diff:  s_rot   = a1*|rot|  + a2d * trans                <- grows with speed

Changes w.r.t. the lab skeleton:
  * all "variances" were really standard deviations; the naming is fixed
  * the angle-normalisation loop only modified copies; now it is vectorised
  * noise acts on rotate-translate-rotate, so lateral spread comes out
    correlated with heading instead of independent per axis
  * no noise is added when the car did not move (prevents standstill diffusion)
  * backwards motion is handled
Default alphas are starting points, not the authors' values. Tune them.
"""

import numpy as np


def _param(node, name, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class MotionModel:

    def __init__(self, node):
        self.model = _param(node, 'motion_model', 'tum')      # 'tum' | 'diff'
        self.a1 = float(_param(node, 'alpha_1', 0.3))
        self.a2 = float(_param(node, 'alpha_2', 0.01))
        self.a3 = float(_param(node, 'alpha_3', 0.5))
        self.a4 = float(_param(node, 'alpha_4', 0.05))
        self.a2_diff = float(_param(node, 'alpha_2_diff', 0.05))
        self.lam = float(_param(node, 'lam_thresh', 0.05))    # metres per update
        self.min_motion = float(_param(node, 'min_motion', 1e-4))
        seed = int(_param(node, 'seed', -1))
        self.rng = np.random.default_rng(None if seed < 0 else seed)

    @staticmethod
    def decompose(odometry):
        """[dx dy dth] (car frame) -> rot1, signed trans, rot2."""
        dx, dy, dth = (float(v) for v in odometry)
        trans = np.hypot(dx, dy)
        if trans < 1e-9:
            return 0.0, 0.0, wrap_angle(dth)
        if dx >= 0.0:
            rot1, sign = np.arctan2(dy, dx), 1.0
        else:                                   # driving backwards
            rot1, sign = np.arctan2(-dy, -dx), -1.0
        return wrap_angle(rot1), sign * trans, wrap_angle(dth - rot1)

    def noise_std(self, rot1, trans, rot2):
        t = abs(trans)
        if self.model == 'tum':
            s_rot = self.a2 / max(t, self.lam)
            s1 = self.a1 * abs(rot1) + s_rot
            s2 = self.a1 * abs(rot2) + s_rot
        else:                                   # classic diff-drive
            s1 = self.a1 * abs(rot1) + self.a2_diff * t
            s2 = self.a1 * abs(rot2) + self.a2_diff * t
        st = self.a3 * t + self.a4 * (abs(rot1) + abs(rot2))
        return s1, st, s2

    def evaluate(self, particles, odometry):
        """
        particles: (N,3) [x y theta]; odometry: [dx dy dtheta] in the car frame.
        returns the updated (N,3) array (modified in place).
        """
        rot1, trans, rot2 = self.decompose(odometry)
        if abs(trans) < self.min_motion and abs(rot2) < self.min_motion:
            return particles

        n = particles.shape[0]
        s1, st, s2 = self.noise_std(rot1, trans, rot2)
        r1 = rot1 + self.rng.normal(0.0, s1, n)
        tr = trans + self.rng.normal(0.0, st, n)
        r2 = rot2 + self.rng.normal(0.0, s2, n)

        th = particles[:, 2]
        particles[:, 0] += tr * np.cos(th + r1)
        particles[:, 1] += tr * np.sin(th + r1)
        particles[:, 2] = wrap_angle(th + r1 + r2)
        return particles
