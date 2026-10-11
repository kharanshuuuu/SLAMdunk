"""Helpers for the particle-filter node: init, resampling, pose estimate."""

import numpy as np


def init_particles(pose, std_xy, std_theta, n, rng=None):
    """Gaussian cloud around an initial pose guess (x, y, theta)."""
    rng = rng or np.random.default_rng()
    p = np.empty((n, 3))
    p[:, 0] = pose[0] + rng.normal(0, std_xy, n)
    p[:, 1] = pose[1] + rng.normal(0, std_xy, n)
    p[:, 2] = pose[2] + rng.normal(0, std_theta, n)
    return p


def effective_sample_size(w):
    return 1.0 / np.sum(w ** 2)


def low_variance_resample(particles, w, rng=None):
    """Systematic resampling; returns new particles with uniform weights."""
    rng = rng or np.random.default_rng()
    n = len(w)
    pos = rng.uniform(0.0, 1.0 / n) + np.arange(n) / n
    idx = np.searchsorted(np.cumsum(w), pos)
    return particles[np.minimum(idx, n - 1)].copy()


def estimate_pose(particles, w):
    """Weighted mean for x,y; weighted circular mean for theta."""
    x = np.sum(w * particles[:, 0])
    y = np.sum(w * particles[:, 1])
    th = np.arctan2(np.sum(w * np.sin(particles[:, 2])),
                    np.sum(w * np.cos(particles[:, 2])))
    return x, y, th
