"""
SynPF sensor model.

Ingredients (see Lim et al., "Robustness Evaluation of Localization Techniques
for Autonomous Racing", DATE 2024):
  1. Boxed LiDAR layout      -> boxed_beam_angles()
  2. Beam sensor model (LUT) -> SensorModel._build_table()
  3. Fast expected ranges    -> RangeLibcBackend (GLT/CDDT/...) or the
                                dependency-free SphereTracingBackend fallback

Changes w.r.t. the lab skeleton:
  * removed the stray `raise NotImplementedError` in the table builder
  * table is indexed in METRES (max_range / (table_width-1) per bin), so it no
    longer depends on the map resolution
  * every mixture component is normalised per column before mixing
  * log-space accumulation + "squash" exponent (beams are not independent)
  * arbitrary beam angles (needed for the boxed layout)
  * evaluate() always returns a valid, normalised weight vector
  * map subscription uses a transient-local QoS so a latched map is received
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

try:  # optional: the accelerator used in the paper
    import range_libc
except ImportError:
    range_libc = None


def _param(node, name, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


# --------------------------------------------------------------------------
# 1. Boxed LiDAR layout
# --------------------------------------------------------------------------
def boxed_beam_angles(num_beams, fov, aspect_ratio):
    """
    Choose `num_beams` beam angles (radians, sensor frame, 0 = straight ahead)
    whose intersections with a rectangular corridor are uniformly spaced along
    the corridor boundary.

    The corridor is centred on the sensor, half-width 1, half-length
    `aspect_ratio` (length : width = aspect_ratio). Large aspect ratios put
    many beams at shallow angles, i.e. looking far down the track.
    """
    a, b = float(aspect_ratio), 1.0
    phi = np.linspace(-fov / 2.0, fov / 2.0, 20001)
    c, s = np.cos(phi), np.sin(phi)
    r = np.minimum(a / np.maximum(np.abs(c), 1e-9),
                   b / np.maximum(np.abs(s), 1e-9))        # ray -> box boundary
    px, py = r * c, r * s
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    targets = np.linspace(0.0, arc[-1], num_beams)          # uniform in arc length
    return np.interp(targets, arc, phi)


def uniform_beam_angles(num_beams, fov):
    return np.linspace(-fov / 2.0, fov / 2.0, num_beams)


# --------------------------------------------------------------------------
# 2. Ray-casting backends: ranges(poses (N,3), beam_angles (K,)) -> (N,K) [m]
# --------------------------------------------------------------------------
class SphereTracingBackend:
    """
    Pure numpy ray marching on a Euclidean distance transform. Works with
    arbitrary beam angles and no extra dependencies. Slower than rangelibc's
    lookup table; use it as a fallback / for testing.
    """

    def __init__(self, occupied, resolution, origin, max_range,
                 lidar_offset=0.0, max_iter=80, hit_eps_px=1.0):
        self.edt = distance_transform_edt(~occupied).astype(np.float32)  # px
        self.h, self.w = occupied.shape
        self.res = resolution
        self.origin = origin
        self.max_px = max_range / resolution
        self.lidar_offset = lidar_offset
        self.max_iter = max_iter
        self.hit_eps = hit_eps_px

    def ranges(self, poses, beam_angles):
        N, K = len(poses), len(beam_angles)
        th = poses[:, 2]
        x = poses[:, 0] + self.lidar_offset * np.cos(th)
        y = poses[:, 1] + self.lidar_offset * np.sin(th)

        ox, oy, yaw = self.origin                      # world -> grid frame
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = x - ox, y - oy
        u0 = np.repeat((c * dx + s * dy) / self.res, K)
        v0 = np.repeat((-s * dx + c * dy) / self.res, K)
        ga = th[:, None] + beam_angles[None, :] - yaw
        du, dv = np.cos(ga).ravel(), np.sin(ga).ravel()

        t = np.zeros(N * K)
        active = np.arange(N * K)
        for _ in range(self.max_iter):
            if active.size == 0:
                break
            uu = u0[active] + t[active] * du[active]
            vv = v0[active] + t[active] * dv[active]
            inside = (uu >= 0) & (uu < self.w - 1) & (vv >= 0) & (vv < self.h - 1)
            d = np.zeros(active.size)                  # outside map == obstacle
            ii = np.nonzero(inside)[0]
            d[ii] = self.edt[np.rint(vv[ii]).astype(np.intp),
                             np.rint(uu[ii]).astype(np.intp)]
            t[active] += d
            active = active[(d > self.hit_eps) & (t[active] < self.max_px)]
        return (np.minimum(t, self.max_px) * self.res).reshape(N, K)


class RangeLibcBackend:
    """
    Wrapper around rangelibc (Walsh & Karaman, CDDT) as used in the paper.
    NOTE: written against the MIT-racecar / pmusau17 fork API. Check class and
    method names against the fork you installed.
    method: 'glt' (giant lookup table, fastest on CPU, memory hungry),
            'cddt' / 'pcddt' (low memory), 'rm' (ray marching).
    """

    def __init__(self, map_msg, max_range, method, theta_bins, lidar_offset=0.0):
        omap = range_libc.PyOMap(map_msg)
        max_px = int(max_range / map_msg.info.resolution)
        if method == 'glt':
            self.rm = range_libc.PyGiantLUTCast(omap, max_px, theta_bins)
        elif method in ('cddt', 'pcddt'):
            self.rm = range_libc.PyCDDTCast(omap, max_px, theta_bins)
            if method == 'pcddt':
                self.rm.prune()
        elif method == 'rm':
            self.rm = range_libc.PyRayMarching(omap, max_px)
        else:
            raise ValueError("unknown range_method '%s'" % method)
        self.lidar_offset = lidar_offset

    def ranges(self, poses, beam_angles):
        N, K = len(poses), len(beam_angles)
        th = poses[:, 2]
        q = np.empty((N * K, 3), dtype=np.float32)
        q[:, 0] = np.repeat(poses[:, 0] + self.lidar_offset * np.cos(th), K)
        q[:, 1] = np.repeat(poses[:, 1] + self.lidar_offset * np.sin(th), K)
        q[:, 2] = (th[:, None] + beam_angles[None, :]).ravel()
        out = np.empty(N * K, dtype=np.float32)
        self.rm.calc_range_many(q, out)
        return out.reshape(N, K).astype(np.float64)


# --------------------------------------------------------------------------
# 3. Sensor model
# --------------------------------------------------------------------------
class SensorModel:

    def __init__(self, node):
        self.node = node
        self.map_topic = _param(node, 'map_topic', '/map')
        self.num_beams = int(_param(node, 'num_beams_per_particle', 60))
        self.fov = float(_param(node, 'scan_field_of_view', 4.71))          # rad
        self.theta_bins = int(_param(node, 'scan_theta_discretization', 112))
        self.max_range = float(_param(node, 'max_range', 10.0))             # m
        self.range_method = _param(node, 'range_method', 'sphere')
        self.beam_layout = _param(node, 'beam_layout', 'boxed')
        self.box_aspect = float(_param(node, 'box_aspect_ratio', 4.0))
        self.lidar_offset = float(_param(node, 'lidar_offset_x', 0.0))      # m
        self.occ_thresh = int(_param(node, 'occupied_threshold', 50))
        self.unknown_as_obstacle = bool(_param(node, 'unknown_as_obstacle', True))

        a = np.array([_param(node, 'z_hit', 0.74), _param(node, 'z_short', 0.07),
                      _param(node, 'z_max', 0.07), _param(node, 'z_rand', 0.12)],
                     dtype=np.float64)
        self.alpha = a / a.sum()                      # mixture weights must sum to 1
        self.sigma_hit = float(_param(node, 'sigma_hit', 0.10))             # m
        self.table_width = int(_param(node, 'table_width', 501))
        self.squash_power = float(_param(node, 'squash_power', 1.0 / 2.2))

        if self.beam_layout == 'boxed':
            self.beam_angles = boxed_beam_angles(self.num_beams, self.fov, self.box_aspect)
        else:
            self.beam_angles = uniform_beam_angles(self.num_beams, self.fov)

        self.bin_m = self.max_range / (self.table_width - 1)
        self._build_table()

        self.backend = None
        self.map_set = False
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.map_subscriber = node.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, qos)

    # ---- table ---------------------------------------------------------
    def _build_table(self):
        """table[z_bin, z*_bin] = p(z | z*), each column sums to 1."""
        W = self.table_width
        z = np.arange(W, dtype=np.float64)[:, None]          # measured (rows)
        d = np.arange(W, dtype=np.float64)[None, :]          # expected (cols)

        p_hit = np.exp(-0.5 * ((z - d) / (self.sigma_hit / self.bin_m)) ** 2)
        p_hit /= p_hit.sum(axis=0, keepdims=True)

        d_safe = np.maximum(d, 1.0)                           # unexpected short reads
        p_short = np.where(z <= d, 2.0 / d_safe * (1.0 - z / d_safe), 0.0)
        p_short /= np.maximum(p_short.sum(axis=0, keepdims=True), 1e-12)

        p_max = np.zeros((W, W))
        p_max[W - 1, :] = 1.0
        p_rand = np.full((W, W), 1.0 / W)

        ah, as_, am, ar = self.alpha
        table = ah * p_hit + as_ * p_short + am * p_max + ar * p_rand
        table /= table.sum(axis=0, keepdims=True)
        self.sensor_model_table = table
        self.log_table = np.log(np.maximum(table, 1e-12))

    def _to_bin(self, ranges):
        r = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        return np.clip(np.rint(r / self.bin_m), 0, self.table_width - 1).astype(np.intp)

    # ---- beam selection --------------------------------------------------
    def beam_indices(self, angle_min, angle_increment, num_scan_points):
        """Indices into a full LiDAR scan matching self.beam_angles."""
        idx = np.rint((self.beam_angles - angle_min) / angle_increment).astype(int)
        return np.clip(idx, 0, num_scan_points - 1)

    def downsample(self, ranges, angle_min, angle_increment):
        ranges = np.asarray(ranges, dtype=np.float64)
        return ranges[self.beam_indices(angle_min, angle_increment, ranges.size)]

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, particles, observation, angle_min=None, angle_increment=None):
        """
        particles:   (N,3) [x y theta] of the base frame
        observation: either the K already-selected ranges, or a full scan
                     together with angle_min / angle_increment.
        returns:     (N,) weights, normalised to sum to 1
        """
        N = len(particles)
        if not self.map_set:
            return np.full(N, 1.0 / N)

        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.size != self.num_beams:
            if angle_min is None or angle_increment is None:
                raise ValueError("full scan given without angle_min/angle_increment")
            obs = self.downsample(obs, angle_min, angle_increment)

        expected = self.backend.ranges(np.asarray(particles, dtype=np.float64),
                                       self.beam_angles)             # (N,K) metres
        logp = self.log_table[self._to_bin(obs)[None, :], self._to_bin(expected)]
        logp = self.squash_power * logp.sum(axis=1)
        logp -= logp.max()
        w = np.exp(logp)
        return w / w.sum()

    # ---- map -----------------------------------------------------------------
    def map_callback(self, map_msg):
        info = map_msg.info
        grid = np.asarray(map_msg.data, dtype=np.int16).reshape(info.height, info.width)
        occupied = grid >= self.occ_thresh
        if self.unknown_as_obstacle:
            occupied |= grid < 0

        p, o = info.origin.position, info.origin.orientation
        yaw = R.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]
        origin = (p.x, p.y, yaw)

        backend = None
        if self.range_method != 'sphere':
            if range_libc is None:
                self.node.get_logger().warn("range_libc not installed, using sphere tracing")
            else:
                try:
                    backend = RangeLibcBackend(map_msg, self.max_range, self.range_method,
                                               self.theta_bins, self.lidar_offset)
                except Exception as e:  # API differs between rangelibc forks
                    self.node.get_logger().warn("rangelibc failed (%s), using sphere tracing" % e)
        if backend is None:
            backend = SphereTracingBackend(occupied, info.resolution, origin,
                                           self.max_range, self.lidar_offset)
        self.backend = backend
        self.map_set = True
        self.node.get_logger().info("SynPF map initialised (%s)" % type(backend).__name__)
