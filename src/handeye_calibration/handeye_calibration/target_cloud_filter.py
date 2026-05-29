"""Filter target-neighborhood points out of a point cloud for MoveIt OctoMap."""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener


def transform_point(point, transform):
    q = transform.transform.rotation
    t = transform.transform.translation
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return rotation @ np.asarray(point, dtype=np.float64) + translation


class TargetCloudFilter(Node):
    """Remove points near the selected target before MoveIt builds OctoMap."""

    def __init__(self):
        super().__init__('target_cloud_filter')
        self.declare_parameter('input_cloud_topic',
                               '/camera/camera/depth/color/points')
        self.declare_parameter('output_cloud_topic',
                               '/camera/camera/depth/color/points_filtered')
        self.declare_parameter('target_topic',
                               '/pixel_to_robot/target_point')
        self.declare_parameter('target_radius', 0.08)
        self.declare_parameter('target_timeout_sec', 10.0)

        self.target_radius = float(self.get_parameter('target_radius').value)
        self.target_timeout_sec = float(
            self.get_parameter('target_timeout_sec').value)
        self._target = None
        self._target_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('output_cloud_topic').value,
            10)
        self.create_subscription(
            PointCloud2,
            self.get_parameter('input_cloud_topic').value,
            self._cloud_cb,
            10)
        self.create_subscription(
            PointStamped,
            self.get_parameter('target_topic').value,
            self._target_cb,
            10)

    def _target_cb(self, msg):
        self._target = msg
        self._target_time = self.get_clock().now()
        self.get_logger().info(
            'Filtering target neighborhood: frame={} point=[{:.3f}, {:.3f}, {:.3f}] '
            'radius={:.3f}m'.format(
                msg.header.frame_id, msg.point.x, msg.point.y, msg.point.z,
                self.target_radius))

    def _active_target(self):
        if self._target is None or self._target_time is None:
            return None
        age = (self.get_clock().now() - self._target_time).nanoseconds * 1e-9
        if age > self.target_timeout_sec:
            return None
        return self._target

    def _cloud_cb(self, msg):
        target = self._active_target()
        if target is None:
            self.cloud_pub.publish(msg)
            return

        target_xyz = np.array(
            [target.point.x, target.point.y, target.point.z],
            dtype=np.float64)
        if target.header.frame_id != msg.header.frame_id:
            try:
                transform = self.tf_buffer.lookup_transform(
                    msg.header.frame_id,
                    target.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.2))
            except TransformException as exc:
                self.get_logger().warn(
                    'Cannot transform target {} -> cloud {}: {}'.format(
                        target.header.frame_id, msg.header.frame_id, exc))
                self.cloud_pub.publish(msg)
                return
            target_xyz = transform_point(target_xyz, transform)

        fields = [field.name for field in msg.fields]
        rows = []
        radius_sq = self.target_radius * self.target_radius
        removed = 0
        for point in point_cloud2.read_points(
                msg, field_names=fields, skip_nans=False):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                delta = np.array([x, y, z], dtype=np.float64) - target_xyz
                if float(delta @ delta) <= radius_sq:
                    removed += 1
                    continue
            rows.append(tuple(point))

        filtered = point_cloud2.create_cloud(msg.header, msg.fields, rows)
        filtered.is_dense = msg.is_dense and removed == 0
        self.cloud_pub.publish(filtered)


def main(args=None):
    rclpy.init(args=args)
    node = TargetCloudFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
