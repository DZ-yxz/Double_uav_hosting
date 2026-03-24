#!/usr/bin/env python3
"""
领导-跟随法 (Leader-Follower) 双无人机编队画圆

UAV1 (Leader): 独立沿圆形轨迹飞行
UAV2 (Follower): 实时跟踪 UAV1 的实际位置 + 固定编队偏移量
"""

import rclpy
import math
import numpy as np
import csv
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint,
    VehicleCommand, VehicleLocalPosition, VehicleStatus,
    TimesyncStatus,
)

# === 场景配置 ===
UAV1_HOME_WORLD = np.array([2.5, 0.0, 0.0])
UAV2_HOME_WORLD = np.array([-2.5, 0.0, 0.0])
TAKEOFF_HEIGHT = -5.0

# === 画圆参数（Leader 的圆形轨迹） ===
CIRCLE_RADIUS = 5.0
CIRCLE_CENTER_X = 0.0
CIRCLE_CENTER_Y = 5.0
CIRCLE_SPEED = 0.015

# === 编队参数（Follower 相对于 Leader 的偏移） ===
# 在世界坐标系下，Follower 在 Leader 左侧 5 米
FORMATION_OFFSET = np.array([-5.0, 0.0, 0.0])

# === 跟随控制增益 ===
# Follower 不是瞬间跳到目标位置，而是以一定增益逼近
# 值越大跟随越紧，1.0 = 完全跟踪（无平滑），0.0 = 不动
FOLLOW_GAIN = 0.15

# === 安全参数 ===
Z_MIN = -7.0
Z_MAX = -2.0
MAX_XY_STEP = 0.5

# === 阶段时长 ===
TAKEOFF_STABLE_TICKS = 200
TRANSITION_TICKS = 200
RAMP_TICKS = 200

# === 频率 ===
HEARTBEAT_RATE = 30.0
CONTROL_RATE = 10.0


class LeaderFollowerCircle(Node):
    def __init__(self) -> None:
        super().__init__('leader_follower_circle')

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )

        # ==================== PX4_1 (Leader) ====================
        self.pub_offboard_d1 = self.create_publisher(
            OffboardControlMode,
            '/px4_1/fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj_d1 = self.create_publisher(
            TrajectorySetpoint,
            '/px4_1/fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd_d1 = self.create_publisher(
            VehicleCommand,
            '/px4_1/fmu/in/vehicle_command', qos_pub)
        self.create_subscription(
            VehicleLocalPosition,
            '/px4_1/fmu/out/vehicle_local_position',
            lambda m: self.pos_cb(m, 1), qos_sub)
        self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status_v1',
            lambda m: self.status_cb(m, 1), qos_sub)
        self.create_subscription(
            TimesyncStatus,
            '/px4_1/fmu/out/timesync_status',
            lambda m: self.timesync_cb(m, 1), qos_sub)

        # ==================== PX4_2 (Follower) ====================
        self.pub_offboard_d2 = self.create_publisher(
            OffboardControlMode,
            '/px4_2/fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj_d2 = self.create_publisher(
            TrajectorySetpoint,
            '/px4_2/fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd_d2 = self.create_publisher(
            VehicleCommand,
            '/px4_2/fmu/in/vehicle_command', qos_pub)
        self.create_subscription(
            VehicleLocalPosition,
            '/px4_2/fmu/out/vehicle_local_position',
            lambda m: self.pos_cb(m, 2), qos_sub)
        self.create_subscription(
            VehicleStatus,
            '/px4_2/fmu/out/vehicle_status_v1',
            lambda m: self.status_cb(m, 2), qos_sub)
        self.create_subscription(
            TimesyncStatus,
            '/px4_2/fmu/out/timesync_status',
            lambda m: self.timesync_cb(m, 2), qos_sub)

        # ==================== 状态变量 ====================
        self.pos_d1 = VehicleLocalPosition()
        self.pos_d2 = VehicleLocalPosition()
        self.status_d1 = VehicleStatus()
        self.status_d2 = VehicleStatus()

        # PX4 时间同步
        self.px4_ts_d1 = 0
        self.px4_ts_d2 = 0
        self.ts_received_d1 = False
        self.ts_received_d2 = False

        # Leader 画圆角度
        self.theta = -math.pi / 2
        self.theta_start = self.theta
        self.circle_complete = False

        # Follower 当前指令位置（世界坐标，用于平滑跟随）
        self.follower_cmd_world = None

        # 上一次指令（用于限速）
        self.last_cmd_d1 = None
        self.last_cmd_d2 = None

        # ---------- 日志 ----------
        timestamp = datetime.now().strftime("%H%M%S")
        self.csv_file = f"log_lf_circle_{timestamp}.csv"
        self.csv_buffer = []
        self.csv_flush_interval = 50
        self.csv_header_written = False

        self.tick = 0
        self.takeoff_tick_start = 0
        self.all_systems_go = False

        # ---------- 定时器 ----------
        self.heartbeat_timer = self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_cb)
        self.control_timer = self.create_timer(
            1.0 / CONTROL_RATE, self.control_cb)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  Leader-Follower 编队画圆控制器")
        self.get_logger().info(f"  Leader (UAV1): 圆心=({CIRCLE_CENTER_X}, {CIRCLE_CENTER_Y}), R={CIRCLE_RADIUS}")
        self.get_logger().info(f"  Follower (UAV2): 偏移={FORMATION_OFFSET}")
        self.get_logger().info(f"  跟随增益={FOLLOW_GAIN}")
        self.get_logger().info(f"  日志: {self.csv_file} (仅第一圈)")
        self.get_logger().info("=" * 55)

    # ==================================================================
    # 时间同步
    # ==================================================================
    def timesync_cb(self, msg, id):
        if id == 1:
            self.px4_ts_d1 = msg.timestamp
            if not self.ts_received_d1:
                self.ts_received_d1 = True
                self.get_logger().info("✓ Leader  (PX4_1) timesync 已连接")
        else:
            self.px4_ts_d2 = msg.timestamp
            if not self.ts_received_d2:
                self.ts_received_d2 = True
                self.get_logger().info("✓ Follower(PX4_2) timesync 已连接")

    def get_ts(self, id):
        return self.px4_ts_d1 if id == 1 else self.px4_ts_d2

    # ==================================================================
    # 心跳回调
    # ==================================================================
    def heartbeat_cb(self):
        if not (self.ts_received_d1 and self.ts_received_d2):
            return

        for id, pub in [(1, self.pub_offboard_d1), (2, self.pub_offboard_d2)]:
            msg = OffboardControlMode()
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
            msg.timestamp = self.get_ts(id)
            pub.publish(msg)

        # Failsafe 自动恢复
        if self.all_systems_go:
            for id in [1, 2]:
                status = self.status_d1 if id == 1 else self.status_d2
                armed = (status.arming_state == VehicleStatus.ARMING_STATE_ARMED)
                offboard = (status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
                if armed and not offboard:
                    self.engage_offboard(id)

    # ==================================================================
    # 状态回调
    # ==================================================================
    def pos_cb(self, msg, id):
        if id == 1:
            self.pos_d1 = msg
        else:
            self.pos_d2 = msg

    def status_cb(self, msg, id):
        if id == 1:
            self.status_d1 = msg
        else:
            self.status_d2 = msg

    # ==================================================================
    # 坐标转换
    # ==================================================================
    @staticmethod
    def local_to_world(local_pos, home_world):
        """本地坐标 → 世界坐标"""
        return local_pos + home_world

    @staticmethod
    def world_to_local(world_pos, home_world):
        """世界坐标 → 本地坐标"""
        return world_pos - home_world

    # ==================================================================
    # Leader 轨迹生成（圆形）
    # ==================================================================
    def compute_leader_target_world(self):
        """计算 Leader 在世界坐标系下的目标位置"""
        x = CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta)
        y = CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta)
        z = TAKEOFF_HEIGHT
        return np.array([x, y, z])

    # ==================================================================
    # Follower 跟随逻辑（核心！）
    # ==================================================================
    def compute_follower_target_world(self):
        """
        Follower 的目标 = Leader 的实际位置 + 编队偏移

        关键区别：
        - 虚拟结构法：Follower 目标基于预计算轨迹
        - 领导跟随法：Follower 目标基于 Leader 的【实时真实位置】
        """
        # 1. 获取 Leader 的实际世界坐标
        leader_local = np.array([
            self.pos_d1.x, self.pos_d1.y, self.pos_d1.z
        ])
        leader_world = self.local_to_world(leader_local, UAV1_HOME_WORLD)

        # 2. 期望位置 = Leader 实际位置 + 编队偏移
        desired_world = leader_world + FORMATION_OFFSET

        # 3. 平滑跟随：Follower 不会瞬间跳到目标，
        #    而是按增益逐步逼近（一阶低通滤波）
        if self.follower_cmd_world is None:
            self.follower_cmd_world = desired_world.copy()
        else:
            self.follower_cmd_world += FOLLOW_GAIN * (
                desired_world - self.follower_cmd_world
            )

        return self.follower_cmd_world.copy()

    # ==================================================================
    # 工具函数
    # ==================================================================
    @staticmethod
    def smooth_step(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def clamp_z(z):
        return max(Z_MIN, min(Z_MAX, z))

    def rate_limit(self, cmd_new, cmd_last, max_step):
        if cmd_last is None:
            return cmd_new.copy()
        diff = cmd_new - cmd_last
        dist = np.linalg.norm(diff[:2])
        if dist > max_step:
            scale = max_step / dist
            limited = cmd_last.copy()
            limited[0] += diff[0] * scale
            limited[1] += diff[1] * scale
            limited[2] = cmd_new[2]
            return limited
        return cmd_new.copy()

    # ==================================================================
    # 日志（缓冲写入，仅画圆阶段）
    # ==================================================================
    def buffer_csv_row(self, leader_cmd_w, follower_cmd_w):
        if self.circle_complete:
            return

        # Leader 实际世界坐标
        l_act_w = self.local_to_world(
            np.array([self.pos_d1.x, self.pos_d1.y, self.pos_d1.z]),
            UAV1_HOME_WORLD)
        # Follower 实际世界坐标
        f_act_w = self.local_to_world(
            np.array([self.pos_d2.x, self.pos_d2.y, self.pos_d2.z]),
            UAV2_HOME_WORLD)
        # 编队误差（Follower 实际位置 vs 期望位置）
        formation_err = np.linalg.norm(
            f_act_w[:2] - (l_act_w[:2] + FORMATION_OFFSET[:2]))

        self.csv_buffer.append([
            self.tick, f"{self.theta:.4f}",
            # Leader 指令 & 实际
            f"{leader_cmd_w[0]:.3f}", f"{leader_cmd_w[1]:.3f}",
            f"{leader_cmd_w[2]:.3f}",
            f"{l_act_w[0]:.3f}", f"{l_act_w[1]:.3f}", f"{l_act_w[2]:.3f}",
            # Follower 指令 & 实际
            f"{follower_cmd_w[0]:.3f}", f"{follower_cmd_w[1]:.3f}",
            f"{follower_cmd_w[2]:.3f}",
            f"{f_act_w[0]:.3f}", f"{f_act_w[1]:.3f}", f"{f_act_w[2]:.3f}",
            # 编队误差
            f"{formation_err:.3f}",
        ])
        if len(self.csv_buffer) >= self.csv_flush_interval:
            self.flush_csv()

    def flush_csv(self):
        if not self.csv_buffer:
            return
        mode = 'a' if self.csv_header_written else 'w'
        with open(self.csv_file, mode, newline='') as f:
            writer = csv.writer(f)
            if not self.csv_header_written:
                writer.writerow([
                    "tick", "theta",
                    "leader_x_cmd", "leader_y_cmd", "leader_z_cmd",
                    "leader_x_act", "leader_y_act", "leader_z_act",
                    "follower_x_cmd", "follower_y_cmd", "follower_z_cmd",
                    "follower_x_act", "follower_y_act", "follower_z_act",
                    "formation_error",
                ])
                self.csv_header_written = True
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()

    # ==================================================================
    # 控制主循环
    # ==================================================================
    def control_cb(self):
        if not (self.ts_received_d1 and self.ts_received_d2):
            return

        self.tick += 1

        # 默认指令：悬停在起飞高度
        cmd_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_yaw = 0.0
        phase = "INIT"

        # Leader 和 Follower 指令的世界坐标（用于日志）
        leader_cmd_world = UAV1_HOME_WORLD + cmd_d1
        follower_cmd_world = UAV2_HOME_WORLD + cmd_d2

        is_armed_d1 = (self.status_d1.arming_state ==
                       VehicleStatus.ARMING_STATE_ARMED)
        is_offboard_d1 = (self.status_d1.nav_state ==
                          VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        is_armed_d2 = (self.status_d2.arming_state ==
                       VehicleStatus.ARMING_STATE_ARMED)
        is_offboard_d2 = (self.status_d2.nav_state ==
                          VehicleStatus.NAVIGATION_STATE_OFFBOARD)

        if is_armed_d1 and is_offboard_d1 and is_armed_d2 and is_offboard_d2:
            if not self.all_systems_go:
                self.get_logger().info(">>> 全部就绪，起飞中... <<<")
                self.takeoff_tick_start = self.tick
            self.all_systems_go = True

        elapsed = (self.tick - self.takeoff_tick_start
                   if self.all_systems_go else 0)

        # ==============================================================
        # 阶段 1：起飞稳定
        # ==============================================================
        if not self.all_systems_go or elapsed < TAKEOFF_STABLE_TICKS:
            phase = "TAKEOFF"
            if self.tick % 5 == 0:
                if not is_offboard_d1:
                    self.engage_offboard(1)
                if not is_offboard_d2:
                    self.engage_offboard(2)
                if is_offboard_d1 and not is_armed_d1:
                    self.arm(1)
                if is_offboard_d2 and not is_armed_d2:
                    self.arm(2)

        # ==============================================================
        # 阶段 2：过渡（Leader 和 Follower 各自平滑过渡到起始位置）
        # ==============================================================
        elif elapsed < TAKEOFF_STABLE_TICKS + TRANSITION_TICKS:
            phase = "TRANSITION"
            t = self.smooth_step(
                (elapsed - TAKEOFF_STABLE_TICKS) / float(TRANSITION_TICKS))

            # Leader 过渡到圆的起始点
            hover_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            leader_start_world = self.compute_leader_target_world()
            leader_start_local = self.world_to_local(
                leader_start_world, UAV1_HOME_WORLD)
            cmd_d1 = hover_d1 * (1.0 - t) + leader_start_local * t

            # Follower 过渡到编队起始位置
            hover_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            follower_start_world = leader_start_world + FORMATION_OFFSET
            follower_start_local = self.world_to_local(
                follower_start_world, UAV2_HOME_WORLD)
            cmd_d2 = hover_d2 * (1.0 - t) + follower_start_local * t

            # 初始化 Follower 平滑跟随的起点
            self.follower_cmd_world = (
                (UAV2_HOME_WORLD + hover_d2) * (1.0 - t)
                + follower_start_world * t
            )

            leader_cmd_world = UAV1_HOME_WORLD + cmd_d1
            follower_cmd_world = UAV2_HOME_WORLD + cmd_d2

        # ==============================================================
        # 阶段 3：画圆（Leader 独立画圆，Follower 实时跟随）
        # ==============================================================
        else:
            phase = "CIRCLE"
            circle_ticks = (elapsed - TAKEOFF_STABLE_TICKS
                            - TRANSITION_TICKS)

            # --- Leader：速度斜坡 + 画圆 ---
            if circle_ticks < RAMP_TICKS:
                spd = CIRCLE_SPEED * self.smooth_step(
                    circle_ticks / float(RAMP_TICKS))
            else:
                spd = CIRCLE_SPEED

            self.theta += spd

            # 检测一圈完成
            if ((self.theta - self.theta_start) >= 2.0 * math.pi + 0.5
                    and not self.circle_complete):
                self.circle_complete = True
                self.flush_csv()
                self.get_logger().info(
                    f"=== 一圈完成! 日志: {self.csv_file} ===")

            # Leader 目标（世界坐标 → 本地坐标）
            leader_cmd_world = self.compute_leader_target_world()
            cmd_d1 = self.world_to_local(leader_cmd_world, UAV1_HOME_WORLD)

            # --- Follower：实时跟随 Leader 实际位置 ---
            follower_cmd_world = self.compute_follower_target_world()
            cmd_d2 = self.world_to_local(follower_cmd_world, UAV2_HOME_WORLD)

            # 日志输出
            if self.tick % 50 == 0:
                progress = ((self.theta - self.theta_start)
                            / (2.0 * math.pi) * 100)
                # 计算编队误差
                f_act = self.local_to_world(
                    np.array([self.pos_d2.x, self.pos_d2.y]),
                    UAV2_HOME_WORLD[:2])
                l_act = self.local_to_world(
                    np.array([self.pos_d1.x, self.pos_d1.y]),
                    UAV1_HOME_WORLD[:2])
                err = np.linalg.norm(
                    f_act - (l_act + FORMATION_OFFSET[:2]))
                self.get_logger().info(
                    f"圈:{progress:.1f}% | "
                    f"Z_L:{self.pos_d1.z:.2f} Z_F:{self.pos_d2.z:.2f} | "
                    f"编队误差:{err:.2f}m")

        # === 安全保护 ===
        cmd_d1[2] = self.clamp_z(cmd_d1[2])
        cmd_d2[2] = self.clamp_z(cmd_d2[2])
        cmd_d1 = self.rate_limit(cmd_d1, self.last_cmd_d1, MAX_XY_STEP)
        cmd_d2 = self.rate_limit(cmd_d2, self.last_cmd_d2, MAX_XY_STEP)
        self.last_cmd_d1 = cmd_d1.copy()
        self.last_cmd_d2 = cmd_d2.copy()

        # === 日志（仅画圆阶段） ===
        if phase == "CIRCLE" and not self.circle_complete:
            self.buffer_csv_row(leader_cmd_world, follower_cmd_world)

        # === 发布指令 ===
        self.pub_traj(1, cmd_d1, cmd_yaw)
        self.pub_traj(2, cmd_d2, cmd_yaw)

    # ==================================================================
    # 发布 / 指令
    # ==================================================================
    def pub_traj(self, id, pos, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        msg.timestamp = self.get_ts(id)
        if id == 1:
            self.pub_traj_d1.publish(msg)
        else:
            self.pub_traj_d2.publish(msg)

    def engage_offboard(self, id):
        self.send_cmd(id, VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                      param1=1.0, param2=6.0)

    def arm(self, id):
        self.send_cmd(id, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                      param1=1.0)

    def send_cmd(self, id, cmd, **kwargs):
        msg = VehicleCommand()
        msg.command = cmd
        msg.param1 = kwargs.get("param1", 0.0)
        msg.param2 = kwargs.get("param2", 0.0)
        msg.target_system = id + 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_ts(id)
        if id == 1:
            self.pub_cmd_d1.publish(msg)
        else:
            self.pub_cmd_d2.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderFollowerCircle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()