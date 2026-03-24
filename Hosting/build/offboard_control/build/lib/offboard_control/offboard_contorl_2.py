#!/usr/bin/env python3
"""
基于行为法 (Behavior-Based) 双无人机编队画圆

核心思想：每架无人机同时受多个"行为"驱动，每个行为产生一个速度向量，
通过加权融合得到最终控制指令。

行为分解：
  1. 轨迹跟踪行为 (Track)    — 跟踪各自的圆形轨迹参考点
  2. 编队保持行为 (Formation) — 维持两机之间的期望相对位置
  3. 避碰行为 (Avoidance)     — 两机距离过近时产生排斥力
  4. 高度保持行为 (Altitude)  — 将高度锁定在目标值
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

# === 画圆参数 ===
CIRCLE_RADIUS = 5.0
CIRCLE_CENTER_X = 0.0
CIRCLE_CENTER_Y = 5.0
CIRCLE_SPEED = 0.015

# === 编队参数（UAV2 相对 UAV1 的期望偏移，世界坐标） ===
FORMATION_OFFSET = np.array([-5.0, 0.0, 0.0])

# === 行为权重 ===
W_TRACK = 1.0           # 轨迹跟踪
W_FORMATION = 0.8       # 编队保持
W_AVOIDANCE = 1.5       # 避碰（最高优先级）
W_ALTITUDE = 1.2        # 高度保持

# === 避碰参数 ===
AVOID_RADIUS = 3.0      # 触发避碰的距离（米）
AVOID_DEADZONE = 1.5    # 避碰力最大时的距离（米）

# === 行为输出限幅 ===
MAX_BEHAVIOR_VEL = 2.0  # 单个行为最大输出速度 (m/tick)
MAX_TOTAL_VEL = 3.0     # 融合后最大速度 (m/tick)

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


class BehaviorBasedCircle(Node):
    def __init__(self) -> None:
        super().__init__('behavior_based_circle')

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

        # ==================== PX4_1 (UAV1) ====================
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

        # ==================== PX4_2 (UAV2) ====================
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

        # ==================== 状态 ====================
        self.pos_d1 = VehicleLocalPosition()
        self.pos_d2 = VehicleLocalPosition()
        self.status_d1 = VehicleStatus()
        self.status_d2 = VehicleStatus()

        self.px4_ts_d1 = 0
        self.px4_ts_d2 = 0
        self.ts_received_d1 = False
        self.ts_received_d2 = False

        self.theta = -math.pi / 2
        self.theta_start = self.theta
        self.circle_complete = False

        self.last_cmd_d1 = None
        self.last_cmd_d2 = None

        # ---------- 日志 ----------
        timestamp = datetime.now().strftime("%H%M%S")
        self.csv_file = f"log_behavior_circle_{timestamp}.csv"
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
        self.get_logger().info("  Behavior-Based 编队画圆控制器")
        self.get_logger().info(f"  权重: Track={W_TRACK} Formation={W_FORMATION}"
                               f" Avoid={W_AVOIDANCE} Alt={W_ALTITUDE}")
        self.get_logger().info(f"  避碰距离={AVOID_RADIUS}m  编队偏移={FORMATION_OFFSET}")
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
                self.get_logger().info("✓ UAV1 timesync 已连接")
        else:
            self.px4_ts_d2 = msg.timestamp
            if not self.ts_received_d2:
                self.ts_received_d2 = True
                self.get_logger().info("✓ UAV2 timesync 已连接")

    def get_ts(self, id):
        return self.px4_ts_d1 if id == 1 else self.px4_ts_d2

    # ==================================================================
    # 心跳
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

        if self.all_systems_go:
            for id in [1, 2]:
                s = self.status_d1 if id == 1 else self.status_d2
                if (s.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                        s.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
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
    def local_to_world(local_pos, home):
        return local_pos + home

    @staticmethod
    def world_to_local(world_pos, home):
        return world_pos - home

    def get_world_pos(self, id):
        """获取指定无人机的世界坐标 [x, y, z]"""
        if id == 1:
            return self.local_to_world(
                np.array([self.pos_d1.x, self.pos_d1.y, self.pos_d1.z]),
                UAV1_HOME_WORLD)
        else:
            return self.local_to_world(
                np.array([self.pos_d2.x, self.pos_d2.y, self.pos_d2.z]),
                UAV2_HOME_WORLD)

    # ==================================================================
    #                       四个行为模块
    # ==================================================================

    def behavior_track(self, current_world, target_world):
        """
        行为 1：轨迹跟踪
        产生一个指向轨迹参考点的速度向量
        """
        diff = target_world - current_world
        return self.clip_vec(diff, MAX_BEHAVIOR_VEL)

    def behavior_formation(self, my_world, other_world, desired_offset):
        """
        行为 2：编队保持
        期望位置 = 另一架的实际位置 + 偏移
        产生一个指向期望编队位置的速度向量
        """
        desired_pos = other_world + desired_offset
        diff = desired_pos - my_world
        return self.clip_vec(diff, MAX_BEHAVIOR_VEL)

    def behavior_avoidance(self, my_world, other_world):
        """
        行为 3：避碰
        当两机 XY 距离 < AVOID_RADIUS 时产生排斥力
        距离越近，排斥力越大（反比关系）
        """
        diff = my_world[:2] - other_world[:2]
        dist = np.linalg.norm(diff)

        if dist >= AVOID_RADIUS or dist < 0.01:
            return np.array([0.0, 0.0, 0.0])

        # 排斥力：距离越近越大
        # 在 AVOID_DEADZONE 时力最大，在 AVOID_RADIUS 时力为 0
        strength = (AVOID_RADIUS - dist) / (AVOID_RADIUS - AVOID_DEADZONE)
        strength = min(strength, 1.0) * MAX_BEHAVIOR_VEL

        direction = diff / dist  # 远离对方的单位向量
        return np.array([direction[0] * strength,
                         direction[1] * strength,
                         0.0])

    def behavior_altitude(self, current_z, target_z):
        """
        行为 4：高度保持
        产生一个纠正高度偏差的速度向量
        """
        err = target_z - current_z
        vel_z = np.clip(err, -MAX_BEHAVIOR_VEL, MAX_BEHAVIOR_VEL)
        return np.array([0.0, 0.0, vel_z])

    # ==================================================================
    # 行为融合：加权求和 → 最终指令
    # ==================================================================
    def fuse_behaviors(self, current_world, track_ref, other_world,
                       formation_offset, target_z):
        """
        将所有行为的输出加权融合，得到最终的位置增量

        current_world: 本机当前世界坐标
        track_ref:     轨迹参考点（世界坐标）
        other_world:   另一架的实际世界坐标
        formation_offset: 本机相对于另一架的期望偏移
        target_z:      目标高度
        """
        v_track = self.behavior_track(current_world, track_ref)
        v_form = self.behavior_formation(
            current_world, other_world, formation_offset)
        v_avoid = self.behavior_avoidance(current_world, other_world)
        v_alt = self.behavior_altitude(current_world[2], target_z)

        # 加权融合
        v_total = (W_TRACK * v_track
                   + W_FORMATION * v_form
                   + W_AVOIDANCE * v_avoid
                   + W_ALTITUDE * v_alt)

        # 限幅
        v_total = self.clip_vec(v_total, MAX_TOTAL_VEL)

        # 最终目标 = 当前位置 + 融合速度
        target = current_world + v_total

        return target, v_track, v_form, v_avoid, v_alt

    # ==================================================================
    # 工具函数
    # ==================================================================
    @staticmethod
    def clip_vec(v, max_norm):
        """将向量限幅到最大模长"""
        n = np.linalg.norm(v)
        if n > max_norm and n > 1e-6:
            return v * (max_norm / n)
        return v.copy()

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
    # 轨迹参考点计算
    # ==================================================================
    def get_circle_ref(self, offset=np.zeros(3)):
        """圆形轨迹上的参考点 + 偏移（世界坐标）"""
        x = CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta)
        y = CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta)
        return np.array([x, y, TAKEOFF_HEIGHT]) + offset

    # ==================================================================
    # 日志
    # ==================================================================
    def buffer_csv_row(self, d1_cmd_w, d2_cmd_w, d1_behaviors, d2_behaviors):
        if self.circle_complete:
            return

        d1_act = self.get_world_pos(1)
        d2_act = self.get_world_pos(2)
        form_err = np.linalg.norm(
            d2_act[:2] - (d1_act[:2] + FORMATION_OFFSET[:2]))

        # 各行为的模长（用于分析哪个行为主导）
        row = [
            self.tick, f"{self.theta:.4f}",
            # UAV1
            f"{d1_cmd_w[0]:.3f}", f"{d1_cmd_w[1]:.3f}", f"{d1_cmd_w[2]:.3f}",
            f"{d1_act[0]:.3f}", f"{d1_act[1]:.3f}", f"{d1_act[2]:.3f}",
            # UAV2
            f"{d2_cmd_w[0]:.3f}", f"{d2_cmd_w[1]:.3f}", f"{d2_cmd_w[2]:.3f}",
            f"{d2_act[0]:.3f}", f"{d2_act[1]:.3f}", f"{d2_act[2]:.3f}",
            # 编队误差
            f"{form_err:.3f}",
            # UAV1 各行为力的模
            f"{np.linalg.norm(d1_behaviors[0]):.3f}",
            f"{np.linalg.norm(d1_behaviors[1]):.3f}",
            f"{np.linalg.norm(d1_behaviors[2]):.3f}",
            f"{np.linalg.norm(d1_behaviors[3]):.3f}",
            # UAV2 各行为力的模
            f"{np.linalg.norm(d2_behaviors[0]):.3f}",
            f"{np.linalg.norm(d2_behaviors[1]):.3f}",
            f"{np.linalg.norm(d2_behaviors[2]):.3f}",
            f"{np.linalg.norm(d2_behaviors[3]):.3f}",
        ]
        self.csv_buffer.append(row)
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
                    "d1_x_cmd", "d1_y_cmd", "d1_z_cmd",
                    "d1_x_act", "d1_y_act", "d1_z_act",
                    "d2_x_cmd", "d2_y_cmd", "d2_z_cmd",
                    "d2_x_act", "d2_y_act", "d2_z_act",
                    "formation_error",
                    "d1_track", "d1_formation", "d1_avoidance", "d1_altitude",
                    "d2_track", "d2_formation", "d2_avoidance", "d2_altitude",
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

        cmd_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_yaw = 0.0
        phase = "INIT"

        # 行为输出占位（用于日志）
        d1_behaviors = [np.zeros(3)] * 4
        d2_behaviors = [np.zeros(3)] * 4
        d1_cmd_world = UAV1_HOME_WORLD + cmd_d1
        d2_cmd_world = UAV2_HOME_WORLD + cmd_d2

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
        # 阶段 1：起飞
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
        # 阶段 2：过渡
        # ==============================================================
        elif elapsed < TAKEOFF_STABLE_TICKS + TRANSITION_TICKS:
            phase = "TRANSITION"
            t = self.smooth_step(
                (elapsed - TAKEOFF_STABLE_TICKS) / float(TRANSITION_TICKS))

            # UAV1 过渡到圆的起始点
            hover_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            ref_d1_world = self.get_circle_ref()
            ref_d1_local = self.world_to_local(ref_d1_world, UAV1_HOME_WORLD)
            cmd_d1 = hover_d1 * (1.0 - t) + ref_d1_local * t

            # UAV2 过渡到编队起始位置
            hover_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            ref_d2_world = self.get_circle_ref(FORMATION_OFFSET)
            ref_d2_local = self.world_to_local(ref_d2_world, UAV2_HOME_WORLD)
            cmd_d2 = hover_d2 * (1.0 - t) + ref_d2_local * t

            d1_cmd_world = UAV1_HOME_WORLD + cmd_d1
            d2_cmd_world = UAV2_HOME_WORLD + cmd_d2

        # ==============================================================
        # 阶段 3：行为法画圆
        # ==============================================================
        else:
            phase = "CIRCLE"
            circle_ticks = (elapsed - TAKEOFF_STABLE_TICKS
                            - TRANSITION_TICKS)

            # 速度斜坡
            if circle_ticks < RAMP_TICKS:
                spd = CIRCLE_SPEED * self.smooth_step(
                    circle_ticks / float(RAMP_TICKS))
            else:
                spd = CIRCLE_SPEED
            self.theta += spd

            # 检测一圈
            if ((self.theta - self.theta_start) >= 2.0 * math.pi + 0.5
                    and not self.circle_complete):
                self.circle_complete = True
                self.flush_csv()
                self.get_logger().info(
                    f"=== 一圈完成! 日志: {self.csv_file} ===")

            # 获取两机实际世界坐标
            pos1_world = self.get_world_pos(1)
            pos2_world = self.get_world_pos(2)

            # --- UAV1 的轨迹参考点（圆上的点） ---
            ref1_world = self.get_circle_ref()

            # --- UAV2 的轨迹参考点（圆 + 编队偏移） ---
            ref2_world = self.get_circle_ref(FORMATION_OFFSET)

            # ========================================
            # UAV1 行为融合
            # ========================================
            # UAV1 看 UAV2：编队偏移取反（UAV2 应该在我的 FORMATION_OFFSET 方向）
            d1_cmd_world, v1_trk, v1_frm, v1_avd, v1_alt = \
                self.fuse_behaviors(
                    current_world=pos1_world,
                    track_ref=ref1_world,
                    other_world=pos2_world,
                    formation_offset=-FORMATION_OFFSET,  # UAV1 视角：UAV2 在反方向
                    target_z=TAKEOFF_HEIGHT,
                )
            d1_behaviors = [v1_trk, v1_frm, v1_avd, v1_alt]

            # ========================================
            # UAV2 行为融合
            # ========================================
            d2_cmd_world, v2_trk, v2_frm, v2_avd, v2_alt = \
                self.fuse_behaviors(
                    current_world=pos2_world,
                    track_ref=ref2_world,
                    other_world=pos1_world,
                    formation_offset=FORMATION_OFFSET,   # UAV2 视角：应在 UAV1 + offset
                    target_z=TAKEOFF_HEIGHT,
                )
            d2_behaviors = [v2_trk, v2_frm, v2_avd, v2_alt]

            # 世界坐标 → 本地坐标
            cmd_d1 = self.world_to_local(d1_cmd_world, UAV1_HOME_WORLD)
            cmd_d2 = self.world_to_local(d2_cmd_world, UAV2_HOME_WORLD)

            # 日志输出
            if self.tick % 50 == 0:
                progress = ((self.theta - self.theta_start)
                            / (2.0 * math.pi) * 100)
                form_err = np.linalg.norm(
                    pos2_world[:2] - (pos1_world[:2] + FORMATION_OFFSET[:2]))
                self.get_logger().info(
                    f"圈:{progress:.1f}% | "
                    f"Z1:{pos1_world[2]:.2f} Z2:{pos2_world[2]:.2f} | "
                    f"编队误差:{form_err:.2f}m | "
                    f"避碰力: D1={np.linalg.norm(v1_avd):.2f} "
                    f"D2={np.linalg.norm(v2_avd):.2f}")

        # === 安全 ===
        cmd_d1[2] = self.clamp_z(cmd_d1[2])
        cmd_d2[2] = self.clamp_z(cmd_d2[2])
        cmd_d1 = self.rate_limit(cmd_d1, self.last_cmd_d1, MAX_XY_STEP)
        cmd_d2 = self.rate_limit(cmd_d2, self.last_cmd_d2, MAX_XY_STEP)
        self.last_cmd_d1 = cmd_d1.copy()
        self.last_cmd_d2 = cmd_d2.copy()

        # === 日志 ===
        if phase == "CIRCLE" and not self.circle_complete:
            self.buffer_csv_row(
                UAV1_HOME_WORLD + cmd_d1,
                UAV2_HOME_WORLD + cmd_d2,
                d1_behaviors, d2_behaviors)

        # === 发布 ===
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
    node = BehaviorBasedCircle()
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