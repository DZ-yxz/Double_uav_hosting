#!/usr/bin/env python3

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


class DualDroneCircleControl(Node):
    def __init__(self) -> None:
        # ============================================================
        # 不设置 use_sim_time！定时器用系统时钟，保证稳定触发
        # ============================================================
        super().__init__('dual_drone_tandem_circle_zlog')

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

        # ---------- PX4_1 发布者 ----------
        self.pub_offboard_d1 = self.create_publisher(
            OffboardControlMode,
            '/px4_1/fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj_d1 = self.create_publisher(
            TrajectorySetpoint,
            '/px4_1/fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd_d1 = self.create_publisher(
            VehicleCommand,
            '/px4_1/fmu/in/vehicle_command', qos_pub)

        # ---------- PX4_1 订阅者 ----------
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

        # ---------- PX4_2 发布者 ----------
        self.pub_offboard_d2 = self.create_publisher(
            OffboardControlMode,
            '/px4_2/fmu/in/offboard_control_mode', qos_pub)
        self.pub_traj_d2 = self.create_publisher(
            TrajectorySetpoint,
            '/px4_2/fmu/in/trajectory_setpoint', qos_pub)
        self.pub_cmd_d2 = self.create_publisher(
            VehicleCommand,
            '/px4_2/fmu/in/vehicle_command', qos_pub)

        # ---------- PX4_2 订阅者 ----------
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

        # ---------- 状态变量 ----------
        self.pos_d1 = VehicleLocalPosition()
        self.pos_d2 = VehicleLocalPosition()
        self.status_d1 = VehicleStatus()
        self.status_d2 = VehicleStatus()

        # ============================================================
        # 关键：PX4 时间戳同步
        # ============================================================
        self.px4_ts_d1 = 0      # PX4_1 最新时间戳 (微秒)
        self.px4_ts_d2 = 0      # PX4_2 最新时间戳 (微秒)
        self.ts_received_d1 = False
        self.ts_received_d2 = False

        self.vs_offset_d1 = np.array([2.5, 0.0, 0.0])
        self.vs_offset_d2 = np.array([-2.5, 0.0, 0.0])
        self.theta = -math.pi / 2
        self.theta_start = self.theta
        self.circle_complete = False

        self.last_cmd_d1 = None
        self.last_cmd_d2 = None

        # ---------- 日志 ----------
        timestamp = datetime.now().strftime("%H%M%S")
        self.csv_file = f"log_circle_z_{timestamp}.csv"
        self.csv_buffer = []
        self.csv_flush_interval = 50
        self.csv_header_written = False

        self.tick = 0
        self.takeoff_tick_start = 0
        self.all_systems_go = False
        self.failsafe_count_d1 = 0
        self.failsafe_count_d2 = 0

        # ---------- 独立定时器（系统时钟驱动，稳定触发） ----------
        self.heartbeat_timer = self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_cb)
        self.control_timer = self.create_timer(
            1.0 / CONTROL_RATE, self.control_cb)

        self.get_logger().info(
            f"=== 启动 | 心跳 {HEARTBEAT_RATE}Hz | 控制 {CONTROL_RATE}Hz ===")
        self.get_logger().info(
            "=== 等待 PX4 timesync... ===")

    # ==================================================================
    # 时间同步回调
    # ==================================================================
    def timesync_cb(self, msg, id):
        if id == 1:
            self.px4_ts_d1 = msg.timestamp
            if not self.ts_received_d1:
                self.ts_received_d1 = True
                self.get_logger().info("✓ PX4_1 timesync 已连接")
        else:
            self.px4_ts_d2 = msg.timestamp
            if not self.ts_received_d2:
                self.ts_received_d2 = True
                self.get_logger().info("✓ PX4_2 timesync 已连接")

    def get_ts(self, id):
        """获取对应 PX4 实例的同步时间戳"""
        return self.px4_ts_d1 if id == 1 else self.px4_ts_d2

    # ==================================================================
    # 心跳回调 — 高频独立，使用 PX4 时间戳
    # ==================================================================
    def heartbeat_cb(self):
        # 如果还没收到 timesync，不发送（PX4 还没准备好）
        if not (self.ts_received_d1 and self.ts_received_d2):
            return

        # UAV1 心跳
        msg1 = OffboardControlMode()
        msg1.position = True
        msg1.velocity = False
        msg1.acceleration = False
        msg1.timestamp = self.get_ts(1)
        self.pub_offboard_d1.publish(msg1)

        # UAV2 心跳
        msg2 = OffboardControlMode()
        msg2.position = True
        msg2.velocity = False
        msg2.acceleration = False
        msg2.timestamp = self.get_ts(2)
        self.pub_offboard_d2.publish(msg2)

        # Failsafe 自动恢复
        if self.all_systems_go:
            a1 = (self.status_d1.arming_state == VehicleStatus.ARMING_STATE_ARMED)
            o1 = (self.status_d1.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
            a2 = (self.status_d2.arming_state == VehicleStatus.ARMING_STATE_ARMED)
            o2 = (self.status_d2.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

            if a1 and not o1:
                self.engage_offboard(1)
                self.failsafe_count_d1 += 1
                if self.failsafe_count_d1 <= 5:
                    self.get_logger().warn(
                        f"UAV1 脱离 Offboard，自动恢复 (第{self.failsafe_count_d1}次)")
            if a2 and not o2:
                self.engage_offboard(2)
                self.failsafe_count_d2 += 1
                if self.failsafe_count_d2 <= 5:
                    self.get_logger().warn(
                        f"UAV2 脱离 Offboard，自动恢复 (第{self.failsafe_count_d2}次)")

    # ==================================================================
    # 回调
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
    # 工具函数
    # ==================================================================
    @staticmethod
    def world_to_local(world_pos, uav_home_world):
        return world_pos - uav_home_world

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
    # 日志（缓冲写入）
    # ==================================================================
    def buffer_csv_row(self, d1_cmd, d2_cmd):
        if self.circle_complete:
            return
        d1_cmd_w = d1_cmd[:2] + UAV1_HOME_WORLD[:2]
        d2_cmd_w = d2_cmd[:2] + UAV2_HOME_WORLD[:2]
        d1_act_w = (np.array([self.pos_d1.x, self.pos_d1.y])
                    + UAV1_HOME_WORLD[:2])
        d2_act_w = (np.array([self.pos_d2.x, self.pos_d2.y])
                    + UAV2_HOME_WORLD[:2])

        self.csv_buffer.append([
            self.tick, f"{self.theta:.4f}",
            f"{d1_cmd_w[0]:.3f}", f"{d1_cmd_w[1]:.3f}", f"{d1_cmd[2]:.3f}",
            f"{d1_act_w[0]:.3f}", f"{d1_act_w[1]:.3f}",
            f"{self.pos_d1.z:.3f}",
            f"{d2_cmd_w[0]:.3f}", f"{d2_cmd_w[1]:.3f}", f"{d2_cmd[2]:.3f}",
            f"{d2_act_w[0]:.3f}", f"{d2_act_w[1]:.3f}",
            f"{self.pos_d2.z:.3f}",
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
                    "d1_x_cmd", "d1_y_cmd", "d1_z_cmd",
                    "d1_x_act", "d1_y_act", "d1_z_act",
                    "d2_x_cmd", "d2_y_cmd", "d2_z_cmd",
                    "d2_x_act", "d2_y_act", "d2_z_act",
                ])
                self.csv_header_written = True
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()

    # ==================================================================
    # 控制主循环
    # ==================================================================
    def control_cb(self):
        # 等待两个 PX4 实例都完成时间同步
        if not (self.ts_received_d1 and self.ts_received_d2):
            return

        self.tick += 1

        cmd_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        cmd_yaw = 0.0
        phase = "INIT"

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

        # ==== 阶段 1：起飞 ====
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

        # ==== 阶段 2：过渡 ====
        elif elapsed < TAKEOFF_STABLE_TICKS + TRANSITION_TICKS:
            phase = "TRANSITION"
            t = self.smooth_step(
                (elapsed - TAKEOFF_STABLE_TICKS) / float(TRANSITION_TICKS))

            hover_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            hover_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
            vc_start = np.array([
                CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta_start),
                CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta_start),
                TAKEOFF_HEIGHT,
            ])
            target_d1 = self.world_to_local(
                vc_start + self.vs_offset_d1, UAV1_HOME_WORLD)
            target_d2 = self.world_to_local(
                vc_start + self.vs_offset_d2, UAV2_HOME_WORLD)
            cmd_d1 = hover_d1 * (1.0 - t) + target_d1 * t
            cmd_d2 = hover_d2 * (1.0 - t) + target_d2 * t

        # ==== 阶段 3：画圆 ====
        else:
            phase = "CIRCLE"
            circle_ticks = (elapsed - TAKEOFF_STABLE_TICKS
                            - TRANSITION_TICKS)
            if circle_ticks < RAMP_TICKS:
                spd = CIRCLE_SPEED * self.smooth_step(
                    circle_ticks / float(RAMP_TICKS))
            else:
                spd = CIRCLE_SPEED

            self.theta += spd

            if ((self.theta - self.theta_start) >= 2.0 * math.pi + 0.5
                    and not self.circle_complete):
                self.circle_complete = True
                self.flush_csv()
                self.get_logger().info(
                    f"=== 一圈完成! 日志: {self.csv_file} | "
                    f"Failsafe: D1={self.failsafe_count_d1} "
                    f"D2={self.failsafe_count_d2} ===")

            vc_pos = np.array([
                CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta),
                CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta),
                TAKEOFF_HEIGHT,
            ])
            cmd_d1 = self.world_to_local(
                vc_pos + self.vs_offset_d1, UAV1_HOME_WORLD)
            cmd_d2 = self.world_to_local(
                vc_pos + self.vs_offset_d2, UAV2_HOME_WORLD)

            if self.tick % 50 == 0:
                progress = ((self.theta - self.theta_start)
                            / (2.0 * math.pi) * 100)
                self.get_logger().info(
                    f"圈进度:{progress:.1f}% | "
                    f"Z1:{self.pos_d1.z:.2f} Z2:{self.pos_d2.z:.2f}")

        # === 安全 ===
        cmd_d1[2] = self.clamp_z(cmd_d1[2])
        cmd_d2[2] = self.clamp_z(cmd_d2[2])
        cmd_d1 = self.rate_limit(cmd_d1, self.last_cmd_d1, MAX_XY_STEP)
        cmd_d2 = self.rate_limit(cmd_d2, self.last_cmd_d2, MAX_XY_STEP)
        self.last_cmd_d1 = cmd_d1.copy()
        self.last_cmd_d2 = cmd_d2.copy()

        if phase == "CIRCLE" and not self.circle_complete:
            self.buffer_csv_row(cmd_d1, cmd_d2)

        self.pub_traj(1, cmd_d1, cmd_yaw)
        self.pub_traj(2, cmd_d2, cmd_yaw)

    # ==================================================================
    # 发布轨迹
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

    # ==================================================================
    # 指令
    # ==================================================================
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
    node = DualDroneCircleControl()
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