#!/usr/bin/env python3
"""
虚拟结构法 + 模糊自适应权重控制 双无人机编队画圆

模糊控制器：
  输入1: e_p  — 位置误差 (m)
  输入2: e_a  — 姿态误差 (rad)
  输入3: de_p — 位置误差变化率 (m/tick)
  输出1: α    — 位置控制权重系数 [0, 1]
  输出2: β    — 姿态控制权重系数 [0, 1]

逻辑：
  位置误差大 → α↑ β↓ → 优先收敛位置
  位置误差小 → α↓ β↑ → 优先保持姿态精度
  误差变化率为负（误差在缩小） → 提前降低 α，避免超调
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


# =====================================================================
#                         模糊控制器
# =====================================================================

class FuzzyController:
    """
    三输入二输出模糊控制器

    输入:
        e_p  : 位置误差归一化值  [-1, 1]
        e_a  : 姿态误差归一化值  [-1, 1]
        de_p : 位置误差变化率归一化值 [-1, 1]

    输出:
        alpha : 位置控制权重 [0, 1]
        beta  : 姿态控制权重 [0, 1]

    隶属度函数: Gaussian
    语言变量:   NB, NM, NS, Z, PS, PM, PB (7个)
    推理方法:   Mamdani
    去模糊化:   加权平均法
    """

    # 7 个语言变量对应的中心值 (在 [-1, 1] 上均匀分布)
    CENTERS = {
        'NB': -1.0,
        'NM': -0.667,
        'NS': -0.333,
        'Z':   0.0,
        'PS':  0.333,
        'PM':  0.667,
        'PB':  1.0,
    }
    LABELS = ['NB', 'NM', 'NS', 'Z', 'PS', 'PM', 'PB']

    # Gaussian 标准差（控制隶属度函数的宽度，越小越尖锐）
    SIGMA = 0.18

    # 输出语言变量映射到 [0, 1] 的值
    OUT_MAP = {
        'NB': 0.0,
        'NM': 0.167,
        'NS': 0.333,
        'Z':  0.5,
        'PS': 0.667,
        'PM': 0.833,
        'PB': 1.0,
    }

    def __init__(self):
        # ============================================================
        # 模糊规则表
        # ============================================================
        # α 规则表 (位置权重): 位置误差大 → α 大
        #
        # 三维规则: rule_alpha[e_p][e_a][de_p] = 输出语言变量
        #
        # 设计原则:
        #   1. e_p 大 → α 大（强力追位置）
        #   2. e_p 小 + e_a 大 → α 适中（留余量给姿态）
        #   3. de_p 负（误差在缩小）→ α 适当降低（防超调）
        #   4. de_p 正（误差在增大）→ α 适当提高（加速追踪）

        self.rules_alpha = self._build_alpha_rules()
        self.rules_beta = self._build_beta_rules()

    def _build_alpha_rules(self):
        """
        构建 α (位置权重) 的三维规则表

        规则逻辑:
        - 基础层 (e_p × e_a): 位置误差越大 α 越大
        - 调节层 (de_p): 误差缩小时降一档，误差增大时升一档
        """
        # 先定义二维基础表: alpha_base[e_p][e_a]
        # 行: e_p (NB→PB), 列: e_a (NB→PB)
        alpha_base = [
            #  e_a:  NB    NM    NS    Z     PS    PM    PB
            ['PB', 'PB', 'PM', 'PM', 'PM', 'PB', 'PB'],  # e_p = NB
            ['PB', 'PM', 'PM', 'PM', 'PM', 'PM', 'PB'],  # e_p = NM
            ['PM', 'PM', 'PS', 'PS', 'PS', 'PM', 'PM'],  # e_p = NS
            ['PS', 'PS', 'NS', 'NB', 'NS', 'PS', 'PS'],  # e_p = Z
            ['PM', 'PM', 'PS', 'PS', 'PS', 'PM', 'PM'],  # e_p = PS
            ['PB', 'PM', 'PM', 'PM', 'PM', 'PM', 'PB'],  # e_p = PM
            ['PB', 'PB', 'PM', 'PM', 'PM', 'PB', 'PB'],  # e_p = PB
        ]

        # 扩展为三维: 根据 de_p 做升/降档调节
        # de_p 负 (误差缩小) → 降一档
        # de_p 正 (误差增大) → 升一档
        # de_p 接近零 → 保持不变
        de_shift = {
            'NB': -2, 'NM': -1, 'NS': -1,
            'Z': 0,
            'PS': 1, 'PM': 1, 'PB': 2,
        }

        rules = {}
        for i, ep_label in enumerate(self.LABELS):
            for j, ea_label in enumerate(self.LABELS):
                base_label = alpha_base[i][j]
                base_idx = self.LABELS.index(base_label)
                for dep_label in self.LABELS:
                    shift = de_shift[dep_label]
                    new_idx = max(0, min(6, base_idx + shift))
                    rules[(ep_label, ea_label, dep_label)] = \
                        self.LABELS[new_idx]
        return rules

    def _build_beta_rules(self):
        """
        构建 β (姿态权重) 的三维规则表

        逻辑与 α 互补:
        - 位置误差大 → β 小（先追位置）
        - 位置误差小 + 姿态误差大 → β 大（精调姿态）
        - de_p 负（位置在收敛）→ β 提高（可以开始关注姿态了）
        """
        beta_base = [
            #  e_a:  NB    NM    NS    Z     PS    PM    PB
            ['NB', 'NB', 'NS', 'NS', 'NS', 'NB', 'NB'],  # e_p = NB
            ['NB', 'NS', 'NS', 'Z',  'NS', 'NS', 'NB'],  # e_p = NM
            ['NS', 'Z',  'PS', 'PS', 'PS', 'Z',  'NS'],  # e_p = NS
            ['PS', 'PM', 'PM', 'PB', 'PM', 'PM', 'PS'],  # e_p = Z
            ['NS', 'Z',  'PS', 'PS', 'PS', 'Z',  'NS'],  # e_p = PS
            ['NB', 'NS', 'NS', 'Z',  'NS', 'NS', 'NB'],  # e_p = PM
            ['NB', 'NB', 'NS', 'NS', 'NS', 'NB', 'NB'],  # e_p = PB
        ]

        # de_p 对 β 的调节方向与 α 相反:
        # de_p 负 (位置在收敛) → β 升一档（可以开始精调姿态）
        # de_p 正 (位置在发散) → β 降一档（先顾位置）
        de_shift = {
            'NB': 2, 'NM': 1, 'NS': 1,
            'Z': 0,
            'PS': -1, 'PM': -1, 'PB': -2,
        }

        rules = {}
        for i, ep_label in enumerate(self.LABELS):
            for j, ea_label in enumerate(self.LABELS):
                base_label = beta_base[i][j]
                base_idx = self.LABELS.index(base_label)
                for dep_label in self.LABELS:
                    shift = de_shift[dep_label]
                    new_idx = max(0, min(6, base_idx + shift))
                    rules[(ep_label, ea_label, dep_label)] = \
                        self.LABELS[new_idx]
        return rules

    @classmethod
    def gaussian_mf(cls, x, center, sigma=None):
        """Gaussian 隶属度函数"""
        if sigma is None:
            sigma = cls.SIGMA
        return math.exp(-0.5 * ((x - center) / sigma) ** 2)

    def fuzzify(self, x):
        """
        模糊化: 计算输入 x 对所有语言变量的隶属度

        返回: dict {label: membership_degree}
        """
        memberships = {}
        for label, center in self.CENTERS.items():
            memberships[label] = self.gaussian_mf(x, center)
        return memberships

    def infer(self, ep_memberships, ea_memberships, dep_memberships,
              rules):
        """
        模糊推理 (Mamdani)

        对所有规则组合:
          1. 取各输入隶属度的最小值 (AND 操作)
          2. 得到每条规则的触发强度
          3. 将同一输出标签的触发强度取最大值 (OR 操作)

        返回: dict {output_label: max_firing_strength}
        """
        output_strengths = {label: 0.0 for label in self.LABELS}

        for ep_label, ep_mu in ep_memberships.items():
            if ep_mu < 1e-6:
                continue
            for ea_label, ea_mu in ea_memberships.items():
                if ea_mu < 1e-6:
                    continue
                for dep_label, dep_mu in dep_memberships.items():
                    if dep_mu < 1e-6:
                        continue

                    # AND: 取最小
                    firing = min(ep_mu, ea_mu, dep_mu)

                    # 查规则表
                    out_label = rules[(ep_label, ea_label, dep_label)]

                    # OR: 取最大
                    if firing > output_strengths[out_label]:
                        output_strengths[out_label] = firing

        return output_strengths

    def defuzzify(self, output_strengths):
        """
        去模糊化: 加权平均法

        output = Σ(strength_i × value_i) / Σ(strength_i)
        """
        numerator = 0.0
        denominator = 0.0
        for label, strength in output_strengths.items():
            value = self.OUT_MAP[label]
            numerator += strength * value
            denominator += strength

        if denominator < 1e-9:
            return 0.5  # 默认中间值
        return numerator / denominator

    def compute(self, e_p_norm, e_a_norm, de_p_norm):
        """
        完整模糊推理流程

        输入 (均已归一化到 [-1, 1]):
            e_p_norm  : 位置误差
            e_a_norm  : 姿态误差
            de_p_norm : 位置误差变化率

        输出:
            alpha : 位置权重 [0, 1]
            beta  : 姿态权重 [0, 1]
        """
        # 钳位到 [-1, 1]
        e_p_norm = max(-1.0, min(1.0, e_p_norm))
        e_a_norm = max(-1.0, min(1.0, e_a_norm))
        de_p_norm = max(-1.0, min(1.0, de_p_norm))

        # 1. 模糊化
        ep_mu = self.fuzzify(e_p_norm)
        ea_mu = self.fuzzify(e_a_norm)
        dep_mu = self.fuzzify(de_p_norm)

        # 2. 模糊推理
        alpha_strengths = self.infer(ep_mu, ea_mu, dep_mu, self.rules_alpha)
        beta_strengths = self.infer(ep_mu, ea_mu, dep_mu, self.rules_beta)

        # 3. 去模糊化
        alpha = self.defuzzify(alpha_strengths)
        beta = self.defuzzify(beta_strengths)

        return alpha, beta


# =====================================================================
#                      归一化参数
# =====================================================================
# 位置误差归一化: e_p / E_P_MAX → [-1, 1]
# 当位置误差达到 E_P_MAX 时，归一化值为 ±1
E_P_MAX = 2.0       # 米
E_A_MAX = math.pi    # 弧度 (180°)
DE_P_MAX =  0.05     # 米/tick


# =====================================================================
#                      场景配置
# =====================================================================
UAV1_HOME_WORLD = np.array([2.5, 0.0, 0.0])
UAV2_HOME_WORLD = np.array([-2.5, 0.0, 0.0])
TAKEOFF_HEIGHT = -5.0

CIRCLE_RADIUS = 5.0
CIRCLE_CENTER_X = 0.0
CIRCLE_CENTER_Y = 5.0
CIRCLE_SPEED = 0.015

Z_MIN = -7.0
Z_MAX = -2.0
MAX_XY_STEP = 0.5

TAKEOFF_STABLE_TICKS = 200
TRANSITION_TICKS = 200
RAMP_TICKS = 200

HEARTBEAT_RATE = 30.0
CONTROL_RATE = 10.0

# 编队偏移
VS_OFFSET_D1 = np.array([2.5, 0.0, 0.0])
VS_OFFSET_D2 = np.array([-2.5, 0.0, 0.0])

# 期望偏航
TARGET_YAW = 0.0


# =====================================================================
#                         主节点
# =====================================================================

class FuzzyVirtualStructureCircle(Node):
    def __init__(self) -> None:
        super().__init__('fuzzy_vs_circle')

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

        # PX4_1
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

        # PX4_2
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

        # 状态
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

        # 上一次位置误差（用于计算 de_p）
        self.prev_ep_d1 = 0.0
        self.prev_ep_d2 = 0.0

        # ============================================================
        # 模糊控制器（两架无人机各一个实例，共享规则但独立状态）
        # ============================================================
        self.fuzzy = FuzzyController()

        # 日志
        timestamp = datetime.now().strftime("%H%M%S")
        self.csv_file = f"log_fuzzy_vs_{timestamp}.csv"
        self.csv_buffer = []
        self.csv_flush_interval = 50
        self.csv_header_written = False

        self.tick = 0
        self.takeoff_tick_start = 0
        self.all_systems_go = False

        self.heartbeat_timer = self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_cb)
        self.control_timer = self.create_timer(
            1.0 / CONTROL_RATE, self.control_cb)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  虚拟结构法 + 模糊自适应权重控制器")
        self.get_logger().info(f"  归一化参数: E_P_MAX={E_P_MAX}m"
                               f" E_A_MAX={math.degrees(E_A_MAX):.0f}°"
                               f" DE_P_MAX={DE_P_MAX}m/tick")
        self.get_logger().info(f"  日志: {self.csv_file} (仅第一圈)")
        self.get_logger().info("=" * 60)

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
    # 坐标 / 工具
    # ==================================================================
    @staticmethod
    def world_to_local(world_pos, home):
        return world_pos - home

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

    @staticmethod
    def normalize_angle(a):
        """将角度归一化到 [-π, π]"""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def get_actual_yaw(self, id):
        """获取无人机的实际偏航角"""
        pos = self.pos_d1 if id == 1 else self.pos_d2
        return float(pos.heading)

    # ==================================================================
    # 模糊自适应权重计算
    # ==================================================================
    def compute_fuzzy_cmd(self, id, desired_pos_local, desired_yaw):
        """
        对一架无人机应用模糊控制:

        1. 计算位置误差 e_p、姿态误差 e_a、位置误差变化率 de_p
        2. 归一化
        3. 模糊推理 → 得到 α, β
        4. 用 α, β 调节最终指令:
             final_pos = actual + α × (desired_pos - actual)
             final_yaw = actual_yaw + β × (desired_yaw - actual_yaw)

        返回: (final_pos_local, final_yaw, alpha, beta, e_p, e_a, de_p)
        """
        pos = self.pos_d1 if id == 1 else self.pos_d2
        actual_local = np.array([pos.x, pos.y, pos.z])
        actual_yaw = self.get_actual_yaw(id)

        # --- 误差计算 ---
        pos_error_vec = desired_pos_local - actual_local
        e_p = np.linalg.norm(pos_error_vec[:2])   # XY 位置误差（标量）
        e_a = self.normalize_angle(desired_yaw - actual_yaw)  # 姿态误差

        # 位置误差变化率
        prev_ep = self.prev_ep_d1 if id == 1 else self.prev_ep_d2
        de_p = e_p - prev_ep
        if id == 1:
            self.prev_ep_d1 = e_p
        else:
            self.prev_ep_d2 = e_p

        # --- 归一化到 [-1, 1] ---
        # e_p 是正值，但模糊控制器是对称设计，取符号
        # 位置误差用模的归一化（始终为正 → 映射到 [0, 1]）
        ep_norm = min(e_p / E_P_MAX, 1.0)     # [0, 1]
        ea_norm = e_a / E_A_MAX                 # [-1, 1]
        dep_norm = de_p / DE_P_MAX              # [-1, 1]

        # --- 模糊推理 ---
        alpha, beta = self.fuzzy.compute(ep_norm, ea_norm, dep_norm)

        # --- 应用权重生成最终指令 ---
        # 位置：实际位置 + α × 误差向量
        final_pos = actual_local + alpha * pos_error_vec
        # Z 轴始终用期望高度（不受模糊控制影响，保证安全）
        final_pos[2] = desired_pos_local[2]

        # 姿态：实际偏航 + β × 偏航误差
        final_yaw = actual_yaw + beta * e_a

        return final_pos, final_yaw, alpha, beta, e_p, e_a, de_p

    # ==================================================================
    # 日志
    # ==================================================================
    def buffer_csv_row(self, d1_data, d2_data):
        if self.circle_complete:
            return

        # d1_data / d2_data = (cmd_local, yaw, alpha, beta, e_p, e_a, de_p)
        d1_cmd_l, d1_yaw, a1, b1, ep1, ea1, dep1 = d1_data
        d2_cmd_l, d2_yaw, a2, b2, ep2, ea2, dep2 = d2_data

        d1_act = np.array([self.pos_d1.x, self.pos_d1.y, self.pos_d1.z])
        d2_act = np.array([self.pos_d2.x, self.pos_d2.y, self.pos_d2.z])
        d1_cmd_w = d1_cmd_l + UAV1_HOME_WORLD
        d2_cmd_w = d2_cmd_l + UAV2_HOME_WORLD
        d1_act_w = d1_act + UAV1_HOME_WORLD
        d2_act_w = d2_act + UAV2_HOME_WORLD

        self.csv_buffer.append([
            self.tick, f"{self.theta:.4f}",
            # UAV1 位置
            f"{d1_cmd_w[0]:.3f}", f"{d1_cmd_w[1]:.3f}", f"{d1_cmd_w[2]:.3f}",
            f"{d1_act_w[0]:.3f}", f"{d1_act_w[1]:.3f}", f"{d1_act_w[2]:.3f}",
            # UAV1 模糊输出
            f"{a1:.4f}", f"{b1:.4f}",
            f"{ep1:.4f}", f"{ea1:.4f}", f"{dep1:.4f}",
            # UAV2 位置
            f"{d2_cmd_w[0]:.3f}", f"{d2_cmd_w[1]:.3f}", f"{d2_cmd_w[2]:.3f}",
            f"{d2_act_w[0]:.3f}", f"{d2_act_w[1]:.3f}", f"{d2_act_w[2]:.3f}",
            # UAV2 模糊输出
            f"{a2:.4f}", f"{b2:.4f}",
            f"{ep2:.4f}", f"{ea2:.4f}", f"{dep2:.4f}",
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
                    "d1_alpha", "d1_beta",
                    "d1_e_p", "d1_e_a", "d1_de_p",
                    "d2_x_cmd", "d2_y_cmd", "d2_z_cmd",
                    "d2_x_act", "d2_y_act", "d2_z_act",
                    "d2_alpha", "d2_beta",
                    "d2_e_p", "d2_e_a", "d2_de_p",
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
        cmd_yaw_d1 = TARGET_YAW
        cmd_yaw_d2 = TARGET_YAW
        phase = "INIT"
        d1_data = (cmd_d1, cmd_yaw_d1, 0.5, 0.5, 0, 0, 0)
        d2_data = (cmd_d2, cmd_yaw_d2, 0.5, 0.5, 0, 0, 0)

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
                vc_start + VS_OFFSET_D1, UAV1_HOME_WORLD)
            target_d2 = self.world_to_local(
                vc_start + VS_OFFSET_D2, UAV2_HOME_WORLD)

            cmd_d1 = hover_d1 * (1.0 - t) + target_d1 * t
            cmd_d2 = hover_d2 * (1.0 - t) + target_d2 * t

        # ==== 阶段 3：模糊自适应画圆 ====
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
                    f"=== 一圈完成! 日志: {self.csv_file} ===")

            # 虚拟中心位置
            vc_x = CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta)
            vc_y = CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta)
            vc_z = TAKEOFF_HEIGHT
            vc_pos = np.array([vc_x, vc_y, vc_z])

            # 期望位置（虚拟结构法 → 本地坐标）
            desired_d1_local = self.world_to_local(
                vc_pos + VS_OFFSET_D1, UAV1_HOME_WORLD)
            desired_d2_local = self.world_to_local(
                vc_pos + VS_OFFSET_D2, UAV2_HOME_WORLD)

            # ====================================================
            # 模糊控制：根据误差自适应调节权重
            # ====================================================
            cmd_d1, cmd_yaw_d1, a1, b1, ep1, ea1, dep1 = \
                self.compute_fuzzy_cmd(1, desired_d1_local, TARGET_YAW)
            cmd_d2, cmd_yaw_d2, a2, b2, ep2, ea2, dep2 = \
                self.compute_fuzzy_cmd(2, desired_d2_local, TARGET_YAW)

            d1_data = (cmd_d1.copy(), cmd_yaw_d1, a1, b1, ep1, ea1, dep1)
            d2_data = (cmd_d2.copy(), cmd_yaw_d2, a2, b2, ep2, ea2, dep2)

            if self.tick % 50 == 0:
                progress = ((self.theta - self.theta_start)
                            / (2.0 * math.pi) * 100)
                self.get_logger().info(
                    f"圈:{progress:.1f}% | "
                    f"D1: α={a1:.2f} β={b1:.2f} ep={ep1:.2f}m | "
                    f"D2: α={a2:.2f} β={b2:.2f} ep={ep2:.2f}m")

        # === 安全 ===
        cmd_d1[2] = self.clamp_z(cmd_d1[2])
        cmd_d2[2] = self.clamp_z(cmd_d2[2])
        cmd_d1 = self.rate_limit(cmd_d1, self.last_cmd_d1, MAX_XY_STEP)
        cmd_d2 = self.rate_limit(cmd_d2, self.last_cmd_d2, MAX_XY_STEP)
        self.last_cmd_d1 = cmd_d1.copy()
        self.last_cmd_d2 = cmd_d2.copy()

        if phase == "CIRCLE" and not self.circle_complete:
            self.buffer_csv_row(d1_data, d2_data)

        self.pub_traj(1, cmd_d1, cmd_yaw_d1)
        self.pub_traj(2, cmd_d2, cmd_yaw_d2)

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
    node = FuzzyVirtualStructureCircle()
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