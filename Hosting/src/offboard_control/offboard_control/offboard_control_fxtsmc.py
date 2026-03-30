#!/usr/bin/env python3
"""
底层自定义控制器: 反步法 (外环/位置) + 固定时间滑模控制 (内环/姿态角速度)
双无人机编队画圆

外环 (Backstepping):
  输入: 期望位置、速度、加速度
  输出: 总推力 T, 期望旋转矩阵 R_des

内环 (Fixed-Time Sliding Mode Control, FxTSMC):
  滑模面: S = e_ω + a·sig(e_ω)^p + b·sig(e_ω)^q
  力矩控制律:
    Tau = ω × (I·ω) + I·ω̇_d + I·Φ⁻¹·(−k₁·sig(S)^α₁ − k₂·sig(S)^α₂ − η·sign(S))
  其中 Φ = I₃ + diag(a·p·|e_ω|^(p−1)) + diag(b·q·|e_ω|^(q−1))

控制输出通过 VehicleThrustSetpoint 和 VehicleTorqueSetpoint 发布给 PX4。
PX4 的 OffboardControlMode 必须设置 thrust_and_torque = True。
"""

import rclpy
import math
import numpy as np
import csv
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, VehicleCommand, VehicleStatus, TimesyncStatus,
    VehicleOdometry, VehicleThrustSetpoint, VehicleTorqueSetpoint,
)


# =====================================================================
#                       无人机物理参数
# =====================================================================
MASS = 2.0            # 质量 (kg)
GRAVITY = 9.81        # 重力加速度 (m/s²)
I_XX = 0.021          # 滚转轴转动惯量 (kg·m²)
I_YY = 0.021          # 俯仰轴转动惯量 (kg·m²)
I_ZZ = 0.04           # 偏航轴转动惯量 (kg·m²)
MAX_THRUST = 30.0     # 最大推力 (N)，用于归一化

# 力矩归一化参考值 (N·m)，对应 PX4 中 [-1, 1] 的满量程
# 根据 x500 模型和电机参数估算，可按需调整
MAX_TORQUE_XY = 1.0   # 滚转/俯仰最大力矩 (N·m)
MAX_TORQUE_Z = 0.5    # 偏航最大力矩 (N·m)

# 内环防奇异最小误差: 避免 |e_ω|→0 时 |e_ω|^(p-1) 发散 (NaN/Inf)
MIN_ERROR_THRESHOLD = 1e-3
# 最小推力因子: 防止推力向量为零导致除零错误 (悬停推力的 10%)
MIN_THRUST_FACTOR = 0.1

# =====================================================================
#                    FxTSMC 内环参数
# =====================================================================
# 滑模面参数 S = e_ω + a·sig(e_ω)^p + b·sig(e_ω)^q
SMC_A = 1.5           # 滑模面系数 a（> 0）
SMC_B = 1.5           # 滑模面系数 b（> 0）
SMC_P = 0.6           # 滑模面指数 p，需满足 0 < p < 1
SMC_Q = 1.4           # 滑模面指数 q，需满足 q > 1

# 固定时间趋近律参数 Ṡ = −k₁·sig(S)^α₁ − k₂·sig(S)^α₂ − η·sign(S)
SMC_K1 = 8.0          # 趋近律增益 k₁（> 0）
SMC_K2 = 8.0          # 趋近律增益 k₂（> 0）
SMC_ALPHA1 = 0.5      # 趋近律指数 α₁，需满足 0 < α₁ < 1
SMC_ALPHA2 = 1.5      # 趋近律指数 α₂，需满足 α₂ > 1
SMC_ETA = 0.5         # 切换增益 η（> 0），抑制抖振

# =====================================================================
#                    反步法外环参数
# =====================================================================
BS_KP = 2.0           # 位置误差增益
BS_KV = 3.0           # 速度误差增益
ATT_KR = 8.0          # 姿态误差 → 期望角速度 增益

# =====================================================================
#                         场景配置
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
CONTROL_RATE = 100.0  # 底层推力/力矩控制必须高频 (Hz)

VS_OFFSET_D1 = np.array([2.5, 0.0, 0.0])
VS_OFFSET_D2 = np.array([-2.5, 0.0, 0.0])

TARGET_YAW = 0.0


# =====================================================================
#                         数学辅助函数
# =====================================================================

def sig(x, alpha):
    """广义符号幂函数 (element-wise): sig(x)^α = |x|^α · sign(x)"""
    return np.abs(x) ** alpha * np.sign(x)


def quat_to_rot(q):
    """
    四元数转旋转矩阵。

    参数 q: [w, x, y, z] (PX4 VehicleOdometry 中的格式)
    返回: 3×3 旋转矩阵 R，满足 v_world = R · v_body
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x**2 + y**2)],
    ])


# =====================================================================
#                  自定义底层控制器
# =====================================================================

class FxTSMCController:
    """
    反步法 (外环) + 固定时间滑模控制 (内环) 控制器。

    外环: 由期望位置/速度通过反步法求出总推力 T 和期望旋转矩阵 R_des。
    内环: 由角速度误差设计固定时间滑模面并计算三轴力矩 Tau。
    """

    def __init__(self):
        self.I = np.array([I_XX, I_YY, I_ZZ])

    # ------------------------------------------------------------------
    # 外环: 反步法位置控制
    # ------------------------------------------------------------------
    def compute_backstepping_outer(self, pos, vel, pos_d, vel_d, acc_d,
                                   yaw_d):
        """
        反步法外环：由位置/速度误差求解期望推力 (归一化) 和期望旋转矩阵。

        坐标系: NED (+z 向下, 推力沿 -z_body 方向)

        返回:
            norm_thrust : 归一化推力标量 [-1, 0]（Z 轴, 向上为负）
            R_des       : 3×3 期望旋转矩阵
        """
        # ---- 反步法虚拟控制量 ----
        e_p = pos - pos_d
        e_v = vel - vel_d
        # 期望加速度 (反步法二阶项)
        a_des = acc_d - BS_KP * e_p - BS_KV * e_v

        # ---- 饱和处理：防止翻转和极端垂直运动 ----
        # 水平加速度限幅: 约35°倾斜角对应 ~6.8 m/s²
        a_des[:2] = np.clip(a_des[:2], -6.8, 6.8)
        # 垂直加速度限幅: 防止极端自由落体或急剧上升
        a_des[2] = np.clip(a_des[2], -5.0, 5.0)

        # ---- 推力向量 (NED) ----
        # F_des = m*(a_des - g_NED), g_NED = [0,0,g] in NED (+z down)
        g_ned = np.array([0.0, 0.0, GRAVITY])
        F_des = MASS * (a_des - g_ned)

        thrust = np.linalg.norm(F_des)
        thrust = max(thrust, MIN_THRUST_FACTOR * MASS * GRAVITY)  # 保证最小推力防除零

        # ---- 期望体轴 z (NED 下, 指向 F_des 方向) ----
        z_b_des = F_des / thrust

        # ---- 利用期望偏航构造期望旋转矩阵 ----
        x_c = np.array([math.cos(yaw_d), math.sin(yaw_d), 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        norm_y = np.linalg.norm(y_b_des)
        if norm_y < 1e-6:
            # 当 z_b_des 接近 x_c 方向时的退化处理
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des = y_b_des / norm_y
        x_b_des = np.cross(y_b_des, z_b_des)

        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        # ---- 推力归一化 [-1, 0] ----
        norm_thrust = -thrust / MAX_THRUST
        norm_thrust = max(-1.0, min(0.0, norm_thrust))

        return norm_thrust, R_des

    # ------------------------------------------------------------------
    # 内环辅助: 由姿态误差计算期望角速度
    # ------------------------------------------------------------------
    @staticmethod
    def compute_omega_d(R_current, R_des):
        """
        基于 SO(3) 姿态误差计算期望角速度 ω_d。

        e_R = vee( 0.5·(R_des^T·R - R^T·R_des) )
        ω_d = −k_R · e_R
        """
        e_R_mat = 0.5 * (R_des.T @ R_current - R_current.T @ R_des)
        # vee 映射: 从反对称矩阵提取轴向量
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
        return -ATT_KR * e_R

    # ------------------------------------------------------------------
    # 内环: 固定时间滑模控制
    # ------------------------------------------------------------------
    def compute_fxtsmc_torque(self, omega, omega_d, omega_dot_d=None):
        """
        固定时间滑模控制内环：计算三轴力矩 Tau。

        滑模面设计:
            S = e_ω + a·sig(e_ω)^p + b·sig(e_ω)^q

        固定时间趋近律:
            Ṡ = −k₁·sig(S)^α₁ − k₂·sig(S)^α₂ − η·sign(S)

        四旋翼刚体动力学:
            I·ω̇ = Tau − ω × (I·ω)
            ω̇ = I⁻¹·(Tau − ω × (I·ω))

        将 Ṡ = Φ·(ω̇ − ω̇_d) 代入趋近律，解出 Tau:
            Tau = ω × (I·ω) + I·ω̇_d + I·Φ⁻¹·(趋近律)

        其中 Φ = I₃ + diag(a·p·|e_ω|^(p−1)) + diag(b·q·|e_ω|^(q−1))

        参数:
            omega      : 当前角速度 (体坐标系, rad/s), shape (3,)
            omega_d    : 期望角速度 (rad/s), shape (3,)
            omega_dot_d: 期望角加速度 (rad/s²); 为 None 时置零

        返回:
            Tau: 三轴力矩 (N·m), shape (3,)
            S  : 滑模面值 (用于调试/日志), shape (3,)
        """
        if omega_dot_d is None:
            omega_dot_d = np.zeros(3)

        # ---- 角速度误差 ----
        e_omega = omega - omega_d

        # ---- 滑模面 S = e_ω + a·sig(e_ω)^p + b·sig(e_ω)^q ----
        S = (e_omega
             + SMC_A * sig(e_omega, SMC_P)
             + SMC_B * sig(e_omega, SMC_Q))

        # ---- Φ 矩阵对角元素 ----
        # Φᵢ = 1 + a·p·|e_ωᵢ|^(p−1) + b·q·|e_ωᵢ|^(q−1)
        # 将 |e_ω| 限制到最小值 MIN_ERROR_THRESHOLD，防止负指数项在误差接近零时发散 (NaN/Inf)
        abs_e = np.abs(e_omega)
        abs_e_safe = np.maximum(abs_e, MIN_ERROR_THRESHOLD)
        phi_diag = (1.0
                    + SMC_A * SMC_P * abs_e_safe ** (SMC_P - 1.0)
                    + SMC_B * SMC_Q * abs_e_safe ** (SMC_Q - 1.0))

        # ---- 固定时间趋近律 ----
        reaching = (-SMC_K1 * sig(S, SMC_ALPHA1)
                    - SMC_K2 * sig(S, SMC_ALPHA2)
                    - SMC_ETA * np.sign(S))

        # ---- Coriolis/陀螺力矩: ω × (I·ω) ----
        I_omega = self.I * omega
        coriolis = np.cross(omega, I_omega)

        # ---- 控制律 ----
        # Tau = ω×(I·ω) + I·ω̇_d + I·(reaching / Φ)
        Tau = coriolis + self.I * omega_dot_d + self.I * (reaching / phi_diag)

        return Tau, S

    # ------------------------------------------------------------------
    # 完整控制计算入口
    # ------------------------------------------------------------------
    def compute_control(self, current_state, desired_state):
        """
        计算归一化推力和归一化力矩。

        current_state: dict { 'pos', 'vel', 'q', 'omega' }
          pos  : 位置 (NED, m), shape (3,)
          vel  : 速度 (NED, m/s), shape (3,)
          q    : 四元数 [w,x,y,z]
          omega: 角速度 (体坐标系, rad/s), shape (3,)

        desired_state: dict { 'pos_d', 'vel_d', 'acc_d', 'yaw_d' }

        返回:
            norm_thrust : 归一化推力 [-1, 0]（Z轴）
            norm_torque : 归一化力矩 [-1, 1], shape (3,)
            debug_info  : dict (S, omega_d, Tau_raw)
        """
        pos = current_state['pos']
        vel = current_state['vel']
        q = current_state['q']
        omega = current_state['omega']

        pos_d = desired_state['pos_d']
        vel_d = desired_state['vel_d']
        acc_d = desired_state['acc_d']
        yaw_d = desired_state['yaw_d']

        # ---- 外环: 反步法 ----
        norm_thrust, R_des = self.compute_backstepping_outer(
            pos, vel, pos_d, vel_d, acc_d, yaw_d)

        # ---- 当前旋转矩阵 ----
        R_cur = quat_to_rot(q)

        # ---- 期望角速度 ----
        omega_d = self.compute_omega_d(R_cur, R_des)

        # ---- 内环: FxTSMC ----
        Tau, S = self.compute_fxtsmc_torque(omega, omega_d)

        # ---- 力矩归一化 ----
        norm_torque = np.array([
            np.clip(Tau[0] / MAX_TORQUE_XY, -1.0, 1.0),
            np.clip(Tau[1] / MAX_TORQUE_XY, -1.0, 1.0),
            np.clip(Tau[2] / MAX_TORQUE_Z,  -1.0, 1.0),
        ])

        debug_info = {'S': S, 'omega_d': omega_d, 'Tau_raw': Tau}
        return norm_thrust, norm_torque, debug_info


# =====================================================================
#                         主节点
# =====================================================================

class FxTSMCCircle(Node):
    """
    双无人机虚拟结构编队画圆节点 (底层推力/力矩控制)。

    与 offboard_control_fuzzy.py 的轨迹生成逻辑保持一致，
    控制输出改为 VehicleThrustSetpoint + VehicleTorqueSetpoint。
    """

    def __init__(self) -> None:
        super().__init__('fxtsmc_circle')

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
        self.pub_thrust_d1 = self.create_publisher(
            VehicleThrustSetpoint,
            '/px4_1/fmu/in/vehicle_thrust_setpoint', qos_pub)
        self.pub_torque_d1 = self.create_publisher(
            VehicleTorqueSetpoint,
            '/px4_1/fmu/in/vehicle_torque_setpoint', qos_pub)
        self.pub_cmd_d1 = self.create_publisher(
            VehicleCommand,
            '/px4_1/fmu/in/vehicle_command', qos_pub)

        # ---------- PX4_1 订阅者 ----------
        self.create_subscription(
            VehicleOdometry,
            '/px4_1/fmu/out/vehicle_odometry',
            lambda m: self.odom_cb(m, 1), qos_sub)
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
        self.pub_thrust_d2 = self.create_publisher(
            VehicleThrustSetpoint,
            '/px4_2/fmu/in/vehicle_thrust_setpoint', qos_pub)
        self.pub_torque_d2 = self.create_publisher(
            VehicleTorqueSetpoint,
            '/px4_2/fmu/in/vehicle_torque_setpoint', qos_pub)
        self.pub_cmd_d2 = self.create_publisher(
            VehicleCommand,
            '/px4_2/fmu/in/vehicle_command', qos_pub)

        # ---------- PX4_2 订阅者 ----------
        self.create_subscription(
            VehicleOdometry,
            '/px4_2/fmu/out/vehicle_odometry',
            lambda m: self.odom_cb(m, 2), qos_sub)
        self.create_subscription(
            VehicleStatus,
            '/px4_2/fmu/out/vehicle_status_v1',
            lambda m: self.status_cb(m, 2), qos_sub)
        self.create_subscription(
            TimesyncStatus,
            '/px4_2/fmu/out/timesync_status',
            lambda m: self.timesync_cb(m, 2), qos_sub)

        # ---------- 状态初始化 ----------
        self.state_d1 = {
            'pos': np.zeros(3),
            'vel': np.zeros(3),
            'q': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega': np.zeros(3),
        }
        self.state_d2 = {
            'pos': np.zeros(3),
            'vel': np.zeros(3),
            'q': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega': np.zeros(3),
        }
        self.status_d1 = VehicleStatus()
        self.status_d2 = VehicleStatus()

        self.px4_ts_d1 = 0
        self.px4_ts_d2 = 0
        self.ts_received_d1 = False
        self.ts_received_d2 = False

        # ---------- 控制器 ----------
        self.controller = FxTSMCController()

        # ---------- 轨迹状态 ----------
        self.theta = -math.pi / 2
        self.theta_start = self.theta
        self.circle_complete = False
        self.tick = 0
        self.takeoff_tick_start = 0
        self.all_systems_go = False

        self.last_pos_d1 = None
        self.last_pos_d2 = None

        # ---------- CSV 日志 ----------
        timestamp = datetime.now().strftime("%H%M%S")
        self.csv_file = f"log_fxtsmc_{timestamp}.csv"
        self.csv_buffer = []
        self.csv_flush_interval = 50
        self.csv_header_written = False

        # ---------- 定时器 ----------
        self.heartbeat_timer = self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_cb)
        self.control_timer = self.create_timer(
            1.0 / CONTROL_RATE, self.control_cb)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  反步法 + 固定时间滑模控制 (FxTSMC)")
        self.get_logger().info(f"  MASS={MASS}kg  I=[{I_XX},{I_YY},{I_ZZ}]")
        self.get_logger().info(
            f"  SMC: a={SMC_A} b={SMC_B} p={SMC_P} q={SMC_Q}"
            f"  k1={SMC_K1} k2={SMC_K2}")
        self.get_logger().info(f"  日志: {self.csv_file}")
        self.get_logger().info("=" * 60)

    # ==================================================================
    # 时间同步回调
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
    # 里程计回调
    # ==================================================================
    def odom_cb(self, msg, id):
        """解析 VehicleOdometry 获取全状态 (NED 坐标系)。"""
        state = self.state_d1 if id == 1 else self.state_d2
        state['pos'] = np.array([
            msg.position[0], msg.position[1], msg.position[2]])
        state['vel'] = np.array([
            msg.velocity[0], msg.velocity[1], msg.velocity[2]])
        # PX4 四元数格式: [w, x, y, z]
        state['q'] = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
        state['omega'] = np.array([
            msg.angular_velocity[0],
            msg.angular_velocity[1],
            msg.angular_velocity[2],
        ])

    # ==================================================================
    # 状态回调
    # ==================================================================
    def status_cb(self, msg, id):
        if id == 1:
            self.status_d1 = msg
        else:
            self.status_d2 = msg

    # ==================================================================
    # 心跳 (OffboardControlMode 心跳, 启用 thrust_and_torque)
    # ==================================================================
    def heartbeat_cb(self):
        if not (self.ts_received_d1 and self.ts_received_d2):
            return
        for id, pub in [(1, self.pub_offboard_d1), (2, self.pub_offboard_d2)]:
            msg = OffboardControlMode()
            msg.position = False
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False
            msg.thrust_and_torque = True  # 底层推力/力矩控制
            msg.direct_actuator = False
            msg.timestamp = self.get_ts(id)
            pub.publish(msg)

        if self.all_systems_go:
            for id in [1, 2]:
                s = self.status_d1 if id == 1 else self.status_d2
                if (s.arming_state == VehicleStatus.ARMING_STATE_ARMED
                        and s.nav_state
                        != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                    self.engage_offboard(id)

    # ==================================================================
    # 辅助工具
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

    def rate_limit(self, pos_new, pos_last, max_step):
        if pos_last is None:
            return pos_new.copy()
        diff = pos_new - pos_last
        dist = np.linalg.norm(diff[:2])
        if dist > max_step:
            scale = max_step / dist
            limited = pos_last.copy()
            limited[0] += diff[0] * scale
            limited[1] += diff[1] * scale
            limited[2] = pos_new[2]
            return limited
        return pos_new.copy()

    # ==================================================================
    # 控制主循环
    # ==================================================================
    def control_cb(self):
        if not (self.ts_received_d1 and self.ts_received_d2):
            return

        self.tick += 1

        # ---- 起飞目标 ----
        hover_pos_d1 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        hover_pos_d2 = np.array([0.0, 0.0, TAKEOFF_HEIGHT])

        is_armed_d1 = (self.status_d1.arming_state
                       == VehicleStatus.ARMING_STATE_ARMED)
        is_offboard_d1 = (self.status_d1.nav_state
                          == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        is_armed_d2 = (self.status_d2.arming_state
                       == VehicleStatus.ARMING_STATE_ARMED)
        is_offboard_d2 = (self.status_d2.nav_state
                          == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

        if (is_armed_d1 and is_offboard_d1
                and is_armed_d2 and is_offboard_d2):
            if not self.all_systems_go:
                self.get_logger().info(">>> 全部就绪，起飞中... <<<")
                self.takeoff_tick_start = self.tick
            self.all_systems_go = True

        elapsed = (self.tick - self.takeoff_tick_start
                   if self.all_systems_go else 0)

        # ---- 期望设定 ----
        desired_d1 = {
            'pos_d': hover_pos_d1,
            'vel_d': np.zeros(3),
            'acc_d': np.zeros(3),
            'yaw_d': TARGET_YAW,
        }
        desired_d2 = {
            'pos_d': hover_pos_d2,
            'vel_d': np.zeros(3),
            'acc_d': np.zeros(3),
            'yaw_d': TARGET_YAW,
        }
        phase = "INIT"

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

            vc_start = np.array([
                CIRCLE_CENTER_X + CIRCLE_RADIUS * math.cos(self.theta_start),
                CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.sin(self.theta_start),
                TAKEOFF_HEIGHT,
            ])
            target_d1 = self.world_to_local(
                vc_start + VS_OFFSET_D1, UAV1_HOME_WORLD)
            target_d2 = self.world_to_local(
                vc_start + VS_OFFSET_D2, UAV2_HOME_WORLD)

            pos_d1 = hover_pos_d1 * (1.0 - t) + target_d1 * t
            pos_d2 = hover_pos_d2 * (1.0 - t) + target_d2 * t

            desired_d1['pos_d'] = pos_d1
            desired_d2['pos_d'] = pos_d2

        # ==== 阶段 3：FxTSMC 画圆 ====
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

            # 期望速度 (对时间微分)
            vc_vx = -CIRCLE_RADIUS * spd * math.sin(self.theta)
            vc_vy = CIRCLE_RADIUS * spd * math.cos(self.theta)
            vc_vel = np.array([vc_vx, vc_vy, 0.0])

            # 期望加速度 (对速度微分)
            vc_ax = -CIRCLE_RADIUS * spd**2 * math.cos(self.theta)
            vc_ay = -CIRCLE_RADIUS * spd**2 * math.sin(self.theta)
            vc_acc = np.array([vc_ax, vc_ay, 0.0])

            desired_d1['pos_d'] = self.world_to_local(
                vc_pos + VS_OFFSET_D1, UAV1_HOME_WORLD)
            desired_d1['vel_d'] = vc_vel
            desired_d1['acc_d'] = vc_acc

            desired_d2['pos_d'] = self.world_to_local(
                vc_pos + VS_OFFSET_D2, UAV2_HOME_WORLD)
            desired_d2['vel_d'] = vc_vel
            desired_d2['acc_d'] = vc_acc

        # ---- 速率限制期望位置 (安全保障) ----
        desired_d1['pos_d'] = self.rate_limit(
            desired_d1['pos_d'], self.last_pos_d1, MAX_XY_STEP)
        desired_d2['pos_d'] = self.rate_limit(
            desired_d2['pos_d'], self.last_pos_d2, MAX_XY_STEP)
        desired_d1['pos_d'][2] = self.clamp_z(desired_d1['pos_d'][2])
        desired_d2['pos_d'][2] = self.clamp_z(desired_d2['pos_d'][2])
        self.last_pos_d1 = desired_d1['pos_d'].copy()
        self.last_pos_d2 = desired_d2['pos_d'].copy()

        # ---- 计算推力和力矩 ----
        thrust_d1, torque_d1, dbg_d1 = self.controller.compute_control(
            self.state_d1, desired_d1)
        thrust_d2, torque_d2, dbg_d2 = self.controller.compute_control(
            self.state_d2, desired_d2)

        # ---- 发布 ----
        self.publish_thrust_torque(1, thrust_d1, torque_d1)
        self.publish_thrust_torque(2, thrust_d2, torque_d2)

        # ---- 日志 ----
        if phase == "CIRCLE" and not self.circle_complete:
            self.buffer_csv_row(
                self.state_d1, desired_d1, thrust_d1, torque_d1, dbg_d1,
                self.state_d2, desired_d2, thrust_d2, torque_d2, dbg_d2)

        if self.tick % 100 == 0:
            progress = ((self.theta - self.theta_start)
                        / (2.0 * math.pi) * 100) if self.all_systems_go else 0
            S1 = dbg_d1['S']
            S2 = dbg_d2['S']
            self.get_logger().info(
                f"[{phase}] 圈:{progress:.1f}%  "
                f"D1 S={np.linalg.norm(S1):.3f} τ={torque_d1}  "
                f"D2 S={np.linalg.norm(S2):.3f} τ={torque_d2}")

    # ==================================================================
    # 发布推力 / 力矩
    # ==================================================================
    def publish_thrust_torque(self, id, thrust, torque):
        ts = self.get_ts(id)

        thrust_msg = VehicleThrustSetpoint()
        thrust_msg.timestamp = ts
        # NED 坐标系: 推力沿体轴 -Z 方向 (Z 分量为负表示向上)
        thrust_msg.xyz = [0.0, 0.0, float(thrust)]

        torque_msg = VehicleTorqueSetpoint()
        torque_msg.timestamp = ts
        torque_msg.xyz = [float(torque[0]), float(torque[1]), float(torque[2])]

        if id == 1:
            self.pub_thrust_d1.publish(thrust_msg)
            self.pub_torque_d1.publish(torque_msg)
        else:
            self.pub_thrust_d2.publish(thrust_msg)
            self.pub_torque_d2.publish(torque_msg)

    # ==================================================================
    # 指令发送
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
        msg.param1 = kwargs.get('param1', 0.0)
        msg.param2 = kwargs.get('param2', 0.0)
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

    # ==================================================================
    # CSV 日志
    # ==================================================================
    def buffer_csv_row(self, st1, des1, thr1, tau1, dbg1,
                       st2, des2, thr2, tau2, dbg2):
        if self.circle_complete:
            return
        S1 = dbg1['S']
        S2 = dbg2['S']
        od1 = dbg1['omega_d']
        od2 = dbg2['omega_d']
        self.csv_buffer.append([
            self.tick, f"{self.theta:.4f}",
            # UAV1
            f"{st1['pos'][0]:.3f}", f"{st1['pos'][1]:.3f}",
            f"{st1['pos'][2]:.3f}",
            f"{des1['pos_d'][0]:.3f}", f"{des1['pos_d'][1]:.3f}",
            f"{des1['pos_d'][2]:.3f}",
            f"{thr1:.4f}",
            f"{tau1[0]:.4f}", f"{tau1[1]:.4f}", f"{tau1[2]:.4f}",
            f"{S1[0]:.4f}", f"{S1[1]:.4f}", f"{S1[2]:.4f}",
            f"{od1[0]:.4f}", f"{od1[1]:.4f}", f"{od1[2]:.4f}",
            # UAV2
            f"{st2['pos'][0]:.3f}", f"{st2['pos'][1]:.3f}",
            f"{st2['pos'][2]:.3f}",
            f"{des2['pos_d'][0]:.3f}", f"{des2['pos_d'][1]:.3f}",
            f"{des2['pos_d'][2]:.3f}",
            f"{thr2:.4f}",
            f"{tau2[0]:.4f}", f"{tau2[1]:.4f}", f"{tau2[2]:.4f}",
            f"{S2[0]:.4f}", f"{S2[1]:.4f}", f"{S2[2]:.4f}",
            f"{od2[0]:.4f}", f"{od2[1]:.4f}", f"{od2[2]:.4f}",
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
                    "d1_x_act", "d1_y_act", "d1_z_act",
                    "d1_x_cmd", "d1_y_cmd", "d1_z_cmd",
                    "d1_thrust",
                    "d1_tau_x", "d1_tau_y", "d1_tau_z",
                    "d1_S_x", "d1_S_y", "d1_S_z",
                    "d1_wd_x", "d1_wd_y", "d1_wd_z",
                    "d2_x_act", "d2_y_act", "d2_z_act",
                    "d2_x_cmd", "d2_y_cmd", "d2_z_cmd",
                    "d2_thrust",
                    "d2_tau_x", "d2_tau_y", "d2_tau_z",
                    "d2_S_x", "d2_S_y", "d2_S_z",
                    "d2_wd_x", "d2_wd_y", "d2_wd_z",
                ])
                self.csv_header_written = True
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()


# =====================================================================
#                              入口
# =====================================================================

def main(args=None):
    rclpy.init(args=args)
    node = FxTSMCCircle()
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
