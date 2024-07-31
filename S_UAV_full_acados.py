from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat, horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from fancy_plots import fancy_plots_2, fancy_plots_1
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import math
import scipy.io

from geometry_msgs.msg import TwistStamped

#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

# Global variables Odometry Drone Condicion inicial
x_real = 3
y_real = 3
z_real = 2
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0

# Angular velocities
qx_real = 0
qy_real = 0.0
qz_real = 0
qw_real = 1
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0

hdp_vision = [0,0,0,0,0,0.0]
axes = [0,0,0,0,0,0]

def odometry_call_back(odom_msg):
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    # Read desired linear velocities from node
    x_real = odom_msg.pose.pose.position.x 
    y_real = odom_msg.pose.pose.position.y
    z_real = odom_msg.pose.pose.position.z
    vx_real = odom_msg.twist.twist.linear.x
    vy_real = odom_msg.twist.twist.linear.y
    vz_real = odom_msg.twist.twist.linear.z


    qx_real = odom_msg.pose.pose.orientation.x
    qy_real = odom_msg.pose.pose.orientation.y
    qz_real = odom_msg.pose.pose.orientation.z
    qw_real = odom_msg.pose.pose.orientation.w

    wx_real = odom_msg.twist.twist.angular.x
    wy_real = odom_msg.twist.twist.angular.y
    wz_real = odom_msg.twist.twist.angular.z
    return None


def Rot_zyx(x):
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Rot Matrix axis X
    RotX = MX.zeros(3, 3)
    RotX[0, 0] = 1.0
    RotX[1, 1] = cos(phi)
    RotX[1, 2] = -sin(phi)
    RotX[2, 1] = sin(phi)
    RotX[2, 2] = cos(phi)

    # Rot Matrix axis Y
    RotY = MX.zeros(3, 3)
    RotY[0, 0] = cos(theta)
    RotY[0, 2] = sin(theta)
    RotY[1, 1] = 1.0
    RotY[2, 0] = -sin(theta)
    RotY[2, 2] = cos(theta)

    RotZ = MX.zeros(3, 3)
    RotZ[0, 0] = cos(psi)
    RotZ[0, 1] = -sin(psi)
    RotZ[1, 0] = sin(psi)
    RotZ[1, 1] = cos(psi)
    RotZ[2, 2] = 1.0

    R = RotZ@RotY@RotX
    return R
def M_matrix_bar(chi, x):

    # Split Parameters
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Constants of the system
    m1 = chi[0];
    Ixx = chi[1];
    Iyy = chi[2];
    Izz = chi[3];

    # Mass Matrix
    M = MX.zeros(6, 6)
    M[0, 0] = m1
    M[1, 1] = m1
    M[2, 2] = m1
    M[3, 3] = Ixx
    M[3, 5] = -Ixx*sin(theta)
    M[4, 4] = Izz + Iyy*cos(phi)**2 - Izz*cos(phi)**2
    M[4, 5] = Iyy*cos(phi)*cos(theta)*sin(phi) - Izz*cos(phi)*cos(theta)*sin(phi)
    M[5, 3] = -Ixx*sin(theta)
    M[5, 4] = Iyy*cos(phi)*cos(theta)*sin(phi) - Izz*cos(phi)*cos(theta)*sin(phi)
    M[5, 5] = Ixx - Ixx*cos(theta)**2 + Iyy*cos(theta)**2 - Iyy*cos(phi)**2*cos(theta)**2 + Izz*cos(phi)**2*cos(theta)**2
    return M

def C_matrix_bar(chi, x):
    # Split Parameters system
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    phi_p = x[9, 0]
    theta_p = x[10, 0]
    psi_p = x[11, 0]

    m1 = chi[0];
    Ixx = chi[1];
    Iyy = chi[2];
    Izz = chi[3];

    C = MX.zeros(6, 6)
    C[3, 4] = (Iyy*psi_p*cos(theta))/2 - (Ixx*psi_p*cos(theta))/2 - (Izz*psi_p*cos(theta))/2 - Iyy*psi_p*cos(phi)**2*cos(theta) + Izz*psi_p*cos(phi)**2*cos(theta) + Iyy*theta_p*cos(phi)*sin(phi) - Izz*theta_p*cos(phi)*sin(phi)
    C[3, 5] = (Iyy*theta_p*cos(theta))/2 - (Ixx*theta_p*cos(theta))/2 - (Izz*theta_p*cos(theta))/2 - Iyy*theta_p*cos(phi)**2*cos(theta) + Izz*theta_p*cos(phi)**2*cos(theta) - Iyy*psi_p*cos(phi)*cos(theta)**2*sin(phi) + Izz*psi_p*cos(phi)*cos(theta)**2*sin(phi)
    C[4, 3] = (Ixx*psi_p*cos(theta))/2 - (Iyy*psi_p*cos(theta))/2 + (Izz*psi_p*cos(theta))/2 + Iyy*psi_p*cos(phi)**2*cos(theta) - Izz*psi_p*cos(phi)**2*cos(theta) - Iyy*theta_p*cos(phi)*sin(phi) + Izz*theta_p*cos(phi)*sin(phi)
    C[4, 4] = Izz*phi_p*cos(phi)*sin(phi) - Iyy*phi_p*cos(phi)*sin(phi)
    C[4, 5] = (Ixx*phi_p*cos(theta))/2 - (Iyy*phi_p*cos(theta))/2 + (Izz*phi_p*cos(theta))/2 + Iyy*phi_p*cos(phi)**2*cos(theta) - Izz*phi_p*cos(phi)**2*cos(theta) - Ixx*psi_p*cos(theta)*sin(theta) + Iyy*psi_p*cos(theta)*sin(theta) - Iyy*psi_p*cos(phi)**2*cos(theta)*sin(theta) + Izz*psi_p*cos(phi)**2*cos(theta)*sin(theta)
    C[5, 3] = (Izz*theta_p*cos(theta))/2 - (Iyy*theta_p*cos(theta))/2 - (Ixx*theta_p*cos(theta))/2 + Iyy*theta_p*cos(phi)**2*cos(theta) - Izz*theta_p*cos(phi)**2*cos(theta) + Iyy*psi_p*cos(phi)*cos(theta)**2*sin(phi) - Izz*psi_p*cos(phi)*cos(theta)**2*sin(phi)
    C[5, 4] = (Izz*phi_p*cos(theta))/2 - (Iyy*phi_p*cos(theta))/2 - (Ixx*phi_p*cos(theta))/2 + Iyy*phi_p*cos(phi)**2*cos(theta) - Izz*phi_p*cos(phi)**2*cos(theta) + Ixx*psi_p*cos(theta)*sin(theta) - Iyy*psi_p*cos(theta)*sin(theta) + Iyy*psi_p*cos(phi)**2*cos(theta)*sin(theta) - Izz*psi_p*cos(phi)**2*cos(theta)*sin(theta) - Iyy*theta_p*cos(phi)*sin(phi)*sin(theta) + Izz*theta_p*cos(phi)*sin(phi)*sin(theta)
    C[5, 5] = Ixx*theta_p*cos(theta)*sin(theta) - Iyy*theta_p*cos(theta)*sin(theta) + Iyy*phi_p*cos(phi)*cos(theta)**2*sin(phi) - Izz*phi_p*cos(phi)*cos(theta)**2*sin(phi) + Iyy*theta_p*cos(phi)**2*cos(theta)*sin(theta) - Izz*theta_p*cos(phi)**2*cos(theta)*sin(theta)
    return C

def G_matrix_bar(chi, x):
    g = 9.81

    # Split Parameters of the system
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Constan values of the system
    m1 = chi[0];
    G = MX.zeros(6, 1)
    G[2, 0] = g*m1
    return G

def S_fuction(chi):
    S = MX.zeros(6, 6)
    S[2, 2] = chi[4]
    S[3, 3] = chi[5]
    S[4, 4] = chi[6]
    S[5, 5] = chi[7]
    return S
def Q_fuction(chi):
    Q = MX.zeros(6, 6)
    Q[3, 3] = chi[8]
    Q[4, 4] = chi[9]
    
    return Q

def E_fuction(chi):
    E = MX.zeros(6, 6)
    E[2,2] = chi[10]
    E[3,3] = chi[11]
    E[4,4] = chi[12]
    E[5,5] = chi[13]
    return E

def T_fuction(chi):
    E = MX.zeros(6,6)
    E[2,2] = chi[14]
    E[5,5] = chi[15]
    return E

def B_fuction(chi):
    m1 = chi[0];
    g = 9.81
    B = MX.zeros(6,1)
    B[2, 0] = m1*g
    return B

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system
    g = 9.81
    
    # Carga el archivo .mat
    mat = scipy.io.loadmat('chi_uav_compact_full_model.mat')

    # Accede a la variable 'values_final'
    values_final = mat['values_final']

    # Convierte la matriz en un vector de nx1
    chi = np.reshape(values_final, (-1, 1))

    #chi = [5.3e-07, 9.4e-07, 3.8e-07, 6.4e-07, 8.05e-06, 0.0002241, 0.0001181, 2.83e-05, 0.0002204, 0.0001103, 7.9e-05, 6.06e-05, 2.36e-05, 1.79e-05, 3.65e-05, 4.4e-06]
    
    m = chi[0]

    # set up states & controls
    # Position
    x1 = MX.sym('x1')
    y1 = MX.sym('y1')
    z1 = MX.sym('z1')
    # Orientation
    phi = MX.sym('phi')
    theta = MX.sym('theta')
    psi = MX.sym('psi')

    # Velocity Linear and Angular
    dx1 = MX.sym('dx1')
    dy1 = MX.sym('dy1')
    dz1 = MX.sym('dz1')
    dphi = MX.sym('dphi')
    dtheta = MX.sym('dtheta')
    dpsi = MX.sym('dpsi')

    # General vector of the states
    x = vertcat(x1, y1, z1, phi, theta, psi, dx1, dy1, dz1, dphi, dtheta, dpsi)

    # Action variables
    zp_ref = MX.sym('F')
    phi_ref = MX.sym('ux')
    theta_ref = MX.sym('uy')
    psi_p_ref = MX.sym('uz')

    # General Vector Action variables
    u = vertcat(zp_ref, phi_ref, theta_ref, psi_p_ref)

    # Variables to explicit function
    x1_dot = MX.sym('x1_dot')
    y1_dot = MX.sym('y1_dot')
    z1_dot = MX.sym('z1_dot')
    phi_dot = MX.sym('phi_dot')
    theta_dot = MX.sym('theta_dot')
    psi_dot = MX.sym('psi_dot')
    dx1_dot = MX.sym('dx1_dot')
    dy1_dot = MX.sym('dy1_dot')
    dz1_dot = MX.sym('dz1_dot')
    dphi_dot = MX.sym('dphi_dot')
    dtheta_dot = MX.sym('dtheta_dot')
    dpsi_dot = MX.sym('dpsi_dot')

    # general vector X dot for implicit function
    xdot = vertcat(x1_dot, y1_dot, z1_dot,  phi_dot, theta_dot, psi_dot, dx1_dot, dy1_dot, dz1_dot, dphi_dot, dtheta_dot, dpsi_dot)

    # Ref system as a external value

    x1_d = MX.sym('x1_d')
    y1_d = MX.sym('y1_d')
    z1_d = MX.sym('z1_d')

    phi_d = MX.sym('phi_d')
    theta_d = MX.sym('theta_d')
    psi_d = MX.sym('psi_d')

    dx1_d = MX.sym('dx1_d')
    dy1_d = MX.sym('dy1_d')
    dz1_d = MX.sym('dz1_d')
    dphi_d = MX.sym('dphi_d')
    dtheta_d = MX.sym('dtheta_d')
    dpsi_d = MX.sym('dpsi_d')

    F_ref_d= MX.sym('ul_ref_d')
    Taux_ref_d= MX.sym('um_ref_d')
    Tauy_ref_d = MX.sym('un_ref_d')
    Tauz_ref_d = MX.sym('w_ref_d')
    
    p = vertcat(x1_d, y1_d, z1_d, phi_d, theta_d, psi_d, dx1_d, dy1_d, dz1_d, dphi_d, dtheta_d, dpsi_d, F_ref_d,Taux_ref_d, Tauy_ref_d, Tauz_ref_d)

    # Rotational Matrix
    R = Rot_zyx(x)
    M_bar = M_matrix_bar(chi, x)
    C_bar = C_matrix_bar(chi, x)
    G_bar = G_matrix_bar(chi, x)
    S = S_fuction(chi)
    Q = Q_fuction(chi)
    E = E_fuction(chi)
    T = T_fuction(chi)
    B = B_fuction(chi)


    R_bar = vertcat(horzcat(R, MX.zeros(3,3)), horzcat(MX.zeros(3,3), MX.eye(3)))


    # Auxiliar Matrices 


    # Aux Control
    u_aux = vertcat(0, 0, zp_ref, phi_ref, theta_ref, psi_p_ref)
    q = x[0:6, 0]
    q_p = x[6:12, 0]

    #Aux = S@u_aux-Q@x[0:6, 0]-E@x[6:12, 0]+B
    Aux = S@u_aux-Q@q-E@q_p+B


    # Aux inverse Matrix
    M_a_r = M_bar + R_bar@T
    inv_M = inv(M_a_r)

    x_pp = inv_M@(R_bar@Aux-C_bar@x[6:12, 0]-G_bar);

    f_expl = MX.zeros(12, 1)
    f_expl[0:6, 0] = x[6:12, 0]
    f_expl[6:12, 0] = x_pp

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((12,))
    return aux_x


def f_yaw():
    value = axes[2]
    return value

def RK4_yaw(x, ts, f_yaw):
    k1 = f_yaw()
    k2 = f_yaw()
    k3 = f_yaw()
    k4 = f_yaw()
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4) 
    aux_x = np.array(x[0]).reshape((1,))
    return aux_x

def Angulo(ErrAng):
    # 1) ARGUMENTOS DE ENTRADA
    # a) ErrAng ----> ángulo en radianes

    # 2) ARGUMENTOS DE SALIDA
    # a) ErrAng ----> ángulo de entrada limitado entre [0 : pi] y [-pi : 0]

    # Limitar el ángulo entre [0 : pi]
    if ErrAng >= math.pi:
        while ErrAng >= math.pi:
            ErrAng = ErrAng - 2 * math.pi
        return ErrAng

    # Limitar el ángulo entre [-pi : 0]
    if ErrAng <= -math.pi:
        while ErrAng <= -math.pi:
            ErrAng = ErrAng + 2 * math.pi
        return ErrAng

    return ErrAng



def visual_callback(msg):

    global hdp_vision 

    vx_visual = msg.twist.linear.x
    vy_visual = msg.twist.linear.y
    vz_visual = msg.twist.linear.z
    wx_visual = msg.twist.angular.x
    wy_visual = msg.twist.angular.y
    wz_visual = msg.twist.angular.z

    hdp_vision = [vx_visual, vy_visual, vz_visual, wx_visual, wy_visual, wz_visual]
    

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_p_min, psi_p_max) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_model()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(6, 6)
    Q_mat[0, 0] = 1
    Q_mat[1, 1] = 1
    Q_mat[2, 2] = 1
    Q_mat[3, 3] = 0
    Q_mat[4, 4] = 0
    Q_mat[5, 5] = 1
    
    # set cost
    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1.5*(1/0.3)
    R_mat[1, 1] = 1.5*(1/0.3)
    R_mat[2, 2] = 1.5*(1/0.3)
    R_mat[3, 3] = 3*(1/2)

    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:6] - model.x[0:6]
    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose

    # set constraints
    ocp.constraints.lbu = np.array([zp_min, phi_min, theta_min, psi_p_min])
    ocp.constraints.ubu = np.array([zp_max, phi_max, theta_max, psi_p_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = x0

    # Restricciones de z

    zmin=1.5
    zmax=50
    #ocp.constraints.lbx = np.array([zmin])
    #ocp.constraints.ubx = np.array([zmax])
    #ocp.constraints.idxbx = np.array([2])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def Euler_p(omega, euler):
    W = np.array([[1, np.sin(euler[0])*np.tan(euler[1]), np.cos(euler[0])*np.tan(euler[1])],
                  [0, np.cos(euler[0]), np.sin(euler[0])],
                  [0, np.sin(euler[0])/np.cos(euler[1]), np.cos(euler[0])/np.cos(euler[1])]])

    euler_p = np.dot(W, omega)
    return euler_p




def send_velocity_control(u, vel_pub, vel_msg):
    # Split  control values
    F = u[0]
    tx = u[1]
    ty = u[2]
    tz = u[3]

    # velocity message
    vel_msg.twist.linear.x = 0.0
    vel_msg.twist.linear.y = 0.0
    vel_msg.twist.linear.z = F

    vel_msg.twist.angular.x = tx
    vel_msg.twist.angular.y = ty
    vel_msg.twist.angular.z = tz

    # Publish control values
    vel_pub.publish(vel_msg)
    return None


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

def send_state_to_topic(state_vector):
    publisher = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    
    # Create an Odometry message
    odometry_msg = Odometry()

    quaternion = euler_to_quaternion(state_vector[3], state_vector[4], state_vector[5])

    odometry_msg.header.frame_id = "odo"
    odometry_msg.header.stamp = rospy.Time.now()
    odometry_msg.pose.pose.position.x = state_vector[0]
    odometry_msg.pose.pose.position.y = state_vector[1]
    odometry_msg.pose.pose.position.z = state_vector[2]
    odometry_msg.pose.pose.orientation.x = quaternion[1]
    odometry_msg.pose.pose.orientation.y = quaternion[2]
    odometry_msg.pose.pose.orientation.z = quaternion[3]
    odometry_msg.pose.pose.orientation.w = quaternion[0]
    odometry_msg.twist.twist.linear.x = state_vector[6]
    odometry_msg.twist.twist.linear.y = state_vector[7]
    odometry_msg.twist.twist.linear.z = state_vector[8]
    odometry_msg.twist.twist.angular.x = state_vector[9]
    odometry_msg.twist.twist.angular.y = state_vector[10]
    odometry_msg.twist.twist.angular.z = state_vector[11]

    
    # Publish the message
    publisher.publish(odometry_msg)


def rc_callback(data):
    # Extraer los datos individuales del mensaje
    global axes
    axes_aux = data.axes
    psi = -np.pi / 2

    R = np.array([[np.cos(psi), -np.sin(psi), 0, 0, 0, 0],
                [np.sin(psi), np.cos(psi), 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
    axes = R@axes_aux


def get_odometry_full():

    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real

    quaternion = [qx_real, qy_real, qz_real, qw_real ]
    r_quat = R.from_quat(quaternion)
    q2e =  r_quat.as_euler('zyx', degrees = False)
    phi = q2e[2]
    theta = q2e[1]
    psi = q2e[0]

    omega = [wx_real, wy_real, wz_real]
    euler = [phi, theta, psi]
    euler_p = Euler_p(omega,euler)

    x_state = [x_real,y_real,z_real,phi,theta,psi,vx_real,vy_real,vz_real, euler_p[0],euler_p[1],euler_p[2]]

    return x_state

def main(vel_pub, vel_msg):
    # Initial Values System
    # Simulation Time
    t_final = 60*10
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 30
    t_prediction = N_horizont/frec

    

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)


    # Vector Initial conditions
    x = np.zeros((12, t.shape[0]+1-N_prediction), dtype = np.double)
    xref = np.zeros((16, t.shape[0]), dtype = np.double)
    x_sim = np.zeros((12, t.shape[0]+1-N_prediction), dtype = np.double)


    aux_yaw_d = np.zeros((1, t.shape[0]+1-N_prediction), dtype = np.double)

    # Read Values Odometry Drone

    #TAREA DESEADA
    value = 18
    xd = lambda t: 4 * np.sin(value*0.04*t) + 3
    yd = lambda t: 4 * np.sin(value*0.08*t)
    zd = lambda t: 2 * np.sin(value*0.08*t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value*0.04*t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value*0.08*t)
    zdp = lambda t: 2 * value * 0.08 * np.cos(value*0.08*t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    # Reference Signal of the system
    xref = np.zeros((16, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd 
    xref[1,:] = hyd
    xref[2,:] = hzd  
    xref[3,:] = 0
    xref[4,:] = 0
    xref[5,:] = 0*psid  
    xref[6,:] = 0 
    xref[7,:] = 0 
   

    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 5
    phi_max = 0.5
    theta_max = 0.5
    psi_p_ref_max = 2
    

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_p_ref_min = -psi_p_ref_max

    # Create Optimal problem
    model, f = f_system_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_p_ref_min, psi_p_ref_max)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    
    
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    # Simulation System
    # Encabezados de los estados
    headers = ["hx", "hy", "hz", "phi", "theta", "psi", "hx_p", "hy_p", "hz_p", "phi_p", "theta_p", "psi_p"]

    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate
    

     # Read Real data
    for ii in range(0,10):
        x[:, 0] = get_odometry_full()
        print("Loading...")
        rate.sleep()

    print("Ready!!!")
       
    aux_yaw_d[:,0] = x[5,0]


    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

         # Reference Signal of the system
        



        
        
        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # SET REFERENCES
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)

        

        # Get Computational Time
        status = acados_ocp_solver.solve()

        if status != 0:
            print("Falla del sistema")
            send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
            break

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        #u_control[:, k] = [0.0,-0.01,0.01,0]
                
        # Send Control values
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        #Imprime El vector de estados
        
        state_vector = np.ravel(x[:, k])
        max_header_length = max(len(header) for header in headers)
        for header, value in zip(headers, state_vector):
            formatted_header = header.ljust(max_header_length)
            print(f"{formatted_header}: {value:.2f}")
        print()


        
        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            
            send_state_to_topic(x[:, k+1])
        else:
            print("Opción no válida")

          
        #aux_yaw_d[:,k+1] = (RK4_yaw(aux_yaw_d[:,k], t_s, f_yaw))

 

        
        delta_t[:, k] = toc_solver
        rate.sleep() 
        toc = time.time() - tic 
        #print(toc)
    
    
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)



    # fig1, ax11 = fancy_plots_1()
    # states_x, = ax11.plot(t[0:x.shape[1]], x[9,:],
    #                 color='#BB5651', lw=1, ls="-")
    # states_y, = ax11.plot(t[0:x.shape[1]], x[10,:],
    #                 color='#69BB51', lw=1, ls="-")
    # states_z, = ax11.plot(t[0:x.shape[1]], x[11,:],
    #                 color='#5189BB', lw=1, ls="-")
    # states_xd, = ax11.plot(t[0:x.shape[1]], xref[9,0:x.shape[1]],
    #                 color='#BB5651', lw=2, ls="--")
    # states_yd, = ax11.plot(t[0:x.shape[1]], xref[10,0:x.shape[1]],
    #                 color='#69BB51', lw=2, ls="--")
    # states_zd, = ax11.plot(t[0:x.shape[1]], xref[11,0:x.shape[1]],
    #                 color='#5189BB', lw=2, ls="--")

    # ax11.set_ylabel(r"$[states]$", rotation='vertical')
    # ax11.set_xlabel(r"$[t]$", labelpad=5)
    # ax11.legend([states_x, states_y, states_z, states_xd, states_yd, states_zd],
    #         [r'$p$', r'$q$', r'$r$', r'$p_d$', r'$q_d$', r'$r_d$'],
    #         loc="best",
    #         frameon=True, fancybox=True, shadow=False, ncol=2,
    #         borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
    #         borderaxespad=0.3, columnspacing=2)
    # ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # fig1.savefig("states_xyz.eps")
    # fig1.savefig("states_xyz.png")
    # fig1
    # #plt.show()

    # fig2, ax12 = fancy_plots_1()
    # states_phi, = ax12.plot(t[0:x.shape[1]], x[3,:],
    #                 color='#BB5651', lw=2, ls="-")
    # states_theta, = ax12.plot(t[0:x.shape[1]], x[4,:],
    #                 color='#69BB51', lw=2, ls="-")
    # states_psi, = ax12.plot(t[0:x.shape[1]], x[5,:],
    #                 color='#5189BB', lw=2, ls="-")

    # ax12.set_ylabel(r"$[states]$", rotation='vertical')
    # ax12.set_xlabel(r"$[t]$", labelpad=5)
    # ax12.legend([states_phi, states_theta, states_psi],
    #         [r'$\phi$', r'$\theta$', r'$\psi$'],
    #         loc="best",
    #         frameon=True, fancybox=True, shadow=False, ncol=2,
    #         borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
    #         borderaxespad=0.3, columnspacing=2)
    # ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # fig2.savefig("states_angles.eps")
    # fig2.savefig("states_angles.png")
    # fig2
    # #plt.show()

    # fig3, ax13 = fancy_plots_1()
    # ## Axis definition necesary to fancy plots
    # ax13.set_xlim((t[0], t[-1]))

    # time_1, = ax13.plot(t[0:delta_t.shape[1]],delta_t[0,:],
    #                 color='#00429d', lw=2, ls="-")
    # tsam1, = ax13.plot(t[0:t_sample.shape[1]],t_sample[0,:],
    #                 color='#9e4941', lw=2, ls="-.")

    # ax13.set_ylabel(r"$[s]$", rotation='vertical')
    # ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    # ax13.legend([time_1,tsam1],
    #         [r'$t_{compute}$',r'$t_{sample}$'],
    #         loc="best",
    #         frameon=True, fancybox=True, shadow=False, ncol=2,
    #         borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
    #         borderaxespad=0.3, columnspacing=2)
    # ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # fig3.savefig("time.eps")
    # fig3.savefig("time.png")
    # fig3
    # #plt.show()

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        odometry_topic = "/dji_sdk/odometry"
        velocity_subscriber = rospy.Subscriber(odometry_topic, Odometry, odometry_call_back)
        vision_sub = rospy.Subscriber("/dji_sdk/visual_servoing/vel/drone_world", TwistStamped, visual_callback, queue_size=10)
        RC_sub = rospy.Subscriber("/dji_sdk/rc", Joy, rc_callback, queue_size=10)
        
        velocity_topic = "/m100/velocityControl"
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher(velocity_topic, TwistStamped, queue_size=10)

        


        main(velocity_publisher, velocity_message)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass