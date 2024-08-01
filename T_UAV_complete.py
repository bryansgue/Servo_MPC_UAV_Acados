from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat, horzcat, vertsplit
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import norm_2
from casadi import cross
from casadi import if_else
from casadi import atan2
from fancy_plots import fancy_plots_2, fancy_plots_1
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import math
import scipy.io
from std_msgs.msg import Float64MultiArray
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



def QuatToRot(quat):
    # Quaternion to Rotational Matrix
    q = quat # Convierte la lista de cuaterniones en un objeto MX
    
    # Calcula la norma 2 del cuaternión
    q_norm = norm_2(q)
    
    # Normaliza el cuaternión dividiendo por su norma
    q_normalized = q / q_norm

    q_hat = MX.zeros(3, 3)

    q_hat[0, 1] = -q_normalized[3]
    q_hat[0, 2] = q_normalized[2]
    q_hat[1, 2] = -q_normalized[1]
    q_hat[1, 0] = q_normalized[3]
    q_hat[2, 0] = -q_normalized[2]
    q_hat[2, 1] = q_normalized[1]

    Rot = MX.eye(3) + 2 * q_hat @ q_hat + 2 * q_normalized[0] * q_hat

    return Rot

def quaternion_multiply(q1, q2):
    # Descomponer los cuaterniones en componentes
    w0, x0, y0, z0 = vertsplit(q1)
    w1, x1, y1, z1 = vertsplit(q2)
    
    # Calcular la parte escalar
    scalar_part = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    
    # Calcular la parte vectorial
    vector_part = vertcat(
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    )
    
    # Combinar la parte escalar y vectorial
    q_result = vertcat(scalar_part, vector_part)
    
    return q_result


def quat_p(quat, omega):
    # Crear un cuaternión de omega con un componente escalar 0
    omega_quat = vertcat(MX(0), omega)
    
    # Calcular la derivada del cuaternión
    q_dot = 0.5 * quaternion_multiply(quat, omega_quat)
    
    return q_dot

def quaternion_error(q_real, quat_d):
    norm_q = norm_2(q_real)
   
    
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    
    q_error = quaternion_multiply(q_inv, quat_d)
    return q_error

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode_complete'
    # Dynamic Values of the system
    m = 1
    e = MX([0, 0, 1])
    g = 9.81
    


    # set up states & controls
    # Position
    p1 = MX.sym('p1')
    p2 = MX.sym('p2')
    p3 = MX.sym('p3')
    # Orientation
    v1 = MX.sym('v1')
    v2 = MX.sym('v2')
    v3 = MX.sym('v3')

    # Velocity Linear and Angular
    q0 = MX.sym('q0')
    q1 = MX.sym('q1')
    q2 = MX.sym('q2')
    q3 = MX.sym('q3')

    # Velocidades angulares
    w1 = MX.sym('w1')
    w2 = MX.sym('w2')
    w3 = MX.sym('w3')

    # General vector of the states
    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3)

    # Action variables
    Tt = MX.sym('Tt')
    tau1 = MX.sym('tau1')
    tau2 = MX.sym('tau2')
    tau3 = MX.sym('tau3')

    # General Vector Action variables
    u = vertcat(Tt, tau1, tau2, tau3)

    # Variables to explicit function
 
    p1_p = MX.sym('p1_p')
    p2_p = MX.sym('p2_p')
    p3_p = MX.sym('p3_p')

    v1_p = MX.sym('v1_p')
    v2_p = MX.sym('v2_p')
    v3_p = MX.sym('v3_p')

    q0_p = MX.sym('q0')
    q1_p = MX.sym('q1')
    q2_p = MX.sym('q2')
    q3_p = MX.sym('q3')

    w1_p = MX.sym('w1_p')
    w2_p = MX.sym('w2_p')
    w3_p = MX.sym('w3_p')

    # general vector X dot for implicit function
    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p, q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p)

    # Ref system as a external value
    p1_d = MX.sym('p1_d')
    p2_d = MX.sym('p2_d')
    p3_d = MX.sym('p3_d')
    
    v1_d = MX.sym('v1_d')
    v2_d = MX.sym('v2_d')
    v3_d = MX.sym('v3_d')

    q0_d = MX.sym('q0_d')
    q1_d = MX.sym('q1_d')
    q2_d = MX.sym('q2_d')
    q3_d = MX.sym('q3_d')

    w1_d = MX.sym('w1_d')
    w2_d = MX.sym('w2_d')
    w3_d = MX.sym('w3_d')

    T_d = MX.sym('T_d')
    tau1_d = MX.sym('tau1_d')
    tau2_d = MX.sym('tau2_d')
    tau3_d = MX.sym('tau3_d')
    
    p = vertcat(p1_d, p2_d, p3_d, v1_d, v2_d, v3_d, q0_d, q1_d, q2_d, q3_d, w1_d, w2_d, w3_d, T_d, tau1_d, tau2_d, tau3_d)

    # Crea una lista de MX con los componentes del cuaternión
    quat = vertcat(q0, q1, q2, q3)
    w = vertcat(w1, w2, w3)
    Rot = QuatToRot(quat)
    # Definición de la matriz de inercia I
    
    Jxx = 0.00305587
    Jyy = 0.00159695
    Jzz = 0.00159687

    I = vertcat(
        horzcat(Jxx, 0, 0),
        horzcat(0, Jyy, 0),
        horzcat(0, 0, Jzz)
    )

    u1 = vertcat(0, 0, Tt)
    u2 = vertcat(tau1, tau2, tau3)

    p_p = vertcat(v1, v2, v3)
    v_p = -e*g + ((Rot @ u1)  / m) 

    

    q_p = quat_p(quat, w)  

    print(I)

    w_p = inv(I) @ (u2 - cross(w, I @ w))

    f_expl = vertcat(
        p_p,
        v_p,
        q_p,
        w_p
    )

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model

    f_impl = x_p - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_p
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
    aux_x = np.array(x[:,0]).reshape((13,))
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
    

def log_cuaternion_casadi(q):
 

    # Descomponer el cuaternio en su parte escalar y vectorial
    q_w = q[0]
    q_v = q[1:]

    q = if_else(
        q_w < 0,
        -q,  # Si q_w es negativo, sustituir q por -q
        q    # Si q_w es positivo o cero, dejar q sin cambios
    )

    # Actualizar q_w y q_v después de cambiar q si es necesario
    q_w = q[0]
    q_v = q[1:]
    
    # Calcular la norma de la parte vectorial usando CasADi
    norm_q_v = norm_2(q_v)

    print(norm_q_v)
    
    # Calcular el ángulo theta
    theta = atan2(norm_q_v, q_w)
    
    log_q = 2 * q_v * theta / norm_q_v
    
    return log_q

def create_ocp_solver_description(x0, N_horizon, t_horizon) -> AcadosOcp:
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

    # Matriz de ganancia Posicion
    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 1
    Q_mat[1, 1] = 1
    Q_mat[2, 2] = 1

    K_mat = MX.zeros(3, 3)
    K_mat[0, 0] = 1
    K_mat[1, 1] = 1
    K_mat[2, 2] = 1
    
    # Matriz de ganancia Acciones de contol
    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 0.01
    R_mat[1, 1] = 0.01
    R_mat[2, 2] = 0.01
    R_mat[3, 3] = 0.01

    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:3] - model.x[0:3]
    quat_error = quaternion_error(model.x[6:10], ocp.p[6:10])

    log_q = log_cuaternion_casadi(quat_error)

    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u + log_q.T @ K_mat @ log_q
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose +  log_q.T @  K_mat @ log_q

    # set constraints
    Tmax = 10*9.81
    taux_max = 0.02
    tauy_max = 0.02
    tauz_max = 0.02

    Tmin = -Tmax 
    taux_min = -taux_max
    tauy_min = - tauy_max
    tauz_min = -tauz_max

    ocp.constraints.lbu = np.array([Tmin,taux_min,tauy_min,tauz_min])
    ocp.constraints.ubu = np.array([Tmax,taux_max,tauy_max,tauz_max])
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

def send_full_state_to_sim(state_vector):
    publisher = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    
    # Create an Odometry message
    odometry_msg = Odometry()

    

    odometry_msg.header.frame_id = "odo"
    odometry_msg.header.stamp = rospy.Time.now()
    odometry_msg.pose.pose.position.x = state_vector[0]
    odometry_msg.pose.pose.position.y = state_vector[1]
    odometry_msg.pose.pose.position.z = state_vector[2]
    odometry_msg.pose.pose.orientation.w = state_vector[6]
    odometry_msg.pose.pose.orientation.x = state_vector[7]
    odometry_msg.pose.pose.orientation.y = state_vector[8]
    odometry_msg.pose.pose.orientation.z = state_vector[9]
    odometry_msg.twist.twist.linear.x = state_vector[3]
    odometry_msg.twist.twist.linear.y = state_vector[4]
    odometry_msg.twist.twist.linear.z = state_vector[5]
    odometry_msg.twist.twist.angular.x = state_vector[10]
    odometry_msg.twist.twist.angular.y = state_vector[11]
    odometry_msg.twist.twist.angular.z = state_vector[12]

    
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


def get_odometry_complete():

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

    x_state = [x_real,y_real,z_real,vx_real,vy_real,vz_real, qw_real, qx_real, qy_real, qz_real, wx_real, wy_real, wz_real ]

    return x_state




    
    
def print_state_vector(state_vector):

# Encabezados de los estados
    headers = ["px", "py", "pz", "vx", "vy", "vz", "qx", "qx", "qy", "qz", "w_x", "w_y", "w_z"]
    
    # Verificar que el tamaño del vector de estado coincida con la cantidad de encabezados
    if len(state_vector) != len(headers):
        raise ValueError(f"El vector de estado tiene {len(state_vector)} elementos, pero se esperaban {len(headers)} encabezados.")
    
    # Determinar la longitud máxima de los encabezados para formateo
    max_header_length = max(len(header) for header in headers)
    
    # Imprimir cada encabezado con el valor correspondiente
    for header, value in zip(headers, state_vector):
        formatted_header = header.ljust(max_header_length)
        print(f"{formatted_header}: {value:.2f}")
    
    # Imprimir una línea en blanco para separación
    print()

def publish_matrix(matrix_data, topic_name='/nombre_del_topico'):

   
    # Inicializa el nodo ROS si aún no está inicializado
   

    # Crea una instancia del mensaje Float64MultiArray
    matrix_msg = Float64MultiArray()

    # Convierte la matriz NumPy en una lista plana
    matrix_data_flat = matrix_data.flatten().tolist()

    # Asigna los datos de la matriz al mensaje
    matrix_msg.data = matrix_data_flat

    # Crea un publicador para el tópico deseado
    matrix_publisher = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10)

    # Publica el mensaje en el tópico
    matrix_publisher.publish(matrix_msg)

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
    x = np.zeros((13, t.shape[0]+1-N_prediction), dtype = np.double)

    x[:, 0] = get_odometry_complete()
    x[:, 0] = [1,1,1,0,0,0,1,0,0,0,0,0,0]
    
    
    #TAREA DESEADA
    value = 10
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

    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'


    # Reference Signal of the system
    xref = np.zeros((17, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd         # px_d
    xref[1,:] = hyd         # py_d
    xref[2,:] = hzd         # pz_d 
    xref[3,:] = 0           # vx_d
    xref[4,:] = 0           # vy_d
    xref[5,:] = 0         # vz_d 
    xref[6,:] = quatd[0, :]         # qw_d
    xref[7,:] = quatd[1, :]         # qx_d
    xref[8,:] = quatd[2, :]        # qy_d
    xref[9,:] = quatd[3, :]         # qz_d
    xref[10,:] = 0         # wx_d
    xref[11,:] = 0         # wy_d
    xref[12,:] = 0         # wz_d
    

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

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction+1))
    simU = np.ndarray((nu, N_prediction))

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 1 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    # Simulation System


    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate
    

     # Read Real data
    for ii in range(0,13):
        x[:, 0] = get_odometry_complete()
        print("Loading...")
        rate.sleep()

    print("Ready!!!")
       
    
    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

               # SET REFERENCES
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)

        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')

        #print(simX[:,10])

        u_control[:, k] = simU[:,0]

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        #u_control[:, k] = [9.81, 0.0000   ,0.0000      , 0.00000]
                
        # Send Control values
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        #Imprime El vector de estados
        


        print_state_vector(x[:, k])


        
        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_complete()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            
            send_full_state_to_sim(x[:, k+1])
        else:
            print("Opción no válida")

          
 
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