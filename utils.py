import math

import cv2
import numpy as np

def generate_checkerboard(
        num_cols=9,  # 棋盘格列数（格子数量）
        num_rows=7,  # 棋盘格行数（格子数量）
        square_size_mm=30,  # 每个格子的实际尺寸（毫米）
        pixels_per_square=1000,  # 每格多少像素，控制分辨率
        output_filename="checkerboard.png"
    ):
    """
    生成棋盘格标定板
    
    Args:
        num_cols (int): 棋盘格列数（格子数量
        num_rows (int): 棋盘格行数（格子数量）
        square_size_mm (int): 每个格子的实际尺寸（毫米）
        pixels_per_square (int): 每格多少像素，控制分辨率
        output_filename (str): 保存的文件名

    Returns:
        None
    """
    
    # 计算棋盘格尺寸
    board_width = num_cols * pixels_per_square
    board_height = num_rows * pixels_per_square

    # 增加边框，方便打印时居中
    margin = pixels_per_square
    image_width = board_width + 2 * margin
    image_height = board_height + 2 * margin

    # 创建全白图像
    image = np.full((image_height, image_width), 255, dtype=np.uint8)

    # 绘制黑白格子
    for j in range(num_rows):
        for i in range(num_cols):
            if (i + j) % 2 == 0:
                x_start = margin + i * pixels_per_square
                y_start = margin + j * pixels_per_square
                x_end = x_start + pixels_per_square
                y_end = y_start + pixels_per_square
                image[y_start:y_end, x_start:x_end] = 0

    # 保存图像
    cv2.imwrite(output_filename, image)
    print(f"保存成功: {output_filename}")
    print(f"图像尺寸: {image_width} x {image_height} 像素")
    print(f"打印时每格大小为：{square_size_mm} mm，请按此比例打印")

    # 显示图像
    cv2.imshow("Checkerboard", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_points(board_size, square_size):
    """
    生成每个棋盘格角点的三维坐标 (世界坐标, Z=0)

    Args:
        board_size (tuple): 棋盘格内角点数量
        square_size (float): 每格实际物理长度（单位：米）
    
    Returns:
        object_points (np.array): 每个点的三维坐标
    """
    object_points = []
    for i in range(board_size[1]):
        for j in range(board_size[0]):
            object_points.append([j * square_size, i * square_size, 0])
    object_points = np.array(object_points, dtype=np.float64)
    
    return object_points

def convert_XYZWPR_to_mat(X, Y, Z, W, P, R):
    """
    将 XYZWPR 位姿格式转换为旋转矩阵和平移向量
    输出为 gripper2base
    
    Args:
        X (float): X轴坐标
        Y (float): Y轴坐标
        Z (float): Z轴坐标
        W (float): 偏航度数
        P (float): 俯仰度数
        R (float): 横滚度数
    
    Returns:
        R_total (np.array): 机械臂末端到基底坐标系的旋转矩阵
        t (np.array): 机械臂末端到基底坐标系的平移矩阵
    """
    # 平移向量 (单位通常是mm)
    t = np.array([[X], [Y], [Z]], dtype=np.float64)

    # 将欧拉角 (W, P, R) 从度转为弧度
    W = math.radians(W)
    P = math.radians(P)
    R = math.radians(R)

    # 绕 Z 轴旋转矩阵
    Rz = np.array([
        [math.cos(W), -math.sin(W), 0],
        [math.sin(W),  math.cos(W), 0],
        [0, 0, 1]
    ])

    # 绕 Y 轴旋转矩阵
    Ry = np.array([
        [math.cos(P), 0, math.sin(P)],
        [0, 1, 0],
        [-math.sin(P), 0, math.cos(P)]
    ])

    # 绕 X 轴旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(R), -math.sin(R)],
        [0, math.sin(R), math.cos(R)]
    ])
    # 最终旋转矩阵 (注意旋转顺序: Z -> Y -> X)
    R_total = Rz @ Ry @ Rx
    return R_total, t


def get_target2cam(image_path, board_size, object_points, camera_matrix, dist_coeffs):
    """
    读取图像，做角点检测，计算标定板坐标到相机坐标系的旋转平移矩阵
    输出为target2cam
    
    Args:
        image_path (str): 图片路径
        board_size (tuple): 棋盘格角点数量, 如(11x8)
        object_points (np.array): 所有角点的三维坐标
        camera_matrix (np.array): 相机内参矩阵
        dist_coeffs (np.array): 相机畸变参数
        
    Returns:
        Rtc (np.array): 标定板坐标到相机坐标系的旋转矩阵
        tvec (np.array): 标定板坐标到相机坐标系的平移矩阵
    """
    # 读取图像 (注意路径根据实际修改)
    image = cv2.imread(image_path)
    if image is None:
        print(f"图像 {image_path} 读取失败")
        return None, None
    
    # 棋盘格角点检测
    ret, corners = cv2.findChessboardCorners(image, board_size)
    
    if ret:
        # 亚像素角点精确化 (提高 solvePnP 精度)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # 使用PnP求解相机坐标系下的棋盘格位姿
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
        
        # 将旋转向量转成旋转矩阵
        Rtc, _ = cv2.Rodrigues(rvec)
        return Rtc, tvec
    
    else:
        print(f"图像 {image_path} 未找到棋盘格")
        return None, None
    
    
    
def get_deflection(fx, fy, u0, v0, u, v):
    """
    计算像素点到图像中心的偏移角度
    
    Args:
        fx (float): 相机内参
        fy (float): 相机内参
        u0 (int): 图像中心x坐标
        v0 (int): 图像中心y坐标
        u (int): 像素点x坐标
        v (int): 像素点y坐标
        
    Returns
        theta_yaw (float): 左右转动度数
        theta_pitch (float): 上下转动度数
    """

    # 计算偏移
    delta_u = u - u0
    delta_v = v - v0

    # 计算角度
    theta_yaw = np.arctan(delta_u / fx)  # 左右
    theta_pitch = -np.arctan(delta_v / fy)  # 上下
    
    # 弧度 -> 角度
    theta_yaw = np.degrees(theta_yaw)
    theta_pitch = np.degrees(theta_pitch)

    # 输出角度 (单位：)
    print("需要左右转动: {:.2f}度".format(theta_yaw))
    print("需要上下转动: {:.2f}度".format(theta_pitch))
    
    return theta_yaw, theta_pitch


def angle_to_control_value(angle_deg):
    """将角度（-120~120）映射为控制值（0~1000）"""
    return int((angle_deg / 120) * 500 + 500)

def control_value_to_angle(control_value):
    """将控制值（0~1000）映射为角度（-120°~120°）"""
    return ((control_value - 500) / 500) * 120


def pixel_to_camera(u, v, depth, fx, fy, cx, cy):
    """
    将像素坐标和深度信息转换为相机坐标系下的三维坐标

    Args:
        u (int): 像素的x坐标
        v (int): 像素的y坐标
        depth (float): 像素点的深度值（单位：米
        fx (float): 相机的水平焦距（像素单位）
        fy (float): 相机的垂直焦距（像素单位）
        cx (float): 主点x坐标，即图像中心点横坐标
        cy (float): 主点y坐标，即图像中心点纵坐标

    Returns:
        tuple: (X, Y, Z) — 该点在相机坐标系下的三维坐标
    """
    Z = depth  # 单位为米
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


def transform_camera_to_base(point_camera, T_base_camera):
    """
    将相机坐标系下的点转换到机械臂基座坐标系

    Args:
        point_camera (array-like): [X, Y, Z] 相机坐标系下的三维点
        T_base_camera (np.ndarray): 4x4 的变换矩阵，相机坐标系 -> 机械臂基底坐标系

    Returns:
        np.ndarray: [X, Y, Z] 机械臂基底坐标系下的三维点
    """
    # 转为齐次坐标
    point_cam_hom = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
    
    # 执行坐标变换
    point_base_hom = T_base_camera @ point_cam_hom
    
    # 去除齐次项，返回3D坐标
    return point_base_hom[:3]