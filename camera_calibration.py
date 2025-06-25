import cv2
import numpy as np
import glob

# === 你需要根据自己的棋盘格实际设置修改下面两个参数 ===
# 棋盘格内角点数量（注意：是内角点数量，不是格子数量！）
chessboard_size = (11, 8)  # 例如 9x7 内角点

# 每个格子的实际尺寸（单位：毫米）
square_size = 20  # 例如每个格子30mm

# === 读取所有图像 ===
# 假设你的图片都在 "./calib_images/" 文件夹下
images = glob.glob('./calib_images2/*.jpg')

# 存储角点的三维世界坐标 和 二维图像坐标
objpoints = []  # 3d 点
imgpoints = []  # 2d 点

# 生成世界坐标系下的角点
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 乘上实际格子大小

# 遍历所有图片
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 可视化检测结果
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(200)
    else:
        print(f"角点未检测成功: {fname}")

cv2.destroyAllWindows()

# 执行标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 打印标定结果
print("\n相机内参矩阵 (Camera Matrix):\n", camera_matrix)
print("\n畸变系数 (Distortion Coefficients):\n", dist_coeffs.ravel())

# 计算重投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("\n平均重投影误差: ", total_error / len(objpoints))
