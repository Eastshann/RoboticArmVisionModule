import yaml
import numpy as np

with open("./config/camera.yaml", "r", encoding='utf-8') as f:
    data = yaml.safe_load(f)

# 还原为 NumPy 矩阵
camera_matrix = np.array(data["camera_matrix"]["data"]).reshape(
    data["camera_matrix"]["rows"], data["camera_matrix"]["cols"]
)
dist_coeffs = np.array(data["distortion_coefficients"]["data"])

T_cam2base = np.array(data["camera_to_base_matrix"]["data"]).reshape(
    data["camera_to_base_matrix"]["rows"],
    data["camera_to_base_matrix"]["cols"]
)

print("相机内参:\n", camera_matrix, "\n")
print("畸变系数:\n", dist_coeffs, "\n")
print("相机到基底的变换矩阵:\n", T_cam2base)
