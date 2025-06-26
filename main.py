from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
from utils import pixel_to_camera, transform_camera_to_base, read_camera_cfg

# 启动
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# 访问接口测试
# http://127.0.0.1:8000/docs

app = FastAPI()
model = YOLO(r"./detect_model/weights/best.pt")
camera_cfg = read_camera_cfg("./config/camera.yaml")

@app.post("/detect_and_convert")
async def detect_and_convert(
    image: UploadFile = File(...),
    depth: UploadFile = File(...)
    ):
    # 读取彩图
    image_bytes = await image.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(img.shape)
    # 读取深度图
    depth_bytes = await depth.read()
    depth_arr = np.frombuffer(depth_bytes, np.uint8)
    depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)  # 原始格式读取
    if depth_img is None:
        return JSONResponse(content={"error": "深度图读取失败"}, status_code=400)
    print(depth_img.shape)

    # #--------------------------------------#
    # #   检测目标 bbox: [x1, y1, x2, y2]
    # #--------------------------------------#
    # results = model(img)
    # base_coords = []
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     bcx, bcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    #     confidence = float(box.conf[0])
    #     class_id = int(box.cls[0])
        
    #     #---------------#
    #     #   获取深度
    #     #---------------#
    #     if 0 <= bcy < depth_img.shape[0] and 0 <= bcx < depth_img.shape[1]:
    #         depth = int(depth_img[bcy, bcx])
    #         depth_m = depth / 1000.0 if depth > 0 else 0.3  # fallback
    #     else:
    #         depth_m = 0.3
        
    #     #-------------------------#
    #     #   像素坐标 → 相机坐标
    #     #-------------------------#
    #     fx, fy = camera_cfg["camera_matrix"][0, 0], camera_cfg["camera_matrix"][1, 1]
    #     cx, cy = camera_cfg["camera_matrix"][0, 2], camera_cfg["camera_matrix"][1, 2]
    #     camera_coord = pixel_to_camera(bcx, bcy, depth_m, fx, fy, cx, cy)
        
    #     #-----------------------------#
    #     #   相机坐标 → 机械臂底座坐标
    #     #-----------------------------#
    #     base_coord = transform_camera_to_base(camera_coord, camera_cfg["T_cam2base"])
        
    #     base_coords.append({
    #         "pixel": [bcx, bcy],
    #         "camera_coord": camera_coord.tolist(),
    #         "base_coord": base_coord.tolist()
    #     })
    base_coords = [
        {
            "pixel": [300, 300],
            "camera_coord": [0.0, -0.1, 0.3],
            "base_coord": [0.0, -0.1, 0.3]
        }
    ]

    return JSONResponse(content={"objects": base_coords})
