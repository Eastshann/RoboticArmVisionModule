import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("无法打开摄像头")
else:
    # 设置分辨率
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 读取实际设置是否成功
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"设置后的分辨率为: {int(width)} x {int(height)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示捕获到的画面
        cv2.imshow('RGB Camera Feed', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
