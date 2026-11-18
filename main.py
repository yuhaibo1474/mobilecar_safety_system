from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r"ultralytics/cfg/models/v8/zhongjianronghe.yaml")
    # model.train(data=r"ultralytics/cfg/datasets/mydata.yaml")  # 训练

    model = YOLO(r"/home/b304/data2/temp_det/YOLOv8-Multi-Modal-Fusion-Network-RGB-IR-main/ultralytics/runs/detect/train9/weights/best.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml",batch=1)  # 验证
    # model.predict(source=r"datasets/LLVIP700/images/test", save=True,imgsz=320)  # 检测
    results = model.predict(source="datasets/LLVIP700/images/test", save=True,imgsz=320)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()

        # 保存为CSV逐图像
        import pandas as pd
        df = pd.DataFrame({
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "x0":(boxes[:, 0] +boxes[:, 2]) / 2,
            "y0":(boxes[:, 1] +boxes[:, 3]) / 2,
            "confidence": confs,
            "class": cls_ids
        })
        df.to_csv(f"/home/b304/data2/temp_det/YOLOv8-Multi-Modal-Fusion-Network-RGB-IR-main/ultralytics/runs/detect/detections/{r.path.split('/')[-1]}.csv", index=False)