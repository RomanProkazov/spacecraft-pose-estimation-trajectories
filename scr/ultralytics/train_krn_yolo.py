from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')

model.train(data='sc_krn_config.yaml', batch=128, epochs=100, mosaic=False)