from ultralytics import YOLO


model = YOLO('yolo11n-seg.pt')

model.train(data='sc_seg_config.yaml', batch=128, epochs=100)