from ultralytics import YOLO


model = YOLO('yolo11n.pt')

model.train(data='sc_config.yaml', batch=128, epochs=10)