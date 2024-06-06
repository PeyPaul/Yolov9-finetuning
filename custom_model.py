from ultralytics import YOLO

model = YOLO('train/weights/best.pt') # Load model

results = model.predict(source = 0, imgsz = 640, conf = 0.3, save = False, show = True) # Run inference on webcam