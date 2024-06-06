from ultralytics import YOLO

model = YOLO('train/weights/best.pt') # Load model

results = model.predict(source = 0, imgsz = 640, conf = 0.3, save = False, show = True) # Run inference on webcam
# Can also use model.predict('data/images') to run inference on images in a directory, or model.predict('data/video.mp4') for video
# In that case, set save = True to save the results to a directory