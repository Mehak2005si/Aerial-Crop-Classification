from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data="dataset",
    epochs=60,
    imgsz=224,
    batch=4,
    patience=10,
    project="runs",
    name="crop_classifier"
)