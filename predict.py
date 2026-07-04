from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="test.png",  
    imgsz=224,
    save=True
)

for result in results:
    probs = result.probs

    print(f"Predicted Class : {result.names[probs.top1]}")
    print(f"Confidence      : {probs.top1conf:.2f}")