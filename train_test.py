import torch
#print(torch.cuda.get_device_name(0))

from ultralytics import YOLO

model = YOLO("yolo11s.pt")

#Train
model.train(
    data = "data_PPE.yaml",
    epochs = 100,
    imgsz = 640,
    task="detect",
    #device="cpu"
)