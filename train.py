import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

image_size = 416

# Create YAML content with absolute paths
base_path = os.path.abspath('./split_data')
yaml_content = f"""path: {base_path}
train: {os.path.join(base_path, 'train', 'images')}
val: {os.path.join(base_path, 'val', 'images')}
test: {os.path.join(base_path, 'test', 'images')}

nc: 1  # number of classes
names: ['solar_panel']  # class names
"""
with open(os.path.join(base_path, 'data.yaml'), 'w') as f:
    f.write(yaml_content)
print(f"data.yaml created at:\n{os.path.join(base_path, 'data.yaml')}")
print(yaml_content)


# Use any model from Ultralytics like YOLO to train the object detection model.
model = YOLO("yolo12m.pt")

results = model.train(
    data="split_data/data.yaml",
    epochs=70,
    imgsz=416,
    batch=100,
    device=device,
    project="models",
    name="yolo12m",
)

model.save("yolo12m_e70_b100.pt")