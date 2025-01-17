import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# 모델 로드
# model = lraspp_mobilenet_v3_large(pretrained=True)
model = lraspp_mobilenet_v3_large(pretrained=True)
# model.to("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model.eval()

# 이미지 전처리
image_path = "path/game_scene_00001.png"
image = Image.open(image_path).convert("RGB")

# 이미지 크롭(아래 중앙, 448x448)
org_w, org_h = image.size
x, y, w, h = org_w // 2 - 224, org_h - 448, 448, 448
image = image.crop((x, y, x + w, y + h))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)
# input_tensor = input_tensor.to(
#     "cuda" if torch.cuda.is_available() else
#     ("mps" if torch.backends.mps.is_available() else "cpu")
# )

# 추론
with torch.no_grad():
    # 속도 측정
    start = time.time()
    output = model(input_tensor)
    print(f"추론 시간: {time.time() - start:.3f}초")
segmentation = output["out"]
segmentation = torch.argmax(segmentation.squeeze(), dim=0).cpu().numpy()

# 색상 맵 시각화
palette = np.array([
    [0, 100, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0]
], dtype=np.uint8)
segmentation_rgb = palette[segmentation]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Segmentation Result")
plt.imshow(segmentation_rgb)
plt.show()