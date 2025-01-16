import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import segmentation_models_pytorch as smp
from util import auto_train, AutoDataset

if __name__ == "__main__":
	seed = 42
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)

	auto_dataset = AutoDataset(
		"../source",  # 데이터 소스 디렉터리
		batch_size=16,  # 배치 크기
	)

	model = smp.MAnet(
		encoder_name="resnet18",
		encoder_depth=5,  # 디코더에서 사용하는 단계 수
		encoder_weights="imagenet",  # ImageNet으로 사전 학습된 가중치 사용
		in_channels=3,  # 입력 채널 (RGB)
		decoder_channels=[256, 128, 64, 32, 16],  # 디코더 채널 수
		classes=auto_dataset.num_class,  # 클래스 수
		# activation="softmax2d"  # 소프트맥스로 출력 (다중 클래스)
	)
	model = model.to("mps")

	auto_train(
		f"../weight/{model.name}.pth",
		auto_dataset,
		model,
		epochs=10,
	)
