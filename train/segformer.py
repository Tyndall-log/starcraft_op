from pathlib import Path
import torch
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
		Path(__file__).parent / "../source",  # 데이터 소스 디렉터리
		batch_size=16,  # 배치 크기
		dataset_count=100,
		sample_count=1,
	)

	encoder_name = "mit_b0"  # MIT 백본 선택 (e.g., mit_b0, mit_b2, mit_b5 등)
	model = smp.Segformer(
		encoder_name=encoder_name,
		encoder_depth=5,  # 디코더에서 사용하는 단계 수
		encoder_weights="imagenet",  # ImageNet으로 사전 학습된 가중치 사용
		in_channels=3,  # 입력 채널 (RGB)
		decoder_segmentation_channels=256,  # 디코더 채널 수
		classes=auto_dataset.num_class,  # 클래스 수
	)
	model = model.to("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

	auto_train(
		Path(__file__).parent / f"../weight/{model.name}.pth",
		auto_dataset,
		model,
		epochs=10,
	)
