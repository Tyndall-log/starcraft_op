import os
from pathlib import Path
import torch
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from map.create_data import make_data, OriginImageInfo
import tqdm


class CustomSegmentationDataset(Dataset):
	def __init__(self, data_list, mask_list, transform=None):
		"""
		데이터와 마스크 리스트를 받아 Custom Dataset 생성
		:param data_list: 학습 이미지 데이터 리스트 (Tensor 형태)
		:param mask_list: 대응하는 마스크 데이터 리스트 (Tensor 형태)
		:param transform: 이미지 데이터에 적용할 변환 (예: ToTensor, Normalize 등)
		"""
		self.data_list = data_list
		self.mask_list = mask_list
		self.transform = transform

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		image = self.data_list[idx]  # Tensor 형태의 이미지 데이터
		mask = self.mask_list[idx]  # Tensor 형태의 마스크 데이터

		if self.transform:
			image = self.transform(image)
			mask = self.transform(mask)

		return image, mask


def train_one_epoch(model, train_loader, criterion, optimizer, device):
	model.train()
	epoch_loss = 0.0

	for images, masks in tqdm.tqdm(train_loader, desc="Training", leave=False):
		images = images.to(device)
		masks = masks.to(device)  # (N, P, H, W)

		optimizer.zero_grad()

		outputs = model(images)
		if isinstance(outputs, dict) and 'out' in outputs:
			outputs = model(images)['out']

		# 손실 계산
		loss = criterion(outputs, masks)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device):
	model.eval()
	epoch_loss = 0.0

	with torch.no_grad():
		for images, masks in tqdm.tqdm(val_loader, desc="Validation", leave=False):
			images = images.to(device)
			masks = masks.to(device)  # (N, P, H, W)

			outputs = model(images)
			if isinstance(outputs, dict) and 'out' in outputs:
				outputs = model(images)['out']
			loss = criterion(outputs, masks)
			epoch_loss += loss.item()

	return epoch_loss / len(val_loader)


class DynamicSegmentationDataset(Dataset):
	def __init__(
		self,
		image_info,
		observer_alpha=None,
		sample_count=1,
		transform=None,
		target_size=None,
		dataset_size=100
	):
		"""
		매 에폭마다 동적으로 데이터를 생성하는 Dataset
		:param image_info: 이미지 정보 (OriginImageInfo)
		:param observer_alpha: 옵저버 알파 값
		:param sample_count: 샘플 생성 개수
		:param transform: 데이터 변환
		:param target_size: 타겟 이미지 크기
		:param dataset_size: Dataset 크기 (에폭마다 생성할 데이터 개수)
		"""
		self.image_info = image_info
		self.observer_alpha = observer_alpha
		self.sample_count = sample_count
		self.transform = transform
		self.target_size = target_size
		self.dataset_size = dataset_size

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		# 데이터 동적 생성
		params = {}
		if self.observer_alpha is not None:
			params['observer_alpha'] = self.observer_alpha
		if self.target_size is not None:
			params['target_image_size'] = self.target_size
		data, mask = make_data(
			self.image_info,
			sample_count=self.sample_count,
			**params
		)
		data, mask = data[0], mask[0]  # make_data는 리스트를 반환하므로 첫 번째 데이터 사용

		if self.transform:
			data = self.transform(data)
			mask = self.transform(mask)

		return data, mask


class AutoDataset:
	train_loader: DataLoader
	val_loader: DataLoader
	num_class: int

	def __init__(
		self,
		image_source_path: str | Path,
		batch_size: int = 16,
		data_split_ratio: float = 0.8,
		observer_alpha: float | None = None,
		dataset_count: int = 100,
		sample_count: int = 1,
		# transform: T.Compose = T.Compose([
		# 	T.Resize((224, 224)),  # 224x224로 리사이즈
		# ]),
		target_size: tuple[int, int] = (448, 448),
	):
		# 이미지 정보 초기화
		self.image_info = OriginImageInfo().automatic_init(image_source_path, background_image_size=target_size)

		# 검증 데이터 고정 생성
		data_list, mask_list = [], []
		val_dataset_count = int(dataset_count * (1 - data_split_ratio))

		for _ in tqdm.tqdm(range(val_dataset_count), desc="Creating Validation Dataset"):
			params = {}
			if observer_alpha is not None:
				params['observer_alpha'] = observer_alpha
			data, mask = make_data(
				self.image_info,
				sample_count=sample_count,
				target_image_size=target_size,
				**params,
			)
			data_list.extend(data)
			mask_list.extend(mask)

		self.num_class = mask_list[0].shape[0]

		# 검증 데이터셋 생성
		self.val_dataset = CustomSegmentationDataset(data_list, mask_list)

		# DataLoader 생성
		self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

		# 학습 데이터셋은 동적 생성
		self.train_dataset = DynamicSegmentationDataset(
			image_info=self.image_info,
			observer_alpha=observer_alpha,
			sample_count=sample_count,
			target_size=target_size,
			dataset_size=int(dataset_count * data_split_ratio),
		)
		self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)


def auto_train(
		save_model_path: str | Path,
		auto_dataset: AutoDataset,
		model: nn.Module,
		epochs: int,
		criterion=None,
		optimizer=None,
		save_every_epoch: bool = False,
	):
	# 데이터셋 생성
	train_loader = auto_dataset.train_loader
	val_loader = auto_dataset.val_loader

	# 손실 함수 및 옵티마이저 설정
	if criterion is None:
		criterion = nn.CrossEntropyLoss()
	if optimizer is None:
		optimizer = optim.Adam(model.parameters(), lr=1e-3)

	# 학습
	best_loss = float('inf')
	device = str(next(model.parameters()).device)
	print(f"Device: {device}")
	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}/{epochs}")
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss = validate_one_epoch(model, val_loader, criterion, device)
		print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

		# 모델 저장
		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(model.state_dict(), save_model_path)
			print("모델 저장 완료!")
		if save_every_epoch:
			# 폴더 생성
			path_model_name = os.path.split(save_model_path)
			path = path_model_name[0]
			model_name = path_model_name[1].split('.')[0]
			new_path = os.path.join(path, model_name)
			os.makedirs(new_path, exist_ok=True)
			torch.save(model.state_dict(), f"{new_path}/{model_name}_{epoch + 1}_{train_loss:.4f}_{val_loss:.4f}.pth")


if __name__ == "__main__":
	# 라이브러리 전용 파일 고지
	print("이 파일을 직접 실행할 수 없습니다.")
	print("반드시 다른 파일에서 import하여 사용해주세요.")