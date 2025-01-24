from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from train.util import (auto_train, AutoDataset)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x):
		identity = self.shortcut(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += identity
		return self.relu(out)


class ResNet(nn.Module):
	def __init__(self, input_channels=3):
		super(ResNet, self).__init__()
		# ResNet의 각 단계 정의
		self.initial = nn.Sequential(
			nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)
		self.layer1 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			ResidualBlock(32, 32),
			ResidualBlock(32, 32)
		)
		self.layer2 = nn.Sequential(
			ResidualBlock(32, 64, stride=2),
			ResidualBlock(64, 64)
		)
		self.layer3 = nn.Sequential(
			ResidualBlock(64, 128, stride=2),
			ResidualBlock(128, 128)
		)
		self.layer4 = nn.Sequential(
			ResidualBlock(128, 256, stride=2),
			ResidualBlock(256, 256)
		)

	def get_stage(self):
		return [
			nn.Identity(),
			self.initial,
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4
		]

	def forward(self, x):
		features = []
		stages = self.get_stage()
		for i in range(len(stages)):
			x = stages[i](x)
			features.append(x)
		return features  # 각 단계의 특징 맵을 반환


class DecoderBlock(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels):
		super(DecoderBlock, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x, skip_connection):
		x = F.interpolate(x, size=skip_connection.size()[2:], mode="bilinear", align_corners=False)
		x = torch.cat((x, skip_connection), dim=1)
		x = self.conv1(x)
		return self.conv2(x)


class UNet(nn.Module):
	def __init__(self, input_channels=3, num_classes=1):
		super(UNet, self).__init__()
		self.encoder = ResNet(input_channels=input_channels)

		# 디코더 블록
		self.up5 = DecoderBlock(256, 128, 128)
		self.up4 = DecoderBlock(128, 64, 64)
		self.up3 = DecoderBlock(64, 32, 32)
		self.up2 = DecoderBlock(32, 32, 32)
		self.up1 = DecoderBlock(32, 3, 32)

		# # 최종 출력 레이어
		self.final = nn.Conv2d(32, num_classes, kernel_size=1)

	def forward(self, x):
		features = self.encoder(x)
		x = self.up5(features[-1], features[-2])  # layer4 + layer3
		x = self.up4(x, features[-3])  # layer3 + layer2
		x = self.up3(x, features[-4])  # layer2 + layer1
		x = self.up2(x, features[-5])  # layer1 + initial
		x = self.up1(x, features[-6])  # initial + identity
		x = self.final(x)
		return x


if __name__ == "__main__":
	seed = 42
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)

	auto_dataset = AutoDataset(
		Path(__file__).parent / "../source",  # 데이터 소스 디렉터리
		batch_size=16,  # 배치 크기
		dataset_count=1000,
		sample_count=2,
		# target_size=(927, 576)  # 생성할 이미지 크기
		target_size=(448, 448)
	)

	encoder_name = "resnet_hd"
	model = UNet(
		input_channels=3,  # 입력 채널 (RGB)
		num_classes=auto_dataset.num_class,  # 클래스 수
	)
	model = model.to("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

	auto_train(
		Path(__file__).parent / f"../weight/u-resnet_hd.pth",
		auto_dataset,
		model,
		epochs=100,
		save_every_epoch=True
	)
