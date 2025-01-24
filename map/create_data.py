import os
import math
from pathlib import Path
import cv2
import random
import numpy as np
import torch


class OriginImageInfo:
	background_list: list[np.ndarray] = []
	background_color: tuple[int, int, int] = (0, 0, 0)
	tile_list: list[list[np.ndarray]] = []
	tile_color_list: list[tuple[int, int, int]] = []
	observer_list: list[np.ndarray] = []
	observer_color: tuple[int, int, int] = (0, 0, 0)
	scourge_list: list[np.ndarray] = []
	scourge_color: tuple[int, int, int] = (0, 0, 0)
	ui_list: list[np.ndarray] = []
	ui_color: tuple[int, int, int] = (0, 0, 0)

	def automatic_init(
		self,
		path: str | Path,
		background_image_size: tuple[int, int] = (448, 448)
	):
		"""
		이미지 폴더에서 이미지를 자동으로 로드합니다.
		:param path: 이미지 폴더 경로
		:param background_image_size: 배경 이미지 크기(가로, 세로)
		:return: 이미지 정보 객체
		"""
		if not os.path.isdir(path):
			print("폴더가 아닙니다.")
			return self
		files = os.listdir(path)
		files.sort()
		for file in files:
			sub_path = os.path.join(path, file)
			if not sub_path.endswith((".png", ".jpg", ".jpeg")):
				continue
			image = cv2.imread(sub_path, cv2.IMREAD_UNCHANGED)  # RGBA로 읽기
			if file.startswith("tile"):
				if image.shape[:2] != (64, 64):
					print("타일 이미지 크기가 64x64가 아닙니다.")
					continue
				number = int(file[4:].split("_")[0].split(".")[0]) - 1
				if len(self.tile_list) <= number:
					self.tile_list.append([])
					average_color = image[:, :, :3].mean(axis=(0, 1)).astype(np.uint8)
					self.tile_color_list.append(tuple(average_color))
				self.tile_list[number].append(image)
			elif file.startswith("observer"):
				if image.shape[:2] != (64, 64):
					print("옵저버 이미지 크기가 64x64가 아닙니다.")
					continue
				self.observer_list.append(image)
				self.observer_color = (0, 255, 255)
			elif file.startswith("scourge"):
				if image.shape[:2] != (64, 64):
					print("스커지 이미지 크기가 64x64가 아닙니다.")
					continue
				self.scourge_list.append(image)
				self.scourge_color = (0, 0, 255)
			elif file.startswith("ui"):
				if image.shape[:2] != (64, 64):
					print("UI 이미지 크기가 64x64가 아닙니다.")
					continue
				self.ui_list.append(image)
				self.ui_color = (0, 255, 0)
			elif file.startswith("background"):
				ih, iw = image.shape[:2]
				if iw < background_image_size[0] or ih < background_image_size[1]:
					print(f"배경 이미지 크기가 {background_image_size}보다 작습니다.")
					continue
				self.background_list.append(image)
				self.background_color = (0, 0, 0)
		return self


def paste_image(
		target_image: np.ndarray,
		target_mark: np.ndarray,
		source_image: np.ndarray,
		source_mark: np.ndarray | tuple[int, float],
		pos: tuple[int, int],
		anker: tuple[float, float] = (0.0, 0.0),
		alpha: float = 1.0,
		blending_mode: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
	"""
	target_image에 source_image를 pos 위치에 붙여넣습니다.
	:param target_image: 붙여넣을 대상 이미지 (H, W, 4) RGBA 형식
	:param target_mark: 붙여넣을 대상 마스크 (C, H, W) 클래스 수
	:param source_image: 붙여넣을 이미지 (h, w, 4) RGBA 형식
	:param source_mark: 붙여넣을 마스크 (C, h, w) 또는 (클래스 번호, 확률)
	:param pos: 붙여넣을 위치 (y, x)
	:param anker: 붙여넣을 앵커 위치 (y, x) 비율 (0.0 ~ 1.0)
	:param alpha: 붙여넣기 알파값 (0.0 ~ 1.0)
	:param blending_mode: 블렌딩 모드 (normal, surface)
	:return: 업데이트된 이미지와 마스크
	"""
	# 원본 이미지 및 마스크 크기
	h, w = source_image.shape[:2]
	c = target_mark.shape[0]  # 클래스 수

	# 앵커 기준 위치 계산
	y_offset = int(pos[0] - h * anker[0])
	x_offset = int(pos[1] - w * anker[1])

	# 붙여넣기 영역
	y1, y2 = max(0, y_offset), min(target_image.shape[0], y_offset + h)
	x1, x2 = max(0, x_offset), min(target_image.shape[1], x_offset + w)

	# 소스 이미지의 유효 영역
	sy1, sy2 = max(0, -y_offset), min(h, target_image.shape[0] - y_offset)
	sx1, sx2 = max(0, -x_offset), min(w, target_image.shape[1] - x_offset)

	# 알파 채널 추출 및 적용
	source_alpha = (source_image[sy1:sy2, sx1:sx2, 3:] / 255.0) * alpha
	source_inv_alpha = 1.0 - source_alpha

	# 이미지 업데이트
	for c in range(4):  # RGBA 채널
		if blending_mode == "normal":
			target_image[y1:y2, x1:x2, c] = (
				source_image[sy1:sy2, sx1:sx2, c] * source_alpha[:, :, 0] +
				target_image[y1:y2, x1:x2, c] * source_inv_alpha[:, :, 0]
			)
		elif blending_mode == "surface":
			target_image[y1:y2, x1:x2, c] = (
				source_image[sy1:sy2, sx1:sx2, c] +
				target_image[y1:y2, x1:x2, c] * source_inv_alpha[:, :, 0]
			)

	# 마스크 업데이트
	if isinstance(source_mark, tuple):
		class_idx, prob = source_mark
		if blending_mode == "normal":
			target_mark[:, y1:y2, x1:x2] *= (1.0 - prob * source_alpha[:, :, 0])
			target_mark[class_idx, y1:y2, x1:x2] += prob * source_alpha[:, :, 0]
		elif blending_mode == "surface":
			target_mark[class_idx, y1:y2, x1:x2] *= (1.0 - prob)
			target_mark[class_idx, y1:y2, x1:x2] += prob
	else:
		if blending_mode == "normal":
			target_mark[:, y1:y2, x1:x2] *= (1.0 - source_alpha[:, :, 0])
			target_mark[:, y1:y2, x1:x2] += source_mark[:, sy1:sy2, sx1:sx2] * source_alpha[:, :, 0]
		elif blending_mode == "surface":
			target_mark[:, y1:y2, x1:x2] *= (1.0 - source_alpha[:, :, 0])
			target_mark[:, y1:y2, x1:x2] += source_mark[:, sy1:sy2, sx1:sx2]

	return target_image, target_mark


def make_data(
	image_info: OriginImageInfo,
	observer_count: int = 10,
	observer_alpha: float = 0.5,
	sample_count: int = 10,
	target_image_size: tuple[int, int] = (448, 448),
	# random_seed: None | int = None,
):
	# if random_seed is not None:
	# 	random.seed(random_seed)  # 랜덤 시드 설정

	data_list = []
	mask_list = []

	tile_class_offset = 4
	num_classes = tile_class_offset + len(image_info.tile_list)  # 배경, 옵저버, 스커지, UI, 타일 클래스
	tiw = target_image_size[0]  # 타일 이미지 크기
	tih = target_image_size[1]

	# 1. 바탕 이미지 생성
	tw = 64  # 타일 크기
	th = 64
	biw = tiw + tw  # 배경 이미지 크기
	bih = tih + th
	canvas = np.zeros((bih, biw, 4), dtype=np.uint8)
	mask = np.zeros((num_classes, bih, biw), dtype=np.float32)
	mask[0, :, :] = 1.0  # 배경 클래스 채우기

	# 2. 8x8 타일 랜덤 배치
	max_i = math.ceil(bih / th)
	max_j = math.ceil(biw / tw)
	for i in range(max_i):
		for j in range(max_j):
			tile_index = random.randrange(0, len(image_info.tile_list) + 1)
			y, x = i * th, j * tw
			if len(image_info.tile_list) <= tile_index:
				continue
			tile_list = image_info.tile_list[tile_index]
			tile = random.choice(tile_list)
			paste_image(
				canvas, mask,
				tile, (tile_class_offset + tile_index, 1.0),
				(y, x),
			)

	for _ in range(sample_count):
		# 3. 배경 선택 및 뷰 이동
		select_background = random.choice(image_info.background_list)
		select_background_h, select_background_w = select_background.shape[:2]
		rw = random.randint(0, select_background_w - tiw)
		rh = random.randint(0, select_background_h - tih)
		sub_canvas = select_background.copy()[rh:rh + tih, rw:rw + tiw]
		sub_mask = np.zeros((num_classes, tih, tiw), dtype=np.float32)
		x = random.randrange(0, tw)
		y = random.randrange(0, th)
		paste_image(
			target_image=sub_canvas, target_mark=sub_mask,
			source_image=canvas, source_mark=mask,
			pos=(-y, -x),
			blending_mode="surface",
		)

		# 4. 스커지 중앙 배치
		scourge = random.choice(image_info.scourge_list)
		x = random.randrange(0, tw // 2) + tiw // 2
		y = random.randrange(0, th // 2) + tih // 2
		paste_image(
			target_image=sub_canvas, target_mark=sub_mask,
			source_image=scourge, source_mark=(1, 1.0),
			pos=(y, x), anker=(0.5, 0.5),
		)

		# 5. 옵저버 랜덤 배치
		for _ in range(observer_count):
			observer = random.choice(image_info.observer_list)
			x, y = random.randrange(0, tiw), random.randrange(0, tih)
			paste_image(
				target_image=sub_canvas, target_mark=sub_mask,
				source_image=observer, source_mark=(2, 1.0),
				pos=(y, x), anker=(0.5, 0.5),
				alpha=observer_alpha,
			)

		# 6. UI 랜덤 배치
		ui = random.choice(image_info.ui_list)
		x, y = random.randrange(0, tiw), random.randrange(0, tih)
		paste_image(
			sub_canvas, sub_mask,
			ui, (3, 1.0),
			(y, x), (0.5, 0.5),
		)

		# 7. 결과 저장
		data_list.append(torch.tensor(sub_canvas[:, :, :3].transpose(2, 0, 1) / 255.0, dtype=torch.float32))  # RGB 텐서
		sub_mask /= sum(sub_mask)  # 정규화
		mask_list.append(torch.tensor(sub_mask))

	return data_list, mask_list


def get_mask_visual(image_info: OriginImageInfo, mask: np.ndarray):
	colors = np.array(
		[image_info.background_color, image_info.scourge_color, image_info.observer_color, image_info.ui_color] +
		image_info.tile_color_list,
		dtype=np.uint8
	)
	return np.tensordot(mask.transpose(1, 2, 0), colors, axes=([2], [0])).astype(np.uint8)


if __name__ == "__main__":
	import time
	random.seed(0)

	target_image_size = (927, 576)
	# target_image_size = (448, 448)

	# 데이터 로드 및 생성
	image_info = OriginImageInfo().automatic_init("../source", background_image_size=target_image_size)
	start_time = time.time()
	data_list, mask_list = [], []
	for _ in range(10):
		a, b = make_data(image_info, observer_alpha=0.5, sample_count=3, target_image_size=target_image_size)
		data_list.extend(a)
		mask_list.extend(b)
	print(f"데이터 생성 시간: {time.time() - start_time:.3f}초")
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

	# 데이터 확인
	for i, (data, mask) in enumerate(zip(data_list, mask_list)):
		data_np = (data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # RGB 데이터
		colors = np.array(
			[image_info.background_color, image_info.scourge_color, image_info.observer_color, image_info.ui_color] +
			image_info.tile_color_list,
			dtype=np.uint8
		)
		mask_np = mask.numpy()

		mask_visual = np.tensordot(mask_np.transpose(1, 2, 0), colors, axes=([2], [0])).astype(np.uint8)
		concat = np.concatenate((data_np, mask_visual), axis=1)
		cv2.imshow(f"Image", concat)
		# cv2.resizeWindow("Image", (concat.shape[1] // 2, concat.shape[0] // 2))

		stop_flag = False
		while True:
			key = cv2.waitKey(10)
			if key == 27:
				stop_flag = True
				break
			elif key != -1:
				break
		# cv2.destroyAllWindows()
		if stop_flag:
			break
