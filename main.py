import cv2
import os
from pathlib import Path
import time
import numpy as np
import torch
from map.create_data import OriginImageInfo, get_mask_visual
import torchvision.transforms as T
import segmentation_models_pytorch as smp


def image_segment(model, image_info, image):
	# 정사각형 이미지로 크롭
	org_h, org_w = image.shape[:2]
	x, y, w, h = org_w // 2 - 224, org_h - 448, 448, 448
	input_image = image[y:y + h, x:x + w]
	input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float()
	input_tensor /= 255.0

	# 이미지 전처리
	preprocess = T.Compose([T.Resize((224, 224))])
	input_tensor = preprocess(input_tensor)
	# input_tensor = input_tensor.to(
	# 	"cuda" if torch.cuda.is_available() else
	# 	("mps" if torch.backends.mps.is_available() else "cpu")
	# )

	# 추론
	start_time = time.time()
	outputs = model(input_tensor)
	print(f"추론 시간: {(time.time() - start_time)*1000:.3f}ms")
	if isinstance(outputs, dict):
		outputs = outputs['out']
	mask = outputs.squeeze().cpu().detach().numpy()
	# mask = np.clip(mask, 0, 1)
	# mask /= np.sum(mask, axis=0, keepdims=True)
	mask = np.exp(mask) / np.sum(np.exp(mask), axis=0, keepdims=True)
	mask_visual = get_mask_visual(image_info, mask)

	# 업샘플링
	mask_visual = cv2.resize(mask_visual, (448, 448), interpolation=cv2.INTER_NEAREST)

	output_image = np.zeros_like(image)
	output_image[y:y + h, x:x + w] = mask_visual
	# image = image[y:y + h, x:x + w]
	return output_image


def play_images_as_video(folder_path, fps=15):
	image_files = sorted(
		[f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
	)

	if not image_files:
		print("폴더에 이미지 파일이 없습니다.")
		return

	total_frames = len(image_files)
	current_frame = 0  # 현재 프레임 인덱스
	delay = int(1000 / fps)  # 밀리초 단위 대기 시간
	step_frames = fps
	stop_flag = False
	tick_flag = False
	break_flag = False
	num_classes = 9
	# model = models.lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
	# model.load_state_dict(torch.load("lraspp_mobilenet_v3_large_best.pth"))
	# model = models.deeplabv3_resnet50(weights=None, num_classes=num_classes)
	# model.load_state_dict(torch.load("deeplabv3_resnet50_best.pth"), strict=False)
	encoder_name = "resnet18"
	model = smp.Unet(
		encoder_name=encoder_name,
		encoder_depth=5,  # 디코더에서 사용하는 단계 수
		encoder_weights="imagenet",  # ImageNet으로 사전 학습된 가중치 사용
		# decoder_segmentation_channels=256,
		# decoder_channels=(32, 16),  # 디코더 채널 수
		classes=num_classes,
		activation=None,
	)
	model.load_state_dict(torch.load(f"weight/u-{encoder_name}.pth"))
	# model.load_state_dict(torch.load(f"weight/Unetresnet18.pth"))
	model.eval()
	model = model.to("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
	image_info = OriginImageInfo().automatic_init("source")

	while True:
		current_time = time.time()

		image_path = os.path.join(folder_path, image_files[current_frame])
		image = cv2.imread(image_path)

		if image is None:
			print(f"이미지를 읽을 수 없습니다: {image_files[current_frame]}")
			current_frame = (current_frame + 1) % total_frames  # 다음 프레임으로 이동
			continue

		if not stop_flag or tick_flag:
			image2 = image_segment(model, image_info, image)
			# 세로로 붙이기
			image = cv2.vconcat([image, image2])

			cv2.imshow("Image Sequence", image)
			print(f"현재 프레임: {current_frame + 1}/{total_frames}")
			tick_flag = False

		# 키 입력 대기
		while True:
			key = cv2.waitKeyEx(1)
			if key == 27:  # ESC 키
				break_flag = True
			elif key & 0xFF == ord('d') or key == 63235:  # 오른쪽 방향키 (앞으로 탐색)
				current_frame += step_frames
				tick_flag = True
			elif key & 0xFF == ord('a') or key == 63234:  # 왼쪽 방향키 (뒤로 탐색)
				current_frame -= step_frames
				tick_flag = True
			elif key & 0xFF == ord('s') or key == 63232:  # 위 방향키 (이전 프레임으로 이동)
				current_frame += 1
				tick_flag = True
			elif key & 0xFF == ord('w') or key == 63233:  # 아래 방향키 (다음 프레임으로 이동)
				current_frame -= 1
				tick_flag = True
			elif key == 32:  # 스페이스바 (일시정지)
				stop_flag = not stop_flag
			if not time.time() - current_time < delay / 1000:
				break
		current_frame += 1 if not stop_flag else 0
		current_frame = current_frame % total_frames

		if break_flag:
			break

	cv2.destroyAllWindows()


# 사용 예시
folder_path = Path(__file__).parent / "path"
play_images_as_video(folder_path)
