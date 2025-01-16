import cv2
import numpy as np
import torch
import torchvision.models.segmentation as models
from map.create_data import OriginImageInfo

def generate_and_compare_labels(model, data_loader, device, image_info):
	"""
	모델을 활용해 라벨을 생성하고, 실제 라벨과 비교.
	:param model: 학습된 모델
	:param data_loader: 데이터 로더 (DataLoader)
	:param device: GPU 또는 CPU 장치
	:param image_info: OriginImageInfo 객체
	"""
	model.eval()
	colors = np.array(
		[image_info.background_color, image_info.scourge_color, image_info.observer_color] +
		image_info.tile_color_list,
		dtype=np.uint8
	)

	with torch.no_grad():
		for i, (images, true_masks) in enumerate(data_loader):
			images = images.to(device)
			true_masks = true_masks.to(device)

			# 모델 추론
			outputs = model(images)['out']  # 예측 결과 (logits)
			preds = torch.argmax(outputs, dim=1)  # 클래스별로 가장 높은 확률 선택 (N, H, W)

			for j in range(images.size(0)):  # 배치 단위로 처리
				image_np = (images[j].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				true_mask_np = true_masks[j].cpu().numpy()
				pred_mask_np = preds[j].cpu().numpy()

				# 시각화를 위한 마스크 색상 매핑
				true_mask_visual = np.tensordot(true_mask_np.transpose(1, 2, 0), colors, axes=([2], [0])).astype(np.uint8)
				pred_mask_visual = colors[pred_mask_np]

				# 결과 시각화
				concat = np.concatenate((image_np, true_mask_visual, pred_mask_visual), axis=1)
				cv2.imshow(f"Image {i * images.size(0) + j}", concat)

				# 종료 및 대기
				stop_flag = False
				while True:
					key = cv2.waitKey(10)
					if key == 27:
						stop_flag = True
						break
					elif key != -1:
						break
				cv2.destroyAllWindows()
				if stop_flag:
					return


if __name__ == "__main__":
	# 데이터셋 및 데이터 로더 준비
	test_dataset = CustomSegmentationDataset(data_list, mask_list)
	test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

	# 모델 로드
	model = models.lraspp_mobilenet_v3_large(weights=None, num_classes=9)
	model.load_state_dict(torch.load("lraspp_mobilenet_v3_large_best.pth"))
	device = torch.device("mps")
	model = model.to(device)

	# 라벨 생성 및 비교
	image_info = OriginImageInfo().automatic_init("source")
	generate_and_compare_labels(model, test_loader, device, image_info)