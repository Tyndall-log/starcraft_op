import numpy as np
import cv2
import time


def estimate_and_display_C(A, B):
	A = np.array(A, dtype=np.float32)
	B = np.array(B, dtype=np.float32)

	# Create a window for displaying images
	cv2.namedWindow("Restored C", cv2.WINDOW_NORMAL)

	# Search for alpha values between 0 and 1
	for alpha in np.linspace(0.1, 1.0, 10):  # Avoid alpha = 0 to prevent division by zero
		# Calculate C
		C = (B - A * (1 - alpha)) / alpha - A

		# Clip values to valid range (0–255)
		C = np.clip(C, 0, 255).astype(np.uint8)

		# Display the image
		cv2.imshow("Restored C", C)

		# # Wait for a short period to show progression
		# time.sleep(0.1)

		# Break the loop if the user presses the 'q' key
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break

		# 이미지 저장
		cv2.imwrite(f"observer/C_alpha_{alpha:.1f}.png", C)

	# Close the display window
	cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
	# Example input images A and B
	A = cv2.imread("image_source/A.png")
	B = cv2.imread("image_source/B.png")

	# Call the function
	estimate_and_display_C(A, B)
