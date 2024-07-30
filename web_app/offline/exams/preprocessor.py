import cv2


class PreProcessor:
    def __init__(self):
        pass

    # def resize_frame(self, frame, resize_percentage):
    #     if resize_percentage:
    #         width = int(frame.shape[1] * resize_percentage / 100)
    #         height = int(frame.shape[0] * resize_percentage / 100)
    #         dimensions = (width, height)
    #         resized_frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    #         return resized_frame
    #     return frame

    def apply_clahe(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def apply_gaussian_filter(self, frame, kernel_size=(5, 5), sigma=1):
        return cv2.GaussianBlur(frame, kernel_size, sigma)

    def apply_adaptive_threshold(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)

    def resize(self, frame, width, height):
        return cv2.resize(frame, (width, height))

    def apply_edge_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_noise_reduction(self, frame, kernel_size=(5, 5), sigma=1):
        return cv2.GaussianBlur(frame, kernel_size, sigma)

    def apply_histogram_equalization(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def non_maximum_suppression(self, boxes, scores, threshold=0.4):
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=0.5, nms_threshold=threshold
        )
        return indices
