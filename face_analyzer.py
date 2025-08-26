import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class FaceAnalyzer:
    """Phân tích khuôn mặt và trích xuất đặc điểm sử dụng OpenCV"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Phát hiện khuôn mặt trong ảnh với nhiều tham số khác nhau"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Cải thiện chất lượng ảnh trước khi phát hiện
            gray = cv2.equalizeHist(gray)
            
            # Thử nhiều tham số khác nhau để tăng khả năng phát hiện
            detection_params = [
                # Tham số mặc định
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                # Tham số nhạy hơn
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
                # Tham số cho khuôn mặt lớn
                {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (50, 50)},
                # Tham số cho khuôn mặt nhỏ
                {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (15, 15)},
                # Tham số rất nhạy
                {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (10, 10)}
            ]
            
            for params in detection_params:
                faces = self.face_cascade.detectMultiScale(gray, **params)
                if len(faces) > 0:
                    self.logger.info(f"Phát hiện {len(faces)} khuôn mặt với tham số: {params}")
                    return [(x, y, w, h) for (x, y, w, h) in faces]
            
            # Nếu vẫn không phát hiện được, thử với ảnh được làm mờ
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            faces = self.face_cascade.detectMultiScale(
                blurred, 
                scaleFactor=1.05, 
                minNeighbors=2, 
                minSize=(10, 10)
            )
            
            if len(faces) > 0:
                self.logger.info(f"Phát hiện {len(faces)} khuôn mặt sau khi làm mờ")
                return [(x, y, w, h) for (x, y, w, h) in faces]
            
            self.logger.warning("Không thể phát hiện khuôn mặt với tất cả tham số")
            return []
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện khuôn mặt: {e}")
            return []

    def extract_facial_features(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Trích xuất đặc điểm khuôn mặt cơ bản"""
        try:
            x, y, w, h = face_coords
            face_roi = image[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện mắt
            eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
            
            # Phát hiện mũi
            nose = self.nose_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
            
            # Phát hiện miệng
            mouth = self.mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
            
            # Tính toán đặc điểm
            features = {
                'face_width': w,
                'face_height': h,
                'face_area': w * h,
                'eye_count': len(eyes),
                'nose_count': len(nose),
                'mouth_count': len(mouth),
                'symmetry_score': self._calculate_symmetry(face_roi),
                'face_ratio': w / h if h > 0 else 1.0
            }
            
            return features
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất đặc điểm: {e}")
            return {}

    def _calculate_symmetry(self, face_roi: np.ndarray) -> float:
        """Tính toán độ đối xứng của khuôn mặt"""
        try:
            if face_roi.size == 0:
                return 0.0
                
            # Chuyển sang grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Lấy nửa trái và phải
            height, width = gray.shape
            mid = width // 2
            
            left_half = gray[:, :mid]
            right_half = gray[:, mid:2*mid] if mid*2 <= width else gray[:, mid:]
            
            # Đảo ngược nửa phải để so sánh
            if right_half.shape[1] > 0:
                right_half_flipped = cv2.flip(right_half, 1)
                
                # Cắt để có cùng kích thước
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
                
                # Tính độ tương đồng
                if left_half.size > 0 and right_half_flipped.size > 0:
                    similarity = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)
                    symmetry_score = np.max(similarity)
                    return float(symmetry_score)
            
            return 0.5  # Giá trị mặc định
        except Exception as e:
            self.logger.error(f"Lỗi khi tính độ đối xứng: {e}")
            return 0.5

    def analyze_skin_texture(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict:
        """Phân tích kết cấu da"""
        try:
            x, y, w, h = face_coords
            face_roi = image[y:y+h, x:x+w]
            
            # Chuyển sang LAB color space để phân tích da
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Tính toán các đặc điểm kết cấu
            # Độ mịn (smoothness) - sử dụng độ lệch chuẩn của gradient
            grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            smoothness = 1.0 / (1.0 + np.std(gradient_magnitude))
            
            # Độ tương phản
            contrast = np.std(l_channel)
            
            # Độ sáng trung bình
            brightness = np.mean(l_channel)
            
            # Phát hiện vùng tối (dark spots)
            dark_threshold = np.percentile(l_channel, 20)
            dark_spots = np.sum(l_channel < dark_threshold) / l_channel.size
            
            # Phát hiện vùng sáng (oily areas)
            bright_threshold = np.percentile(l_channel, 80)
            oily_areas = np.sum(l_channel > bright_threshold) / l_channel.size
            
            texture_features = {
                'smoothness': float(smoothness),
                'contrast': float(contrast),
                'brightness': float(brightness),
                'dark_spots_ratio': float(dark_spots),
                'oily_areas_ratio': float(oily_areas),
                'texture_complexity': float(np.std(gradient_magnitude))
            }
            
            return texture_features
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích kết cấu da: {e}")
            return {}

    def get_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Lấy landmarks khuôn mặt (đơn giản hóa)"""
        try:
            # Sử dụng OpenCV để phát hiện các điểm đặc trưng
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện mắt
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Phát hiện mũi
            nose = self.nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Phát hiện miệng
            mouth = self.mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            landmarks = []
            
            # Thêm tâm của các vùng phát hiện được
            for (x, y, w, h) in eyes:
                landmarks.append([x + w//2, y + h//2])
            
            for (x, y, w, h) in nose:
                landmarks.append([x + w//2, y + h//2])
                
            for (x, y, w, h) in mouth:
                landmarks.append([x + w//2, y + h//2])
            
            return np.array(landmarks) if landmarks else None
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy landmarks: {e}")
            return None

    def analyze_complete_face(self, image: np.ndarray) -> Dict:
        """Phân tích hoàn chỉnh khuôn mặt"""
        try:
            # Phát hiện khuôn mặt
            faces = self.detect_faces(image)
            
            if not faces:
                return {
                    'error': 'Không phát hiện được khuôn mặt trong ảnh',
                    'face_count': 0
                }
            
            # Lấy khuôn mặt đầu tiên
            face_coords = faces[0]
            
            # Phân tích đặc điểm cơ bản
            basic_features = self.extract_facial_features(image, face_coords)
            
            # Phân tích kết cấu da
            texture_features = self.analyze_skin_texture(image, face_coords)
            
            # Lấy landmarks
            landmarks = self.get_face_landmarks(image)
            
            # Tổng hợp kết quả
            analysis_result = {
                'face_count': len(faces),
                'primary_face': {
                    'coordinates': face_coords,
                    'basic_features': basic_features,
                    'texture_features': texture_features,
                    'landmarks_count': len(landmarks) if landmarks is not None else 0
                },
                'all_faces': faces,
                'analysis_timestamp': str(np.datetime64('now'))
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích khuôn mặt: {e}")
            return {
                'error': f'Lỗi khi phân tích: {str(e)}',
                'face_count': 0
            } 