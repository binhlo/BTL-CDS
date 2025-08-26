import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

class SkinAnalyzer:
    """Phân tích và phân loại loại da và tình trạng da sử dụng machine learning"""
    
    def __init__(self, model_path: str = "skin_type_model.pkl", cond_model_path: str = "skin_condition_model.pkl"):
        self.model_path = model_path
        self.cond_model_path = cond_model_path
        self.scaler_path = "skin_type_scaler.pkl"
        self.cond_scaler_path = "skin_condition_scaler.pkl"
        self.model = None
        self.cond_model = None
        self.scaler = StandardScaler()
        self.cond_scaler = StandardScaler()
        self.is_trained = False
        self.is_cond_trained = False
        
        # Định nghĩa các loại da
        self.skin_types = {
            0: "Da khô",
            1: "Da dầu", 
            2: "Da hỗn hợp",
            3: "Da nhạy cảm"
        }
        
        # Mô tả đặc điểm từng loại da
        self.skin_descriptions = {
            "Da khô": "Da thiếu độ ẩm, thường có cảm giác căng, bong tróc và dễ xuất hiện nếp nhăn",
            "Da dầu": "Da tiết nhiều bã nhờn, thường bóng và dễ bị mụn trứng cá",
            "Da hỗn hợp": "Da có vùng khô và vùng dầu, thường T-zone (trán, mũi, cằm) dầu hơn",
            "Da nhạy cảm": "Da dễ bị kích ứng, đỏ, ngứa khi tiếp xúc với các thành phần mỹ phẩm"
        }
        
        # Định nghĩa các loại tình trạng da
        self.skin_conditions = {
            0: "Bình thường",
            1: "Mụn",
            2: "Nám/tàn nhang",
            3: "Lỗ chân lông to"
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Tải model nếu có sẵn
        self._load_model()
        self._load_cond_model()
    
    def _load_model(self):
        """Tải model đã được huấn luyện"""
        try:
            model_exists = os.path.exists(self.model_path)
            scaler_exists = os.path.exists(self.scaler_path)
            if model_exists and scaler_exists:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                self.logger.info("Đã tải model và scaler loại da thành công")
            elif model_exists and not scaler_exists:
                # Model có nhưng thiếu scaler -> tránh lỗi StandardScaler chưa fit
                self.logger.warning("Tìm thấy model loại da nhưng thiếu scaler. Sẽ cần huấn luyện lại để tạo scaler.")
                self.is_trained = False
            else:
                self.logger.info("Không tìm thấy model, sẽ huấn luyện model mới")
        except Exception as e:
            self.logger.error(f"Lỗi khi tải model: {e}")
    
    def _load_cond_model(self):
        """Tải model phân loại tình trạng da"""
        try:
            cond_model_exists = os.path.exists(self.cond_model_path)
            cond_scaler_exists = os.path.exists(self.cond_scaler_path)
            if cond_model_exists and cond_scaler_exists:
                self.cond_model = joblib.load(self.cond_model_path)
                self.cond_scaler = joblib.load(self.cond_scaler_path)
                self.is_cond_trained = True
                self.logger.info("Đã tải model và scaler tình trạng da thành công")
            elif cond_model_exists and not cond_scaler_exists:
                self.logger.warning("Tìm thấy model tình trạng da nhưng thiếu scaler. Sẽ cần huấn luyện lại để tạo scaler.")
                self.is_cond_trained = False
            else:
                self.logger.info("Không tìm thấy model tình trạng da, sẽ huấn luyện model mới")
        except Exception as e:
            self.logger.error(f"Lỗi khi tải model tình trạng da: {e}")
    
    def extract_skin_features(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Trích xuất đặc điểm da từ ảnh và chuẩn hóa về thang 0..1 để mô hình phân biệt theo từng ảnh tốt hơn."""
        try:
            x, y, w, h = face_coords
            # Chỉ xử lý vùng khuôn mặt, không toàn bộ ảnh
            face_roi = image[y:y+h, x:x+w]
            
            # Chuyển sang các không gian màu khác nhau
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # Đặc điểm màu sắc (chuẩn hóa)
            # HSV - Hue (0-179), Saturation (0-255), Value (0-255)
            h_mean, s_mean, v_mean = cv2.mean(hsv)[:3]
            h_std, s_std, v_std = np.std(hsv, axis=(0, 1))[:3]
            h_mean = float(h_mean) / 179.0
            s_mean = float(s_mean) / 255.0
            v_mean = float(v_mean) / 255.0
            h_std = float(h_std) / 90.0
            s_std = float(s_std) / 128.0
            v_std = float(v_std) / 128.0
            
            # LAB - Lightness, A, B (0-255)
            l_mean, a_mean, b_mean = cv2.mean(lab)[:3]
            l_std, a_std, b_std = np.std(lab, axis=(0, 1))[:3]
            l_mean = float(l_mean) / 255.0
            a_mean = float(a_mean) / 255.0
            b_mean = float(b_mean) / 255.0
            l_std = float(l_std) / 128.0
            a_std = float(a_std) / 128.0
            b_std = float(b_std) / 128.0
            
            # Đặc điểm kết cấu
            # Độ mịn (smoothness)
            lap_var_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Quy về 0..1 bằng ngưỡng hóa mềm
            laplacian_var = float(min(lap_var_raw / 1000.0, 1.0))
            
            # Độ tương phản
            contrast = float(min(gray.std() / 128.0, 1.0))
            
            # Độ sáng
            brightness = float(gray.mean() / 255.0)
            
            # Phát hiện vết thâm, mụn
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            dark_spots = cv2.countNonZero(thresh)
            total_pixels = gray.size
            dark_spot_ratio = dark_spots / total_pixels
            
            # Phát hiện vùng dầu (bóng)
            # Sử dụng ngưỡng để phát hiện vùng sáng
            _, bright_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            oily_areas = cv2.countNonZero(bright_thresh)
            oily_ratio = oily_areas / total_pixels
            
            # Tính toán gradient để đánh giá kết cấu
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_complexity = float(min(np.mean(gradient_magnitude) / 255.0, 1.0))
            
            # Tổng hợp tất cả đặc điểm
            features = [
                h_mean, s_mean, v_mean, h_std, s_std, v_std,
                l_mean, a_mean, b_mean, l_std, a_std, b_std,
                laplacian_var, contrast, brightness, 
                dark_spot_ratio, oily_ratio, texture_complexity
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất đặc điểm da: {e}")
            return np.zeros(18, dtype=np.float32)
    
    def generate_synthetic_data(self, n_samples: int = 1000):
        """Tạo dữ liệu giả lập cho loại da và tình trạng da"""
        try:
            np.random.seed(42)
            features_list = []
            skin_type_labels = []
            skin_cond_labels = []
            for skin_type in range(4):
                n_type_samples = n_samples // 4
                for skin_cond in range(4):
                    n_cond_samples = n_type_samples // 4
                    for _ in range(n_cond_samples):
                        # Sinh đặc trưng cơ bản như cũ
                        if skin_type == 0:  # Da khô
                            base_features = [0.5, 0.3, 0.4, 0.1, 0.1, 0.1, 0.4, 0.0, 0.0, 0.1, 0.1, 0.1, 0.3, 0.2, 0.3, 0.4, 0.2, 0.3]
                            noise_scale = 0.1
                        elif skin_type == 1:  # Da dầu
                            base_features = [0.5, 0.6, 0.7, 0.1, 0.1, 0.1, 0.6, 0.0, 0.0, 0.1, 0.1, 0.1, 0.7, 0.3, 0.5, 0.1, 0.6, 0.5]
                            noise_scale = 0.1
                        elif skin_type == 2:  # Da hỗn hợp
                            base_features = [0.5, 0.4, 0.5, 0.1, 0.1, 0.1, 0.5, 0.0, 0.0, 0.1, 0.1, 0.1, 0.5, 0.25, 0.4, 0.25, 0.4, 0.4]
                            noise_scale = 0.15
                        else:  # Da nhạy cảm
                            base_features = [0.5, 0.3, 0.4, 0.15, 0.15, 0.15, 0.4, 0.0, 0.0, 0.15, 0.15, 0.15, 0.6, 0.4, 0.3, 0.3, 0.5, 0.5]
                            noise_scale = 0.12
                        # Điều chỉnh đặc trưng theo tình trạng da
                        if skin_cond == 1:  # Mụn
                            base_features[15] += 0.2  # dark_spot_ratio tăng
                        elif skin_cond == 2:  # Nám
                            base_features[13] += 0.2  # contrast tăng
                        elif skin_cond == 3:  # Lỗ chân lông to
                            base_features[17] += 0.2  # texture_complexity tăng
                        noise = np.random.normal(0, noise_scale, len(base_features))
                        features = np.array(base_features) + noise
                        features = np.clip(features, 0, 1)
                        features_list.append(features)
                        skin_type_labels.append(skin_type)
                        skin_cond_labels.append(skin_cond)
            return np.array(features_list), np.array(skin_type_labels), np.array(skin_cond_labels)
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo dữ liệu giả lập: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def train_model(self, features: np.ndarray, labels: np.ndarray):
        """Huấn luyện model phân loại loại da"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Chuẩn hóa dữ liệu
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Tạo và huấn luyện model
            self.model = RandomForestClassifier(
                n_estimators=40,  # giảm số cây
                max_depth=7,      # giảm độ sâu
                random_state=42,
                class_weight='balanced',
                n_jobs=-1        # chạy song song
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Đánh giá model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Độ chính xác model: {accuracy:.3f}")
            self.logger.info(f"Báo cáo phân loại:\n{classification_report(y_test, y_pred)}")
            
            # Lưu model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            self.is_trained = True
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện model: {e}")
            return 0.0

    def train_condition_model(self, features: np.ndarray, cond_labels: np.ndarray):
        """Huấn luyện model phân loại tình trạng da"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, cond_labels, test_size=0.2, random_state=42, stratify=cond_labels)
            X_train_scaled = self.cond_scaler.fit_transform(X_train)
            X_test_scaled = self.cond_scaler.transform(X_test)
            self.cond_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, class_weight='balanced')
            self.cond_model.fit(X_train_scaled, y_train)
            y_pred = self.cond_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            joblib.dump(self.cond_model, self.cond_model_path)
            joblib.dump(self.cond_scaler, self.cond_scaler_path)
            self.is_cond_trained = True
            self.logger.info(f"Huấn luyện model tình trạng da xong, độ chính xác: {acc:.3f}")
            return acc
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện model tình trạng da: {e}")
            return 0.0

    def predict_skin_type(self, features: np.ndarray) -> Dict:
        """Dự đoán loại da"""
        try:
            if not self.is_trained or self.model is None:
                return {'error': 'Model chưa được huấn luyện'}
            # Đảm bảo scaler đã được fit
            if not hasattr(self.scaler, 'scale_'):
                return {'error': "Bộ chuẩn hóa (scaler) chưa được huấn luyện"}
            
            # Chuẩn hóa đặc điểm
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Dự đoán
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            skin_type = self.skin_types[prediction]
            confidence = probabilities[prediction]
            
            # Tạo kết quả chi tiết
            result = {
                'skin_type': skin_type,
                'skin_type_code': int(prediction),
                'confidence': float(confidence),
                'all_probabilities': {
                    self.skin_types[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'description': self.skin_descriptions[skin_type]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi dự đoán loại da: {e}")
            return {'error': str(e)}

    def predict_skin_condition(self, features: np.ndarray) -> Dict:
        """Dự đoán tình trạng da từ đặc trưng"""
        try:
            if not self.is_cond_trained or self.cond_model is None:
                return {"error": "Model tình trạng da chưa được huấn luyện"}
            if not hasattr(self.cond_scaler, 'scale_'):
                return {"error": "Bộ chuẩn hóa (scaler) cho tình trạng da chưa được huấn luyện"}
            features_scaled = self.cond_scaler.transform(features)
            pred = self.cond_model.predict(features_scaled)[0]
            proba = self.cond_model.predict_proba(features_scaled)[0]
            return {
                "skin_condition": self.skin_conditions.get(pred, "Không xác định"),
                "confidence": float(np.max(proba)),
                "all_probabilities": {self.skin_conditions[i]: float(p) for i, p in enumerate(proba)}
            }
        except Exception as e:
            return {"error": str(e)}

    def auto_train(self):
        """Tự động huấn luyện cả model loại da và tình trạng da với dữ liệu giả lập"""
        try:
            self.logger.info("Bắt đầu tạo dữ liệu giả lập...")
            features, labels, cond_labels = self.generate_synthetic_data(1000)
            
            if len(features) == 0:
                return False
            
            self.logger.info("Bắt đầu huấn luyện model loại da...")
            accuracy_type = self.train_model(features, labels)
            
            self.logger.info("Bắt đầu huấn luyện model tình trạng da...")
            accuracy_cond = self.train_condition_model(features, cond_labels)
            
            ok = (accuracy_type > 0.6) and (accuracy_cond > 0.6)
            if ok:
                self.logger.info("Huấn luyện cả hai model thành công!")
            else:
                self.logger.warning("Độ chính xác của một trong các model chưa cao")
            return ok
        except Exception as e:
            self.logger.error(f"Lỗi khi tự động huấn luyện: {e}")
            return False
    
    def get_skin_care_tips(self, skin_type: str) -> List[str]:
        """Lấy lời khuyên chăm sóc da theo loại da"""
        tips = {
            "Da khô": [
                "Sử dụng kem dưỡng ẩm đặc biệt cho da khô",
                "Tránh rửa mặt quá nhiều lần trong ngày",
                "Sử dụng sản phẩm có chứa Hyaluronic Acid",
                "Đắp mặt nạ dưỡng ẩm 2-3 lần/tuần",
                "Tránh sử dụng sản phẩm có cồn"
            ],
            "Da dầu": [
                "Sử dụng sữa rửa mặt dịu nhẹ, không gây khô",
                "Sử dụng kem dưỡng ẩm không gây nhờn",
                "Tránh sử dụng sản phẩm có dầu",
                "Sử dụng sản phẩm có chứa Salicylic Acid",
                "Đắp mặt nạ đất sét 1-2 lần/tuần"
            ],
            "Da hỗn hợp": [
                "Sử dụng sản phẩm phù hợp cho từng vùng da",
                "T-zone: sản phẩm kiểm soát dầu",
                "Vùng má: sản phẩm dưỡng ẩm",
                "Sử dụng kem dưỡng ẩm nhẹ",
                "Đắp mặt nạ theo vùng da"
            ],
            "Da nhạy cảm": [
                "Sử dụng sản phẩm không gây kích ứng",
                "Tránh sản phẩm có hương liệu",
                "Test sản phẩm trước khi sử dụng",
                "Sử dụng kem chống nắng vật lý",
                "Tránh tẩy tế bào chết mạnh"
            ]
        }
        
        return tips.get(skin_type, ["Không có lời khuyên cụ thể cho loại da này"]) 