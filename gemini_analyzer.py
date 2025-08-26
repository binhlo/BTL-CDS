import google.generativeai as genai
import streamlit as st
from PIL import Image
import base64
import io
import json
from typing import Dict, List, Optional

class GeminiAnalyzer:
    """Phân tích khuôn mặt sử dụng Google Gemini AI"""
    
    def __init__(self, api_key: str = None):
        """
        Khởi tạo Gemini Analyzer
        
        Args:
            api_key: API key của Google Gemini. Nếu không có, sẽ tìm trong st.secrets
        """
        # Ưu tiên API key truyền vào; nếu không, lấy từ secrets
        self.api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
        self.is_available = False
        self.model = None
        self.model_name = None
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._init_model()
        except Exception:
            # Không làm vỡ UI; sẽ cho phép người dùng nhập API key ở UI
            self.is_available = False

    def set_api_key(self, api_key: str) -> bool:
        """Thiết lập/đổi API key lúc chạy. Trả về True nếu sẵn sàng."""
        try:
            if not api_key:
                raise ValueError("API key trống")
            genai.configure(api_key=api_key)
            self.api_key = api_key
            self._init_model()
            return self.is_available
        except Exception as _:
            self.is_available = False
            return False

    def _init_model(self) -> None:
        """Khởi tạo model khả dụng cho phân tích ảnh.
        Thử theo danh sách model mới nhất có hỗ trợ hình ảnh.
        """
        self.is_available = False
        self.model = None
        self.model_name = None
        model_candidates = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-1.0-pro-vision',
            'gemini-pro-vision'
        ]
        for name in model_candidates:
            try:
                candidate = genai.GenerativeModel(name)
                # Gọi nhẹ để xác thực (đếm token rẻ hơn generate)
                candidate.count_tokens("ping")
                self.model = candidate
                self.model_name = name
                self.is_available = True
                break
            except Exception:
                continue
    
    def analyze_face_with_gemini(self, image: Image.Image, user_concerns: List[str] = None) -> Dict:
        """
        Phân tích khuôn mặt sử dụng Gemini AI
        
        Args:
            image: Ảnh khuôn mặt cần phân tích
            user_concerns: Danh sách vấn đề da người dùng quan tâm
            
        Returns:
            Dict chứa kết quả phân tích từ Gemini
        """
        if not self.is_available:
            return {
                'error': 'Gemini AI không khả dụng. Vui lòng kiểm tra API key.',
                'available': False
            }
        
        try:
            # Tạo prompt chi tiết cho Gemini
            prompt = self._create_analysis_prompt(user_concerns)
            
            # Thêm hint để tăng tính riêng theo ảnh
            meta_hint = "Phân tích riêng cho ảnh này, không sử dụng kết quả trước đó, tập trung mô tả chi tiết thay vì chung chung."
            # Phân tích ảnh với Gemini
            response = self.model.generate_content([prompt + "\n\n" + meta_hint, image])
            
            # Xử lý và parse kết quả
            analysis_result = self._parse_gemini_response(response.text)
            
            return {
                'success': True,
                'gemini_analysis': analysis_result,
                'raw_response': response.text,
                'available': True
            }
            
        except Exception as e:
            error_msg = str(e)
            # Xử lý lỗi quota cụ thể
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                return {
                    'error': 'Đã vượt quá giới hạn sử dụng Gemini API. Vui lòng thử lại sau hoặc kiểm tra gói dịch vụ của bạn.',
                    'error_type': 'quota_exceeded',
                    'available': False,
                    'model_in_use': self.model_name,
                    'suggestion': 'Hãy thử lại sau 1-2 phút hoặc nâng cấp gói dịch vụ Gemini API.'
                }
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                return {
                    'error': 'API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra API key Gemini.',
                    'error_type': 'auth_error',
                    'available': False,
                    'model_in_use': self.model_name,
                    'suggestion': 'Kiểm tra API key trong file .streamlit/secrets.toml'
                }
            else:
                return {
                    'error': f'Lỗi khi phân tích với Gemini: {error_msg}',
                    'error_type': 'general_error',
                    'available': False,
                    'model_in_use': self.model_name
                }
    
    def _create_analysis_prompt(self, user_concerns: List[str] = None) -> str:
        """Tạo prompt chi tiết cho Gemini"""
        
        base_prompt = """
        Bạn là một chuyên gia da liễu và thẩm mỹ có kinh nghiệm. Hãy phân tích khuôn mặt trong ảnh và đưa ra đánh giá chi tiết về:

        1. **Đặc điểm khuôn mặt:**
        - Hình dạng khuôn mặt (tròn, vuông, trái tim, oval)
        - Tỷ lệ khuôn mặt (đối xứng, cân đối)
        - Đặc điểm nổi bật (mắt, mũi, miệng, cằm)

        2. **Phân tích da:**
        - Loại da (khô, dầu, hỗn hợp, nhạy cảm)
        - Tình trạng da (mụn, vết thâm, nếp nhăn, lỗ chân lông)
        - Độ sáng và tông màu da
        - Các vấn đề da cần chú ý

        3. **Đánh giá tổng quan:**
        - Điểm mạnh của khuôn mặt
        - Các vấn đề cần cải thiện
        - Độ tuổi ước tính dựa trên tình trạng da

        4. **Khuyến nghị chăm sóc:**
        - Quy trình skincare phù hợp
        - Sản phẩm nên sử dụng
        - Lời khuyên về lối sống

        Hãy trả lời bằng tiếng Việt, chi tiết và dễ hiểu. Sử dụng emoji để làm cho câu trả lời sinh động.
        """
        
        if user_concerns:
            concerns_text = ", ".join(user_concerns)
            base_prompt += f"\n\n**Lưu ý đặc biệt:** Người dùng quan tâm đến các vấn đề: {concerns_text}. Hãy tập trung phân tích và đưa ra khuyến nghị cụ thể cho những vấn đề này."
        
        return base_prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse và cấu trúc hóa kết quả từ Gemini"""
        
        try:
            # Tách response thành các phần
            sections = {
                'face_features': '',
                'skin_analysis': '',
                'overall_assessment': '',
                'care_recommendations': '',
                'estimated_age': '',
                'skin_type': '',
                'main_concerns': []
            }
            
            # Tìm kiếm các từ khóa để phân loại thông tin
            lines = response_text.split('\n')
            current_section = 'general'
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Phân loại theo từ khóa
                if any(keyword in line.lower() for keyword in ['đặc điểm', 'hình dạng', 'tỷ lệ']):
                    current_section = 'face_features'
                elif any(keyword in line.lower() for keyword in ['da', 'tình trạng', 'mụn', 'vết thâm']):
                    current_section = 'skin_analysis'
                elif any(keyword in line.lower() for keyword in ['đánh giá', 'tổng quan', 'điểm mạnh']):
                    current_section = 'overall_assessment'
                elif any(keyword in line.lower() for keyword in ['khuyến nghị', 'chăm sóc', 'quy trình']):
                    current_section = 'care_recommendations'
                
                # Thêm nội dung vào section tương ứng
                if current_section in sections:
                    if sections[current_section]:
                        sections[current_section] += '\n' + line
                    else:
                        sections[current_section] = line
            
            # Trích xuất thông tin cụ thể
            analysis_result = {
                'face_features': sections['face_features'],
                'skin_analysis': sections['skin_analysis'],
                'overall_assessment': sections['overall_assessment'],
                'care_recommendations': sections['care_recommendations'],
                'raw_response': response_text
            }
            
            # Trích xuất loại da
            skin_keywords = {
                'da khô': ['khô', 'khô ráp', 'thiếu ẩm'],
                'da dầu': ['dầu', 'bóng nhờn', 'tiết dầu'],
                'da hỗn hợp': ['hỗn hợp', 'vùng chữ T', 'khô và dầu'],
                'da nhạy cảm': ['nhạy cảm', 'dễ kích ứng', 'đỏ']
            }
            
            for skin_type, keywords in skin_keywords.items():
                if any(keyword in response_text.lower() for keyword in keywords):
                    analysis_result['skin_type'] = skin_type
                    break
            
            # Trích xuất độ tuổi ước tính
            age_patterns = ['tuổi', 'độ tuổi', 'khoảng', 'ước tính']
            for line in lines:
                if any(pattern in line.lower() for pattern in age_patterns):
                    if '20' in line or '25' in line:
                        analysis_result['estimated_age'] = '20-25'
                    elif '30' in line or '35' in line:
                        analysis_result['estimated_age'] = '30-35'
                    elif '40' in line or '45' in line:
                        analysis_result['estimated_age'] = '40-45'
                    elif '50' in line or '55' in line:
                        analysis_result['estimated_age'] = '50+'
                    break
            
            return analysis_result
            
        except Exception as e:
            # Nếu parse thất bại, trả về response gốc
            return {
                'raw_response': response_text,
                'parse_error': str(e)
            }
    
    def get_skin_care_tips_from_gemini(self, skin_type: str, concerns: List[str] = None) -> List[str]:
        """Lấy lời khuyên chăm sóc da từ Gemini dựa trên loại da và vấn đề cụ thể"""
        
        if not self.is_available:
            return [
                "Gemini AI không khả dụng. Vui lòng kiểm tra API key.",
                "Sử dụng các lời khuyên mặc định từ hệ thống."
            ]
        
        try:
            prompt = f"""
            Bạn là chuyên gia da liễu. Hãy đưa ra 5 lời khuyên cụ thể và thực tế để chăm sóc da {skin_type}.
            
            {f'Người dùng đang gặp vấn đề: {", ".join(concerns)}' if concerns else ''}
            
            Hãy trả lời bằng tiếng Việt, ngắn gọn và dễ thực hiện. Mỗi lời khuyên chỉ 1-2 câu.
            """
            
            response = self.model.generate_content(prompt)
            tips = [tip.strip() for tip in response.text.split('\n') if tip.strip()]
            
            # Giới hạn 5 lời khuyên
            return tips[:5] if len(tips) >= 5 else tips
            
        except Exception as e:
            return [
                f"Không thể lấy lời khuyên từ Gemini: {str(e)}",
                "Sử dụng lời khuyên mặc định từ hệ thống."
            ]
    
    def compare_with_traditional_analysis(self, gemini_result: Dict, traditional_result: Dict) -> Dict:
        """So sánh kết quả phân tích của Gemini với phân tích truyền thống"""
        
        comparison = {
            'skin_type_match': False,
            'confidence_difference': 0,
            'additional_insights': [],
            'recommendations': []
        }
        
        # So sánh loại da
        if isinstance(gemini_result, dict) and 'skin_type' in gemini_result and 'skin_type' in traditional_result:
            gemini_skin = gemini_result.get('skin_type', 'Không xác định')
            traditional_skin = traditional_result.get('skin_type', 'Không xác định')
            
            comparison['skin_type_match'] = gemini_skin == traditional_skin
            
            if not comparison['skin_type_match']:
                comparison['recommendations'].append(
                    f"Phân tích AI truyền thống: {traditional_skin}, Gemini AI: {gemini_skin}. "
                    "Có thể cần xem xét cả hai kết quả để đưa ra quyết định cuối cùng."
                )
        
        # So sánh độ tin cậy
        if 'confidence' in traditional_result:
            traditional_conf = traditional_result['confidence']
            # Gemini không có số liệu confidence cụ thể, nhưng có thể đánh giá chất lượng response
            gemini_quality = len(gemini_result.get('raw_response', '')) / 100  # Đơn giản hóa
            comparison['confidence_difference'] = abs(traditional_conf - gemini_quality)
        
        # Thêm insights từ Gemini
        if isinstance(gemini_result, dict) and 'care_recommendations' in gemini_result:
            care_rec = gemini_result.get('care_recommendations', '')
            if care_rec:
                comparison['additional_insights'].append(
                    f"Gemini AI đưa ra khuyến nghị chăm sóc chi tiết: {care_rec[:100]}..."
                )
        
        return comparison
