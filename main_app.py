import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import hashlib
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import binascii

# Import các module đã tạo
from face_analyzer import FaceAnalyzer
from skin_analyzer import SkinAnalyzer
from product_recommender import ProductRecommender
from gemini_analyzer import GeminiAnalyzer

class SkincareAIApp:
    """Ứng dụng chính tích hợp tất cả các module"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Skincare AI - Tư vấn chăm sóc da cá nhân hóa",
            page_icon="🌸",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Khởi tạo các module
        self.face_analyzer = FaceAnalyzer()
        self.skin_analyzer = SkinAnalyzer()
        self.product_recommender = ProductRecommender()
        
        # Khởi tạo Gemini AI
        self.gemini_analyzer = GeminiAnalyzer()
        
        # Khởi tạo session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
        if 'gemini_analysis' not in st.session_state:
            st.session_state.gemini_analysis = None
        if 'current_image_hash' not in st.session_state:
            st.session_state.current_image_hash = None
        if 'auth_user' not in st.session_state:
            st.session_state.auth_user = None
        if 'models_ready' not in st.session_state:
            st.session_state.models_ready = self.skin_analyzer.is_trained and self.skin_analyzer.is_cond_trained
        
        # Không tự huấn luyện ngay để màn đăng nhập nhanh hơn; sẽ huấn luyện sau khi đăng nhập
    
    def main(self):
        """Hàm chính chạy ứng dụng"""
        # Chọn chủ đề màu sắc
        theme_map = {
            "Hồng rực rỡ": "pink",
            "Xanh dương": "blue",
            "Tím": "purple",
            "Xanh mint": "mint",
            "Tối (Dark)": "dark",
            "Công nghệ (Tech)": "tech",
        }
        default_theme = st.session_state.get("ui_theme", "blue")
        theme_label_to_key = {label: key for label, key in theme_map.items()}
        current_label = next((label for label, key in theme_map.items() if key == default_theme), "Xanh dương")
        chosen_label = st.selectbox("Chủ đề màu sắc", list(theme_map.keys()), index=list(theme_map.keys()).index(current_label))
        st.session_state.ui_theme = theme_map[chosen_label]

        # Áp dụng style tùy chỉnh theo chủ đề
        self._apply_custom_styles(st.session_state.ui_theme)
        # Header (tiêu đề công nghệ)
        st.markdown("<h1 class='app-title'>Skincare AI Studio</h1>", unsafe_allow_html=True)
        st.markdown("<div class='app-subtitle'>Phân tích khuôn mặt & tư vấn skincare thông minh</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Đã bỏ phần nhập API key; hệ thống tự lấy từ secrets nếu có

        # Xử lý đăng nhập/đăng ký trước khi vào nội dung chính
        if not self._auth_section():
            return

        # Đảm bảo model đã sẵn sàng sau khi đăng nhập
        if not st.session_state.models_ready:
            with st.spinner("Đang chuẩn bị mô hình phân tích da (lần đầu có thể mất 1-2 phút)..."):
                ok = self.skin_analyzer.auto_train()
                st.session_state.models_ready = bool(ok)
                if not ok:
                    st.warning("Không thể huấn luyện đầy đủ mô hình. Bạn vẫn có thể sử dụng các tính năng khác.")

        # Main content - 3 cột ngang
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            self._face_analysis_section()
        
        with col2:
            self._analysis_results_section()
        
        with col3:
            self._product_recommendations_section()
        
        # Hiển thị kết quả Gemini toàn chiều ngang (nếu có)
        self._gemini_results_section()

        # Tab quy trình skincare ở dưới
        st.markdown("---")
        self._skincare_routine_section()

    # ========== AUTH ==========
    def _get_users_db_path(self) -> str:
        return os.path.join(os.getcwd(), 'users.json')

    def _get_avatars_dir(self) -> str:
        avatars_dir = os.path.join(os.getcwd(), 'assets', 'avatars')
        os.makedirs(avatars_dir, exist_ok=True)
        return avatars_dir

    def _save_avatar(self, username: str, file_bytes: bytes) -> str:
        try:
            avatars_dir = self._get_avatars_dir()
            # Chuẩn hóa và resize về 256x256
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            image = image.resize((256, 256))
            avatar_path = os.path.join(avatars_dir, f"{username}.png")
            image.save(avatar_path, format='PNG')
            return avatar_path
        except Exception:
            return ""

    def _avatar_to_base64(self, path: str) -> str:
        try:
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return ""

    def _generate_default_avatar(self, username: str) -> str:
        """Tạo avatar mặc định theo tên (chứa chữ cái viết tắt)."""
        try:
            initials = (username.strip()[:2] or "U").upper()
            # Màu nền theo hash tên để cố định
            h = int(hashlib.md5(username.encode('utf-8')).hexdigest()[:6], 16)
            r = 100 + (h & 0x3F)       # 100..163
            g = 120 + ((h >> 6) & 0x3F) # 120..183
            b = 140 + ((h >> 12) & 0x3F) # 140..203
            size = 256
            img = Image.new('RGB', (size, size), (r, g, b))
            # Vẽ hình tròn mask để tạo avatar tròn
            mask = Image.new('L', (size, size), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, size, size), fill=255)
            circle = Image.new('RGB', (size, size), (r, g, b))
            circle.putalpha(mask)
            base = Image.new('RGBA', (size, size))
            base.paste(circle, (0, 0))
            # Vẽ chữ cái ở giữa
            draw = ImageDraw.Draw(base)
            try:
                font = ImageFont.truetype("arial.ttf", 110)
            except Exception:
                font = ImageFont.load_default()
            tw, th = draw.textbbox((0, 0), initials, font=font)[2:4] if hasattr(draw, 'textbbox') else draw.textsize(initials, font=font)
            draw.text(((size - tw) / 2, (size - th) / 2), initials, fill=(255, 255, 255), font=font)
            avatars_dir = self._get_avatars_dir()
            avatar_path = os.path.join(avatars_dir, f"{username}.png")
            base.convert('RGB').save(avatar_path, format='PNG')
            return avatar_path
        except Exception:
            return ""

    def _ensure_user_avatar(self, username: str, users: dict) -> str:
        """Đảm bảo người dùng có avatar; nếu thiếu thì tự tạo và lưu đường dẫn."""
        profile = users.get(username, {})
        avatar_path = profile.get('avatar_path', '')
        if not avatar_path or not os.path.exists(avatar_path):
            avatar_path = self._generate_default_avatar(username)
            if username in users:
                users[username]['avatar_path'] = avatar_path
                self._save_users(users)
        return avatar_path

    def _load_users(self) -> dict:
        try:
            db_path = self._get_users_db_path()
            if not os.path.exists(db_path):
                return {}
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_users(self, users: dict) -> bool:
        try:
            db_path = self._get_users_db_path()
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _hash_password(self, password: str, salt: bytes = None) -> tuple:
        if salt is None:
            salt = os.urandom(16)
        # PBKDF2-HMAC-SHA256
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
        return (binascii.hexlify(salt).decode('ascii'), binascii.hexlify(dk).decode('ascii'))

    def _verify_password(self, password: str, salt_hex: str, hash_hex: str) -> bool:
        try:
            salt = binascii.unhexlify(salt_hex.encode('ascii'))
            _, computed_hash_hex = self._hash_password(password, salt)
            return computed_hash_hex == hash_hex
        except Exception:
            return False

    def _auth_section(self) -> bool:
        """Hiển thị giao diện đăng nhập/đăng ký. Trả về True nếu đã đăng nhập."""
        # Nếu đã đăng nhập
        if st.session_state.auth_user:
            with st.sidebar:
                # Hiển thị avatar tròn và chào mừng
                users = self._load_users()
                # Đảm bảo luôn có avatar
                avatar_path = self._ensure_user_avatar(st.session_state.auth_user, users)
                avatar_b64 = self._avatar_to_base64(avatar_path) if avatar_path and os.path.exists(avatar_path) else ""
                if avatar_b64:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:12px;'>"
                        f"<img src='data:image/png;base64,{avatar_b64}' style='width:56px;height:56px;border-radius:50%;border:2px solid rgba(255,255,255,0.6);box-shadow:0 4px 12px rgba(0,0,0,0.15);'/>"
                        f"<div><div style='font-weight:700'>Xin chào, {st.session_state.auth_user}</div>"
                        f"<div style='opacity:.7;font-size:12px'>Chúc bạn một ngày tốt lành ✨</div></div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"Xin chào, **{st.session_state.auth_user}** 👋")

                with st.expander("Ảnh đại diện"):
                    new_avatar = st.file_uploader("Cập nhật avatar", type=["png","jpg","jpeg"], key="upd_avatar")
                    if new_avatar is not None:
                        saved_path = self._save_avatar(st.session_state.auth_user, new_avatar.getvalue())
                        if saved_path:
                            users[st.session_state.auth_user]['avatar_path'] = saved_path
                            self._save_users(users)
                            st.success("Đã cập nhật ảnh đại diện!")
                            st.rerun()
                if st.button("Đăng xuất"):
                    st.session_state.auth_user = None
                    # Xóa dữ liệu phiên liên quan ảnh để tránh rò rỉ giữa người dùng
                    st.session_state.analysis_results = None
                    st.session_state.recommendations = None
                    st.session_state.gemini_analysis = None
                    st.session_state.current_image_hash = None
                    st.rerun()
            return True

        st.markdown("""
        <div style='display:flex;justify-content:center;'>
          <div style='max-width:840px;width:100%;'>
            <h3 style='margin:0 0 8px 0;'>🔐 Đăng nhập để sử dụng Skincare AI</h3>
          </div>
        </div>
        """, unsafe_allow_html=True)
        auth_tabs = st.tabs(["Đăng nhập", "Đăng ký"])

        # Đăng nhập
        with auth_tabs[0]:
            login_col1, login_col2 = st.columns([1, 1])
            with login_col1:
                st.markdown("<div style='padding:16px;border:1px solid rgba(0,0,0,.08);border-radius:12px;background:rgba(255,255,255,.65);backdrop-filter:saturate(130%) blur(2px);box-shadow:0 8px 24px rgba(0,0,0,.08);'>", unsafe_allow_html=True)
                username = st.text_input("Tên đăng nhập")
                password = st.text_input("Mật khẩu", type='password')
                if st.button("Đăng nhập", type="primary"):
                    users = self._load_users()
                    record = users.get(username)
                    if record and self._verify_password(password, record.get('salt', ''), record.get('hash', '')):
                        st.session_state.auth_user = username
                        st.success("Đăng nhập thành công!")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
                st.markdown("</div>", unsafe_allow_html=True)
            with login_col2:
                st.info("Nếu chưa có tài khoản, hãy chuyển sang tab Đăng ký.")

        # Đăng ký
        with auth_tabs[1]:
            reg_col1, reg_col2 = st.columns([1, 1])
            with reg_col1:
                st.markdown("<div style='padding:16px;border:1px solid rgba(0,0,0,.08);border-radius:12px;background:rgba(255,255,255,.65);backdrop-filter:saturate(130%) blur(2px);box-shadow:0 8px 24px rgba(0,0,0,.08);'>", unsafe_allow_html=True)
                new_username = st.text_input("Tên đăng nhập mới")
                new_password = st.text_input("Mật khẩu mới", type='password')
                confirm_password = st.text_input("Xác nhận mật khẩu", type='password')
                avatar_file = st.file_uploader("Ảnh đại diện (tùy chọn)", type=["png","jpg","jpeg"], key="reg_avatar")
                if st.button("Tạo tài khoản"):
                    if not new_username or not new_password:
                        st.warning("Vui lòng nhập đầy đủ thông tin.")
                    elif len(new_username) < 3:
                        st.warning("Tên đăng nhập phải có ít nhất 3 ký tự.")
                    elif len(new_password) < 6:
                        st.warning("Mật khẩu phải có ít nhất 6 ký tự.")
                    elif new_password != confirm_password:
                        st.warning("Mật khẩu xác nhận không khớp.")
                    else:
                        users = self._load_users()
                        if new_username in users:
                            st.error("Tên đăng nhập đã tồn tại.")
                        else:
                            salt_hex, hash_hex = self._hash_password(new_password)
                            avatar_path = ""
                            if avatar_file is not None:
                                avatar_path = self._save_avatar(new_username, avatar_file.getvalue())
                            users[new_username] = {
                                'salt': salt_hex,
                                'hash': hash_hex,
                                'created_at': datetime.now().isoformat(),
                                'avatar_path': avatar_path or self._generate_default_avatar(new_username)
                            }
                            if self._save_users(users):
                                st.success("Đăng ký thành công! Hãy đăng nhập để tiếp tục.")
                            else:
                                st.error("Không thể lưu tài khoản. Vui lòng thử lại.")
                st.markdown("</div>", unsafe_allow_html=True)
            with reg_col2:
                st.info("Mẹo: Sử dụng mật khẩu mạnh gồm chữ hoa, chữ thường, số và ký tự đặc biệt.")

        return False

    def _apply_custom_styles(self, theme: str = "pink"):
        """Thêm CSS theo chủ đề màu."""
        css = self._get_theme_css(theme)
        st.markdown(css, unsafe_allow_html=True)

    def _get_theme_css(self, theme: str) -> str:
        if theme == "tech":
            primary = "#00E5FF"; accent1 = "#00C2FF"; accent2 = "#00FFA8"; bg1 = "#0A0F1E"; bg2 = "#0E1528"; text = "#E6F7FF"
        elif theme == "blue":
            primary = "#0052CC"; accent1 = "#228BE6"; accent2 = "#66B2FF"; bg1 = "#E7F1FF"; bg2 = "#F4F9FF"; text = "#003B8E"
        elif theme == "purple":
            primary = "#6A00FF"; accent1 = "#9B5CF6"; accent2 = "#C7A0FF"; bg1 = "#F3E9FF"; bg2 = "#FBF7FF"; text = "#4B00B5"
        elif theme == "mint":
            primary = "#00C2A8"; accent1 = "#23D5AB"; accent2 = "#7CF8E1"; bg1 = "#E6FFFB"; bg2 = "#F6FFFE"; text = "#00796B"
        elif theme == "dark":
            primary = "#7C4DFF"; accent1 = "#00E5FF"; accent2 = "#80D8FF"; bg1 = "#0E0F13"; bg2 = "#161821"; text = "#E6E6E6"
        else:  # pink
            primary = "#E00070"; accent1 = "#FF6FB5"; accent2 = "#FF9AD3"; bg1 = "#FFE3F2"; bg2 = "#FFF1FF"; text = "#C6006E"

        return f"""
        <style>
        html, body, [data-testid='stAppViewContainer'] {{
            background: radial-gradient(60% 60% at 50% 0%, {bg2} 0%, {bg1} 100%);
        }}
        h1, h2, h3, h4 {{ color: {primary}; }}
        .app-title {{
            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
            font-weight: 800; letter-spacing: 0.5px; margin: 0;
            background: linear-gradient(90deg, {primary}, {accent2});
            -webkit-background-clip: text; background-clip: text; color: transparent;
            text-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
        }}
        .app-subtitle {{
            color: {text}; opacity: 0.9; margin-top: -6px; margin-bottom: 6px;
            font-size: 16px;
        }}
        div.stButton > button {{
            background: linear-gradient(135deg, {primary} 0%, {accent1} 50%, {accent2} 100%);
            color: #FFFFFF; border: 0; border-radius: 14px;
            padding: 0.65rem 1.05rem; font-weight: 700; letter-spacing: 0.2px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            transition: transform .05s ease-in, filter .1s ease-in;
        }}
        div.stButton > button:hover {{ filter: brightness(1.08); transform: translateY(-1px); }}
        [data-testid='stFileUploaderDropzone'] {{
            background-color: {bg1}; border: 2px dashed {accent1}; border-radius: 14px;
        }}
        [data-testid='stExpander'] > details {{
            background: {bg1}; border: 1px solid {accent2}; border-radius: 12px;
        }}
        .streamlit-expanderHeader {{ font-weight: 800; color: {text}; }}
        [data-testid='stMetricValue'] {{ color: {primary}; }}
        [data-testid='stMetricLabel'] {{ color: #5B5560; }}
        [data-testid='stPlotlyChart'] {{ background: #FFFFFF; border-radius: 12px; padding: 10px; }}
        /* Card effect for columns */
        section.main > div.block-container {{
            backdrop-filter: saturate(130%) blur(2px);
        }}
        /* Fix metric truncation - custom metric text */
        .metric-text {{
            font-size: 2rem; font-weight: 700; color: {primary};
            white-space: nowrap; overflow: visible; text-overflow: clip;
        }}
        </style>
        """
    
    def _face_analysis_section(self):
        """Phần phân tích khuôn mặt"""
        st.subheader("📸 Phân tích khuôn mặt")

        # Cho phép người dùng chọn giữa tải ảnh lên hoặc chụp ảnh bằng webcam
        tab_upload, tab_camera = st.tabs(["Tải ảnh", "Chụp ảnh"])
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Chọn ảnh khuôn mặt (JPG, PNG):",
                type=["jpg", "jpeg", "png"],
                help="Chọn ảnh có khuôn mặt rõ ràng, ánh sáng tốt",
            )
        with tab_camera:
            camera_file = st.camera_input(
                "Chụp ảnh từ webcam",
                help="Hãy đảm bảo khuôn mặt chiếm ít nhất 50% khung hình và ánh sáng đều"
            )

        image_file = camera_file or uploaded_file
        if image_file is None:
            return

        # Tính hash ảnh để phát hiện khi người dùng tải ảnh khác
        try:
            file_bytes = image_file.getvalue()
            image_hash = hashlib.md5(file_bytes).hexdigest()
        except Exception:
            image_hash = None

        # Nếu ảnh mới khác ảnh trước đó, reset toàn bộ kết quả cũ
        if image_hash and st.session_state.get("current_image_hash") != image_hash:
            st.session_state.current_image_hash = image_hash
            st.session_state.analysis_results = None
            st.session_state.gemini_analysis = None
            st.session_state.recommendations = None

        # Hiển thị ảnh gốc
        image = Image.open(image_file)
        st.image(image, caption="Ảnh gốc", use_container_width=True)

        # Chuyển đổi sang OpenCV format
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Phân tích AI truyền thống", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích với AI truyền thống..."):
                    face_analysis = self.face_analyzer.analyze_complete_face(image_array)
                    if "error" not in face_analysis and face_analysis["face_count"] > 0:
                        face_coords = face_analysis["primary_face"]["coordinates"]
                        skin_features = self.skin_analyzer.extract_skin_features(image_array, tuple(face_coords))
                        skin_prediction = self.skin_analyzer.predict_skin_type(skin_features.reshape(1, -1))
                        skin_condition_pred = self.skin_analyzer.predict_skin_condition(skin_features.reshape(1, -1))
                        st.session_state.analysis_results = {
                            "face_analysis": face_analysis,
                            "skin_features": skin_features.tolist(),
                            "skin_prediction": skin_prediction,
                            "skin_condition_prediction": skin_condition_pred,
                            "image": image_array,
                            "image_hash": image_hash,
                        }
                        st.session_state.recommendations = None
                        st.success("Phân tích AI truyền thống hoàn tất!")
                    else:
                        st.error("Không thể phát hiện khuôn mặt trong ảnh. Vui lòng thử ảnh khác.")

        with col2:
            if st.button("🤖 Phân tích với Gemini AI", type="secondary", use_container_width=True):
                if self.gemini_analyzer.is_available:
                    with st.spinner("Đang phân tích với Gemini AI..."):
                        gemini_result = self.gemini_analyzer.analyze_face_with_gemini(image)
                        if isinstance(gemini_result, dict) and gemini_result.get("success", False):
                            st.session_state.gemini_analysis = {**gemini_result, "image_hash": image_hash}
                            st.success("Phân tích Gemini AI hoàn tất!")
                        else:
                            error_msg = (
                                gemini_result.get("error", "Lỗi không xác định")
                                if isinstance(gemini_result, dict)
                                else str(gemini_result)
                            )
                            error_type = gemini_result.get("error_type", "general_error") if isinstance(gemini_result, dict) else "general_error"
                            suggestion = gemini_result.get("suggestion", "") if isinstance(gemini_result, dict) else ""
                            
                            if error_type == "quota_exceeded":
                                st.error(f"⚠️ {error_msg}")
                                if suggestion:
                                    st.info(f"💡 **Gợi ý:** {suggestion}")
                                st.info("🔗 **Tham khảo:** https://ai.google.dev/gemini-api/docs/rate-limits")
                            elif error_type == "auth_error":
                                st.error(f"🔑 {error_msg}")
                                if suggestion:
                                    st.info(f"💡 **Gợi ý:** {suggestion}")
                            else:
                                st.error(f"❌ Lỗi phân tích Gemini: {error_msg}")
                else:
                    st.error("Gemini AI không khả dụng. Vui lòng kiểm tra API key.")

        if st.button("🚀 Phân tích kết hợp (AI + Gemini)", type="primary", use_container_width=True):
            with st.spinner("Đang phân tích kết hợp..."):
                face_analysis = self.face_analyzer.analyze_complete_face(image_array)
                if "error" not in face_analysis and face_analysis["face_count"] > 0:
                    face_coords = face_analysis["primary_face"]["coordinates"]
                    skin_features = self.skin_analyzer.extract_skin_features(image_array, tuple(face_coords))
                    skin_prediction = self.skin_analyzer.predict_skin_type(skin_features.reshape(1, -1))
                    skin_condition_pred = self.skin_analyzer.predict_skin_condition(skin_features.reshape(1, -1))
                    gemini_result = self.gemini_analyzer.analyze_face_with_gemini(image)
                    st.session_state.analysis_results = {
                        "face_analysis": face_analysis,
                        "skin_features": skin_features.tolist(),
                        "skin_prediction": skin_prediction,
                        "skin_condition_prediction": skin_condition_pred,
                        "image": image_array,
                        "image_hash": image_hash,
                    }
                    if isinstance(gemini_result, dict) and gemini_result.get("success", False):
                        st.session_state.gemini_analysis = {**gemini_result, "image_hash": image_hash}
                    st.session_state.recommendations = None
                    st.success("Phân tích kết hợp hoàn tất!")
                else:
                    st.error("Không thể phát hiện khuôn mặt trong ảnh. Vui lòng thử lại với ảnh khác.")

        with st.expander("📋 Hướng dẫn chụp ảnh"):
            st.markdown(
                """
                **Để có kết quả phân tích chính xác nhất:**
                ✅ **Nên làm:**
                - Chụp ảnh khuôn mặt rõ ràng
                - Ánh sáng đều, không quá tối hoặc quá sáng
                - Khuôn mặt chiếm ít nhất 50% ảnh
                - Chụp thẳng mặt, không nghiêng quá nhiều
                ❌ **Không nên:**
                - Ảnh mờ, không rõ nét
                - Ánh sáng quá mạnh tạo bóng
                - Khuôn mặt bị che khuất
                - Ảnh có nhiều người
                """
            )
    
    def _analysis_results_section(self):
        """Phần hiển thị kết quả phân tích"""
        st.subheader("🔍 Kết quả phân tích")
        
        if st.session_state.analysis_results is None:
            st.info("Hãy tải ảnh và phân tích ở cột bên trái để xem kết quả.")
            return
        
        results = st.session_state.analysis_results
        current_hash = st.session_state.get('current_image_hash')
        # Nếu kết quả đang hiển thị không khớp hash ảnh hiện tại thì không hiển thị để tránh lặp
        if results and results.get('image_hash') and current_hash and results.get('image_hash') != current_hash:
            st.info("Kết quả cũ không khớp ảnh hiện tại. Vui lòng phân tích lại.")
            return
        
        # Thông tin cơ bản
        face_info = results['face_analysis']
        st.metric("Số khuôn mặt phát hiện", face_info['face_count'])
        
        # Kiểm tra và hiển thị điểm đối xứng
        if 'primary_face' in face_info and 'basic_features' in face_info['primary_face']:
            basic_features = face_info['primary_face']['basic_features']
            if 'symmetry_score' in basic_features:
                st.metric("Điểm đối xứng", f"{basic_features['symmetry_score']:.3f}")
            
            # Hiển thị thêm thông tin chi tiết
            if 'face_area' in basic_features:
                st.metric("Diện tích khuôn mặt", f"{basic_features['face_area']:,} px²")
            if 'eye_count' in basic_features:
                st.metric("Số mắt phát hiện", basic_features['eye_count'])
        else:
            st.metric("Điểm đối xứng", "Không có dữ liệu")

        # Thông tin loại da
        skin_pred = results['skin_prediction']
        if 'error' not in skin_pred:
            st.success(f"**Loại da:** {skin_pred['skin_type']}")
            st.metric("Độ tin cậy", f"{skin_pred['confidence']:.1%}")
            # Hiển thị mô tả loại da
            st.info(f"**Mô tả:** {skin_pred['description']}")
        else:
            st.error(f"Lỗi dự đoán: {skin_pred['error']}")

        # Thông tin tình trạng da
        skin_cond_pred = results.get('skin_condition_prediction', None)
        if skin_cond_pred and 'error' not in skin_cond_pred:
            st.success(f"**Tình trạng da:** {skin_cond_pred['skin_condition']}")
            st.metric("Độ tin cậy tình trạng", f"{skin_cond_pred['confidence']:.1%}")
        elif skin_cond_pred and 'error' in skin_cond_pred:
            st.error(f"Lỗi dự đoán tình trạng da: {skin_cond_pred['error']}")
        
        # Tóm tắt nhanh từ Gemini (nếu có) ngay trong phần kết quả
        if st.session_state.get('gemini_analysis'):
            g = st.session_state.gemini_analysis
            g_hash = g.get('image_hash')
            if not g_hash or g_hash == st.session_state.get('current_image_hash'):
                g_data = g.get('gemini_analysis', {})
                g_skin = g_data.get('skin_type', 'Không xác định')
                g_age = g_data.get('estimated_age', '')
                with st.expander("🤖 Tóm tắt từ Gemini AI"):
                    st.markdown(f"- **Loại da (Gemini)**: {g_skin}")
                    if g_age:
                        st.markdown(f"- **Độ tuổi ước tính**: {g_age}")
                    if g_data.get('overall_assessment'):
                        st.markdown("- **Đánh giá tổng quan:**")
                        st.markdown(g_data.get('overall_assessment', ''))

        # Hiển thị ảnh đã phân tích
        if 'image' in results and 'face_analysis' in results and 'primary_face' in results['face_analysis']:
            try:
                # Vẽ khung khuôn mặt
                image_with_face = results['image'].copy()
                face_coords = results['face_analysis']['primary_face']['coordinates']
                x, y, w, h = face_coords
                cv2.rectangle(image_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Chuyển BGR sang RGB để hiển thị
                image_rgb = cv2.cvtColor(image_with_face, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Khuôn mặt được phát hiện", use_container_width=True)
                
                # Hiển thị thông tin tọa độ
                st.info(f"**Vị trí:** X={x}, Y={y}, Rộng={w}, Cao={h}")
            except Exception as e:
                st.error(f"Lỗi khi hiển thị ảnh: {str(e)}")
        
        # Không hiển thị Gemini ở giữa nữa; phần này đã chuyển sang full-width
    
    def _product_recommendations_section(self):
        """Phần tư vấn sản phẩm"""
        st.subheader("💡 Tư vấn sản phẩm")
        
        if st.session_state.analysis_results is None:
            st.info("Hãy phân tích khuôn mặt trước để nhận tư vấn sản phẩm.")
            return
        
        skin_pred = st.session_state.analysis_results['skin_prediction']
        if 'error' in skin_pred:
            st.error("Không thể tư vấn sản phẩm do lỗi phân tích da.")
            return
        
        # Thông tin cơ bản cho tư vấn
        st.info(f"**Loại da:** {skin_pred['skin_type']}")

        # Tự động tạo khuyến nghị sau khi có kết quả phân tích
        if st.session_state.recommendations is None:
            with st.spinner("Đang tạo khuyến nghị dựa trên kết quả phân tích..."):
                # Kết hợp mối quan tâm từ Gemini nếu có
                concerns = ["da khô"]
                if st.session_state.get('gemini_analysis'):
                    g = st.session_state.gemini_analysis
                    if (not g.get('image_hash')) or g.get('image_hash') == st.session_state.get('current_image_hash'):
                        g_text = (g.get('gemini_analysis', {}) or {}).get('skin_analysis', '')
                        g_text_lower = g_text.lower()
                        concerns = []
                        if any(k in g_text_lower for k in ["mụn", "acne"]):
                            concerns.append("mụn")
                        if any(k in g_text_lower for k in ["vết thâm", "thâm", "nám", "tàn nhang", "dark spot"]):
                            concerns.append("vết thâm")
                        if "lỗ chân lông" in g_text_lower:
                            concerns.append("lỗ chân lông to")
                        if any(k in g_text_lower for k in ["dầu", "bóng nhờn"]):
                            concerns.append("da dầu")
                        if any(k in g_text_lower for k in ["khô", "thiếu ẩm"]):
                            concerns.append("da khô")
                        if not concerns:
                            concerns = ["da khô"]
                st.session_state.recommendations = self.product_recommender.get_product_recommendations(
                    skin_type=skin_pred['skin_type'],
                    skin_condition=None,
                    skin_concerns=concerns,  # kết hợp concerns từ Gemini nếu có
                    age_group="26-35",
                    budget_level="trung bình",
                    products_per_category=2
                )
                st.success("Đã tạo khuyến nghị sản phẩm cho ảnh hiện tại.")
        
        # Hiển thị khuyến nghị
        if st.session_state.recommendations:
            self._display_recommendations_compact(st.session_state.recommendations)
            # Hiển thị thêm khuyến nghị chăm sóc từ Gemini nếu có
            if st.session_state.get('gemini_analysis'):
                g = st.session_state.gemini_analysis
                if (not g.get('image_hash')) or g.get('image_hash') == st.session_state.get('current_image_hash'):
                    g_data = g.get('gemini_analysis', {})
                    care = g_data.get('care_recommendations', '')
                    if care:
                        with st.expander("🤖 Khuyến nghị chăm sóc từ Gemini"):
                            self._render_text_columns(care, num_cols=2)
    
    def _display_recommendations_compact(self, recommendations):
        """Hiển thị khuyến nghị sản phẩm dạng compact"""
        st.subheader("🎁 Sản phẩm được khuyến nghị")
        
        # Thông tin tổng quan
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Loại da**")
            st.markdown(f"<div class='metric-text'>{recommendations['skin_type']}</div>", unsafe_allow_html=True)
        with col2:
            st.metric("Số sản phẩm", len(recommendations['recommended_products']))
        
        # Giải thích
        st.info(f"**Giải thích:** {recommendations['explanation']}")
        
        # Hiển thị từng sản phẩm dạng compact
        for i, product_info in enumerate(recommendations['recommended_products']):
            category = product_info['category']
            product = product_info['product']
            
            with st.expander(f"{i+1}. {self._get_category_name(category)} - {product['name']}"):
                st.markdown(f"**Thương hiệu:** {product['brand']}")
                st.markdown(f"**Giá:** {product['price']}")
                st.markdown(f"**Đánh giá:** {'⭐' * int(product['rating'])} ({product['rating']})")
                
                # Thành phần chính
                st.markdown("**Thành phần chính:**")
                for ingredient in product['ingredients'][:3]:  # Chỉ hiển thị 3 thành phần đầu
                    st.markdown(f"- {ingredient}")
                
                # Lợi ích chính
                st.markdown("**Lợi ích:**")
                for benefit in product['benefits'][:2]:  # Chỉ hiển thị 2 lợi ích đầu
                    st.markdown(f"- {benefit}")
        
        # Nút lưu khuyến nghị
        if st.button("💾 Lưu khuyến nghị", use_container_width=True):
            filename = f"skincare_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if self.product_recommender.save_recommendations(recommendations, filename):
                st.success(f"Đã lưu khuyến nghị vào {filename}")
            else:
                st.error("Lỗi khi lưu khuyến nghị")
    
    def _skincare_routine_section(self):
        """Phần quy trình skincare"""
        st.header("📋 Quy trình skincare")
        
        if not st.session_state.recommendations:
            st.info("Hãy tạo khuyến nghị sản phẩm trước để xem quy trình skincare.")
            return
        
        routine = st.session_state.recommendations['skincare_routine']
        
        st.subheader("🔄 Quy trình chăm sóc da hàng ngày")
        
        # Hiển thị quy trình
        for step in routine:
            with st.expander(f"Bước {step['step']}: {self._get_category_name(step['category'])}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Sản phẩm:** {step['product_name']}")
                    st.markdown(f"**Thương hiệu:** {step['brand']}")
                    st.markdown(f"**Hướng dẫn sử dụng:** {step['usage']}")
                
                with col2:
                    # Icon cho từng bước
                    icons = {
                        "cleanser": "🧼",
                        "serum": "💧",
                        "moisturizer": "🧴",
                        "sunscreen": "☀️"
                    }
                    icon = icons.get(step['category'], "📦")
                    st.markdown(f"<h1 style='text-align: center;'>{icon}</h1>", unsafe_allow_html=True)
        
        # Lời khuyên chăm sóc da
        st.subheader("💡 Lời khuyên chăm sóc da")
        
        skin_type = st.session_state.recommendations['skin_type']
        tips = self.skin_analyzer.get_skin_care_tips(skin_type)
        
        for i, tip in enumerate(tips, 1):
            st.markdown(f"{i}. {tip}")
        
        # Biểu đồ thời gian sử dụng
        st.subheader("⏰ Thời gian sử dụng sản phẩm")
        
        # Tạo dữ liệu thời gian
        time_data = {
            "Sáng": ["Rửa mặt", "Serum", "Dưỡng ẩm", "Chống nắng"],
            "Tối": ["Rửa mặt", "Serum", "Dưỡng ẩm"]
        }
        
        fig = go.Figure()
        
        for time, steps in time_data.items():
            fig.add_trace(go.Bar(
                name=time,
                x=steps,
                y=[1] * len(steps),
                text=steps,
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Lịch trình sử dụng sản phẩm",
            xaxis_title="Các bước",
            yaxis_title="Tần suất",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _gemini_results_section(self):
        """Hiển thị kết quả phân tích từ Gemini AI với nội dung tư vấn chuyên sâu"""
        if not st.session_state.get('gemini_analysis'):
            return
        current_hash = st.session_state.get('current_image_hash')
        if st.session_state.gemini_analysis.get('image_hash') and current_hash and st.session_state.gemini_analysis.get('image_hash') != current_hash:
            return
        
        st.markdown("---")
        st.markdown("## 🤖 Tư vấn chuyên gia từ Gemini AI")
        st.markdown("*Phân tích chuyên sâu và lời khuyên cá nhân hóa từ AI thế hệ mới*")

        gemini_result = st.session_state.gemini_analysis.get('gemini_analysis', {})

        # Tạo layout tư vấn chuyên nghiệp
        consulting_col1, consulting_col2 = st.columns([1, 1])
        
        with consulting_col1:
            # Phân tích chuyên sâu về da
            with st.container():
                st.markdown("### 🔬 Chẩn đoán chuyên sâu")
                if isinstance(gemini_result, dict) and gemini_result.get('skin_analysis'):
                    st.markdown("**Tình trạng da hiện tại:**")
                    self._render_consulting_content(gemini_result.get('skin_analysis', ''))
                    
                if isinstance(gemini_result, dict) and gemini_result.get('overall_assessment'):
                    st.markdown("**Đánh giá tổng quan:**")
                    self._render_consulting_content(gemini_result.get('overall_assessment', ''))

            # Kế hoạch chăm sóc cá nhân hóa
            with st.container():
                st.markdown("### 📋 Kế hoạch chăm sóc cá nhân")
                if isinstance(gemini_result, dict) and gemini_result.get('care_recommendations'):
                    self._render_consulting_content(gemini_result.get('care_recommendations', ''))
                    
                # Thêm lời khuyên chuyên gia
                self._render_expert_advice(gemini_result)

        with consulting_col2:
            # Phân tích đặc điểm và tư vấn làm đẹp
            with st.container():
                st.markdown("### 💄 Tư vấn làm đẹp")
                if isinstance(gemini_result, dict) and gemini_result.get('face_features'):
                    st.markdown("**Phân tích đặc điểm:**")
                    self._render_consulting_content(gemini_result.get('face_features', ''))
                
                # Tư vấn makeup và styling
                self._render_beauty_consulting(gemini_result)

            # Theo dõi tiến trình và lời khuyên dài hạn
            with st.container():
                st.markdown("### 📈 Theo dõi & Lời khuyên dài hạn")
                self._render_longterm_consulting(gemini_result)

        # So sánh với phân tích truyền thống (toàn chiều ngang)
        if st.session_state.analysis_results:
            st.markdown("### 🔍 So sánh & Xác thực kết quả")
            comparison = self.gemini_analyzer.compare_with_traditional_analysis(
                gemini_result,
                st.session_state.analysis_results['skin_prediction']
            )
            
            comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1])
            
            with comp_col1:
                if comparison['skin_type_match']:
                    st.success("✅ **Kết quả nhất quán**\nCả AI truyền thống và Gemini đều đưa ra cùng kết luận về loại da")
                else:
                    st.warning("⚠️ **Kết quả khác biệt**\nCần xem xét thêm để đưa ra kết luận chính xác")
            
            with comp_col2:
                st.markdown("**Khuyến nghị từ so sánh:**")
                for rec in comparison.get('recommendations', []):
                    st.info(f"💡 {rec}")
            
            with comp_col3:
                st.markdown("**Thông tin bổ sung:**")
                for insight in comparison.get('additional_insights', []):
                    st.info(f"🔍 {insight}")
        
        # Thêm phần tư vấn cá nhân hóa
        self._render_personalized_consultation()

    def _render_consulting_content(self, content: str):
        """Hiển thị nội dung tư vấn với định dạng chuyên nghiệp"""
        if not content:
            return
        
        # Xử lý và format nội dung để dễ đọc hơn
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Thêm icon cho các điểm quan trọng
            if line.startswith('-') or line.startswith('•'):
                line = f"🔸 {line[1:].strip()}"
            elif any(keyword in line.lower() for keyword in ['khuyến nghị', 'nên', 'should']):
                line = f"💡 {line}"
            elif any(keyword in line.lower() for keyword in ['cảnh báo', 'tránh', 'không nên']):
                line = f"⚠️ {line}"
            formatted_lines.append(line)
        
        st.markdown('\n\n'.join(formatted_lines))

    def _render_expert_advice(self, gemini_result: dict):
        """Hiển thị lời khuyên chuyên gia bổ sung"""
        st.markdown("**🩺 Lời khuyên chuyên gia:**")
        
        # Tạo lời khuyên dựa trên kết quả phân tích
        advice_points = [
            "💧 **Hydration**: Duy trì độ ẩm cho da bằng cách uống đủ nước (8-10 ly/ngày)",
            "🌙 **Sleep**: Ngủ đủ 7-8 tiếng để da tự phục hồi và tái tạo",
            "🥗 **Nutrition**: Bổ sung vitamin C, E và omega-3 cho da khỏe mạnh",
            "☀️ **Sun Protection**: Sử dụng kem chống nắng SPF 30+ hàng ngày",
            "🧘 **Stress Management**: Quản lý stress để giảm tác động xấu lên da"
        ]
        
        for advice in advice_points:
            st.markdown(advice)

    def _render_beauty_consulting(self, gemini_result: dict):
        """Tư vấn làm đẹp và makeup phù hợp"""
        st.markdown("**✨ Tư vấn makeup & styling:**")
        
        beauty_tips = [
            "🎨 **Foundation**: Chọn tone nền phù hợp với undertone da",
            "👁️ **Eye makeup**: Tôn lên đặc điểm mắt với màu sắc hài hòa",
            "💋 **Lip color**: Chọn màu môi cân bằng với tông màu da",
            "✨ **Highlight**: Sử dụng highlighter để tạo điểm nhấn tự nhiên",
            "🌈 **Color harmony**: Phối màu makeup theo nguyên tắc bánh xe màu"
        ]
        
        for tip in beauty_tips:
            st.markdown(tip)

    def _render_longterm_consulting(self, gemini_result: dict):
        """Tư vấn dài hạn và theo dõi tiến trình"""
        st.markdown("**📅 Kế hoạch dài hạn:**")
        
        longterm_plan = [
            "📊 **Theo dõi**: Chụp ảnh định kỳ để theo dõi tiến trình (2 tuần/lần)",
            "🔄 **Điều chỉnh**: Thay đổi routine theo mùa và tình trạng da",
            "👩‍⚕️ **Chuyên gia**: Tham khảo bác sĩ da liễu nếu có vấn đề nghiêm trọng",
            "📚 **Học hỏi**: Cập nhật kiến thức chăm sóc da thường xuyên",
            "💪 **Kiên trì**: Duy trì routine ít nhất 4-6 tuần để thấy hiệu quả"
        ]
        
        for plan in longterm_plan:
            st.markdown(plan)
        
        # Thêm timeline tư vấn
        st.markdown("**⏰ Timeline kỳ vọng:**")
        timeline_data = {
            "Tuần 1-2": "Làm quen với routine, da có thể purging",
            "Tuần 3-4": "Da bắt đầu ổn định, giảm kích ứng",
            "Tuần 5-8": "Thấy cải thiện rõ rệt về texture và tone",
            "Tuần 9-12": "Kết quả ổn định, da khỏe mạnh hơn"
        }
        
        for period, expectation in timeline_data.items():
            st.markdown(f"• **{period}**: {expectation}")

    def _render_personalized_consultation(self):
        """Phần tư vấn cá nhân hóa bổ sung"""
        st.markdown("### 🎯 Tư vấn cá nhân hóa")
        
        # Tạo form tư vấn interactive
        with st.expander("💬 Đặt câu hỏi cho chuyên gia AI"):
            user_question = st.text_area(
                "Bạn có thắc mắc gì về chăm sóc da?",
                placeholder="Ví dụ: Tại sao da tôi hay bị mụn vào mùa hè? Tôi nên dùng serum nào cho da nhạy cảm?",
                height=100
            )
            
            if st.button("💡 Nhận tư vấn", key="consultation_btn"):
                if user_question and self.gemini_analyzer.is_available:
                    with st.spinner("Đang phân tích và tư vấn..."):
                        consultation_prompt = f"""
                        Bạn là chuyên gia da liễu với 15 năm kinh nghiệm. Hãy trả lời câu hỏi sau một cách chuyên nghiệp và chi tiết:
                        
                        Câu hỏi: {user_question}
                        
                        Hãy đưa ra:
                        1. Giải thích nguyên nhân
                        2. Giải pháp cụ thể
                        3. Sản phẩm khuyên dùng
                        4. Lời khuyên phòng ngừa
                        
                        Trả lời bằng tiếng Việt, dễ hiểu và thực tế.
                        """
                        
                        try:
                            consultation_result = self.gemini_analyzer.model.generate_content(consultation_prompt)
                            st.markdown("**🩺 Lời tư vấn từ chuyên gia:**")
                            st.markdown(consultation_result.text)
                        except Exception as e:
                            st.error(f"Không thể tư vấn lúc này: {str(e)}")
                elif not user_question:
                    st.warning("Vui lòng nhập câu hỏi để nhận tư vấn")
                else:
                    st.error("Gemini AI chưa sẵn sàng. Vui lòng kiểm tra API key.")

    def _render_text_columns(self, text: str, num_cols: int = 2) -> None:
        """Hiển thị một đoạn markdown theo nhiều cột ngang.
        Chia đều các dòng sang num_cols cột để đọc nhanh hơn."""
        if not text:
            return
        lines = [ln for ln in text.split('\n') if ln.strip()]
        if num_cols < 2 or len(lines) < 8:
            # Nội dung ngắn thì hiển thị một cột bình thường
            st.markdown(text)
            return
        cols = st.columns(num_cols)
        # Chia đều theo số lượng dòng
        chunk_size = (len(lines) + num_cols - 1) // num_cols
        for i in range(num_cols):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(lines))
            with cols[i]:
                st.markdown('\n'.join(lines[start:end]))
    
    def _get_category_name(self, category: str) -> str:
        """Chuyển đổi tên danh mục sang tiếng Việt"""
        category_names = {
            "cleanser": "Sữa rửa mặt",
            "serum": "Serum",
            "moisturizer": "Kem dưỡng ẩm",
            "sunscreen": "Kem chống nắng"
        }
        return category_names.get(category, category)


#SỬA
rec = ProductRecommender(product_file="products.xlsx", seed=42)

result = rec.get_product_recommendations(
    skin_type="Da hỗn hợp",
    skin_condition="Mụn",
    skin_concerns=["mụn", "lỗ chân lông to"],
    age_group="26-35",
    budget_level="trung bình",
    max_products=6,
    products_per_category=1
)

rec.save_recommendations(result, "skincare_recommendations.json")
#SỬA


def main():
    """Hàm chính khởi chạy ứng dụng"""
    app = SkincareAIApp()
    app.main()

if __name__ == "__main__":
    main() 