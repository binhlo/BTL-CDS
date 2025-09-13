# 🌸 Skincare AI - Hệ thống tư vấn chăm sóc da cá nhân hóa

Hệ thống AI thông minh phân tích khuôn mặt và tư vấn sản phẩm skincare phù hợp với từng loại da.

## ✨ Tính năng chính

### 🔍 Phân tích khuôn mặt
- **Nhận diện khuôn mặt** sử dụng OpenCV và Haar Cascade
- **Trích xuất đặc điểm** khuôn mặt (mắt, mũi, miệng)
- **Tính toán đối xứng** và tỷ lệ khuôn mặt
- **Phân tích kết cấu da** (độ mịn, tương phản, độ sáng)
- **🤖 Phân tích AI nâng cao** với Google Gemini Pro Vision
- **So sánh kết quả** giữa AI truyền thống và Gemini AI

### 🧬 Phân tích loại da bằng AI
- **Machine Learning** sử dụng Random Forest Classifier
- **Phân loại 4 loại da chính**: Da khô, Da dầu, Da hỗn hợp, Da nhạy cảm
- **Trích xuất 18 đặc điểm** từ ảnh để phân tích
- **Tự động huấn luyện** với dữ liệu giả lập

### 💡 Tư vấn sản phẩm thông minh
- **Cơ sở dữ liệu sản phẩm** đa dạng với 4 danh mục chính
- **Thuật toán khuyến nghị** dựa trên loại da, vấn đề da, độ tuổi, ngân sách
- **Tư vấn cá nhân hóa** phù hợp với nhu cầu cụ thể

### 📋 Quy trình skincare
- **Lịch trình sử dụng** sản phẩm theo buổi sáng/tối
- **Hướng dẫn chi tiết** cho từng bước
- **Lời khuyên chăm sóc** theo loại da

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống
- Python 3.8+
- Windows 10/11 (đã test)
- RAM: 4GB+ (khuyến nghị 8GB+)
- Webcam hoặc ảnh để test

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình Gemini AI (Tùy chọn)
1. Lấy API key từ [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Tạo file `.streamlit/secrets.toml` và thêm API key:
```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

### Bước 3: Chạy ứng dụng
```bash
streamlit run main_app.py
```

Ứng dụng sẽ mở trong trình duyệt tại: `http://localhost:8501`

## 📁 Cấu trúc dự án

```
BTL-CĐS/
├── main_app.py              # Ứng dụng chính (Streamlit)
├── face_analyzer.py         # Module phân tích khuôn mặt
├── skin_analyzer.py         # Module phân tích loại da (ML)
├── product_recommender.py   # Module tư vấn sản phẩm
├── gemini_analyzer.py       # Module phân tích với Gemini AI
├── requirements.txt         # Thư viện cần thiết
├── .streamlit/secrets.toml  # Cấu hình API key (tùy chọn)
├── README.md               # Hướng dẫn sử dụng
└── skincare_recommendations_*.json  # File kết quả (tự động tạo)
```

## 🎯 Hướng dẫn sử dụng

### 1. Phân tích khuôn mặt
- **Tải ảnh lên**: Chọn ảnh có khuôn mặt rõ ràng
- **Chụp ảnh tốt**: Ánh sáng đều, khuôn mặt chiếm 50%+ ảnh
- **Nhấn "Phân tích"**: Hệ thống sẽ tự động phân tích

### 2. Xem kết quả phân tích
- **Thông tin cơ bản**: Số khuôn mặt, điểm đối xứng, loại da
- **Biểu đồ chi tiết**: Đặc điểm khuôn mặt, kết cấu da
- **Xác suất loại da**: Phân bố xác suất các loại da

### 3. Nhận tư vấn sản phẩm
- **Cập nhật thông tin**: Độ tuổi, vấn đề da, ngân sách
- **Tạo khuyến nghị**: Hệ thống sẽ đề xuất sản phẩm phù hợp
- **Xem chi tiết**: Thông tin sản phẩm, giá cả, thành phần

### 4. Quy trình skincare
- **Lịch trình sử dụng**: Hướng dẫn từng bước
- **Lời khuyên**: Tips chăm sóc da theo loại da
- **Lưu kết quả**: Xuất file JSON để tham khảo

## 🔧 Tùy chỉnh và mở rộng

### Thêm sản phẩm mới
Chỉnh sửa `product_recommender.py`:
```python
# Thêm sản phẩm vào self.products_database
"new_category": {
    "Da khô": [
        {
            "name": "Tên sản phẩm",
            "brand": "Thương hiệu",
            "price": "Giá",
            "ingredients": ["Thành phần"],
            "benefits": ["Lợi ích"],
            "rating": 4.5
        }
    ]
}
```

### Huấn luyện lại model
```python
# Tự động huấn luyện
skin_analyzer.auto_train()

# Hoặc huấn luyện thủ công
features, labels = skin_analyzer.generate_synthetic_data(2000)
accuracy = skin_analyzer.train_model(features, labels)
```

### Thêm loại da mới
Chỉnh sửa `skin_analyzer.py`:
```python
self.skin_types = {
    0: "Da khô",
    1: "Da dầu", 
    2: "Da hỗn hợp",
    3: "Da nhạy cảm",
    4: "Da mới"  # Thêm loại da mới
}
```
### Hình ảnh demo
Hình ảnh demo
<img width="1822" height="673" alt="image" src="https://github.com/user-attachments/assets/0ff59feb-1435-4c10-a199-774fde28c6bf" />
<img width="1774" height="839" alt="image" src="https://github.com/user-attachments/assets/c3b58069-08b7-44d8-a0cc-3b9ac3c27739" />
<img width="1804" height="837" alt="image" src="https://github.com/user-attachments/assets/f17558d2-6863-4c4a-ac62-c092da97cae5" />
<img width="1797" height="867" alt="image" src="https://github.com/user-attachments/assets/dc8935ff-91ec-4b0d-a82e-acfe76c5319f" />
<img width="1807" height="722" alt="image" src="https://github.com/user-attachments/assets/b5490e63-31b6-440c-a1d0-5083fd9ce226" />
<img width="1840" height="657" alt="image" src="https://github.com/user-attachments/assets/7823c700-6e5b-4b3a-bf95-6f765968fef3" />
<img width="1835" height="405" alt="image" src="https://github.com/user-attachments/assets/7b837934-7dba-4667-8d45-6059c0efc9ee" />
<img width="1756" height="629" alt="image" src="https://github.com/user-attachments/assets/24456829-f384-4583-8315-28c5fefd7217" />



## 📊 Hiệu suất và độ chính xác

### Model phân loại da
- **Độ chính xác**: 30-65% (với dữ liệu giả lập)
- **Thời gian huấn luyện**: ~30 giây (1000 mẫu)
- **Thời gian dự đoán**: <1 giây

### Phân tích khuôn mặt
- **Tỷ lệ phát hiện**: 95%+ (với ảnh chất lượng tốt)
- **Xử lý ảnh**: Hỗ trợ JPG, PNG, độ phân giải cao
- **Độ chính xác đặc điểm**: 80-85%

## 🐛 Xử lý lỗi thường gặp

### Lỗi "Không tìm thấy khuôn mặt"
- **Nguyên nhân**: Ảnh mờ, ánh sáng kém, khuôn mặt bị che
- **Giải pháp**: Chụp lại ảnh với ánh sáng tốt, khuôn mặt rõ ràng

### Lỗi "Model chưa được huấn luyện"
- **Nguyên nhân**: Model chưa được tạo hoặc bị hỏng
- **Giải pháp**: Nhấn "Huấn luyện lại model" trong sidebar

### Lỗi cài đặt thư viện
- **Nguyên nhân**: Phiên bản Python không tương thích
- **Giải pháp**: Sử dụng Python 3.8-3.11, cài đặt từng thư viện

## 🔮 Tính năng tương lai

### Ngắn hạn (1-2 tháng)
- [ ] Hỗ trợ video real-time
- [ ] Thêm loại da (da mụn, da lão hóa)
- [ ] Tích hợp camera webcam

### Trung hạn (3-6 tháng)
- [ ] Mobile app (React Native)
- [ ] AI chatbot tư vấn
- [ ] Theo dõi tiến trình da

### Dài hạn (6+ tháng)
- [ ] Phân tích da 3D
- [ ] Tích hợp IoT devices
- [ ] Cộng đồng người dùng

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Hãy:

1. **Fork** dự án
2. **Tạo branch** mới (`git checkout -b feature/AmazingFeature`)
3. **Commit** thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. **Push** lên branch (`git push origin feature/AmazingFeature`)
5. **Tạo Pull Request**

## 📄 Giấy phép

Dự án này được phát hành dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 👥 Tác giả

**Nhóm BTL-CĐS** - Đại học [Tên trường]
- **Sinh viên 1**: [Tên] - [MSSV]
- **Sinh viên 2**: [Tên] - [MSSV]
- **Giảng viên hướng dẫn**: [Tên]

## 🙏 Lời cảm ơn

- **OpenCV** - Thư viện xử lý ảnh
- **Streamlit** - Framework web app
- **Scikit-learn** - Machine learning
- **Cộng đồng open source** - Hỗ trợ và đóng góp

## 📞 Liên hệ

- **Email**: [email@example.com]
- **GitHub**: [github.com/username]
- **Website**: [website.com]

---

⭐ **Nếu dự án này hữu ích, hãy cho chúng tôi một ngôi sao trên GitHub!** 
