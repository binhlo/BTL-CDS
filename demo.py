#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để test các module cơ bản của hệ thống Skincare AI
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import os

def create_demo_face_image():
    """Tạo ảnh khuôn mặt demo để test"""
    # Tạo ảnh trắng 400x400
    img = Image.new('RGB', (400, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Vẽ khuôn mặt đơn giản
    # Khuôn mặt (hình tròn)
    draw.ellipse([100, 100, 300, 300], outline='black', width=2)
    
    # Mắt (2 hình tròn nhỏ)
    draw.ellipse([150, 180, 180, 200], fill='black')  # Mắt trái
    draw.ellipse([220, 180, 250, 200], fill='black')  # Mắt phải
    
    # Mũi (hình tam giác)
    draw.polygon([(200, 220), (190, 250), (210, 250)], fill='black')
    
    # Miệng (hình cung)
    draw.arc([180, 260, 220, 280], start=0, end=180, fill='black', width=2)
    
    # Lưu ảnh
    filename = "demo_face.jpg"
    img.save(filename)
    print(f"Đã tạo ảnh demo: {filename}")
    return filename

def test_face_analyzer():
    """Test module phân tích khuôn mặt"""
    print("\n=== TEST FACE ANALYZER ===")
    
    try:
        from face_analyzer import FaceAnalyzer
        
        # Tạo ảnh demo
        demo_image_path = create_demo_face_image()
        
        # Đọc ảnh
        image = cv2.imread(demo_image_path)
        if image is None:
            print("❌ Không thể đọc ảnh demo")
            return False
        
        # Khởi tạo analyzer
        analyzer = FaceAnalyzer()
        
        # Phân tích khuôn mặt
        print("🔍 Đang phân tích khuôn mặt...")
        results = analyzer.analyze_complete_face(image)
        
        if 'error' in results:
            print(f"❌ Lỗi: {results['error']}")
            return False
        
        print("✅ Phân tích khuôn mặt thành công!")
        print(f"   - Số khuôn mặt: {results['face_count']}")
        print(f"   - Điểm đối xứng: {results['basic_features']['symmetry_score']:.3f}")
        print(f"   - Độ mịn da: {results['texture_features']['smoothness']:.3f}")
        
        # Xóa file demo
        if os.path.exists(demo_image_path):
            os.remove(demo_image_path)
        
        return True
        
    except ImportError as e:
        print(f"❌ Không thể import FaceAnalyzer: {e}")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi test FaceAnalyzer: {e}")
        return False

def test_skin_analyzer():
    """Test module phân tích loại da"""
    print("\n=== TEST SKIN ANALYZER ===")
    
    try:
        from skin_analyzer import SkinAnalyzer
        
        # Khởi tạo analyzer
        analyzer = SkinAnalyzer()
        
        # Test tạo dữ liệu giả lập
        print("📊 Đang tạo dữ liệu giả lập...")
        features, labels = analyzer.generate_synthetic_data(100)
        
        if len(features) == 0:
            print("❌ Không thể tạo dữ liệu giả lập")
            return False
        
        print(f"✅ Đã tạo {len(features)} mẫu dữ liệu")
        print(f"   - Số đặc điểm: {features.shape[1]}")
        print(f"   - Số loại da: {len(set(labels))}")
        
        # Test huấn luyện model
        print("🤖 Đang huấn luyện model...")
        accuracy = analyzer.train_model(features, labels)
        
        if accuracy > 0.7:
            print(f"✅ Huấn luyện thành công! Độ chính xác: {accuracy:.3f}")
        else:
            print(f"⚠️ Độ chính xác thấp: {accuracy:.3f}")
        
        # Test dự đoán
        print("🔮 Đang test dự đoán...")
        test_features = features[0:1]  # Lấy mẫu đầu tiên
        prediction = analyzer.predict_skin_type(test_features)
        
        if 'error' not in prediction:
            print(f"✅ Dự đoán thành công: {prediction['skin_type']}")
            print(f"   - Độ tin cậy: {prediction['confidence']:.1%}")
        else:
            print(f"❌ Lỗi dự đoán: {prediction['error']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Không thể import SkinAnalyzer: {e}")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi test SkinAnalyzer: {e}")
        return False

def test_product_recommender():
    """Test module tư vấn sản phẩm"""
    print("\n=== TEST PRODUCT RECOMMENDER ===")
    
    try:
        from product_recommender import ProductRecommender
        
        # Khởi tạo recommender
        recommender = ProductRecommender()
        
        # Test tạo khuyến nghị
        print("💡 Đang tạo khuyến nghị sản phẩm...")
        recommendations = recommender.get_product_recommendations(
            skin_type="Da khô",
            skin_concerns=["da khô", "nếp nhăn"],
            age_group="26-35",
            budget_level="trung bình"
        )
        
        if 'error' in recommendations:
            print(f"❌ Lỗi: {recommendations['error']}")
            return False
        
        print("✅ Tạo khuyến nghị thành công!")
        print(f"   - Loại da: {recommendations['skin_type']}")
        print(f"   - Số sản phẩm: {len(recommendations['recommended_products'])}")
        print(f"   - Tổng chi phí: {recommendations['total_estimated_cost']:,} VNĐ")
        
        # Test lưu khuyến nghị
        print("💾 Đang lưu khuyến nghị...")
        if recommender.save_recommendations(recommendations, "test_recommendations.json"):
            print("✅ Lưu khuyến nghị thành công!")
            
            # Xóa file test
            if os.path.exists("test_recommendations.json"):
                os.remove("test_recommendations.json")
        else:
            print("❌ Lỗi khi lưu khuyến nghị")
        
        return True
        
    except ImportError as e:
        print(f"❌ Không thể import ProductRecommender: {e}")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi test ProductRecommender: {e}")
        return False

def main():
    """Hàm chính chạy tất cả test"""
    print("🚀 Bắt đầu test hệ thống Skincare AI...")
    print("=" * 50)
    
    # Test từng module
    tests = [
        ("Face Analyzer", test_face_analyzer),
        ("Skin Analyzer", test_skin_analyzer),
        ("Product Recommender", test_product_recommender)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Lỗi không mong muốn trong {test_name}: {e}")
            results.append((test_name, False))
    
    # Tổng kết kết quả
    print("\n" + "=" * 50)
    print("📊 TỔNG KẾT KẾT QUẢ TEST")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Tổng cộng: {passed}/{total} test thành công")
    
    if passed == total:
        print("🎉 Tất cả test đều thành công! Hệ thống hoạt động tốt.")
    elif passed > 0:
        print("⚠️ Một số test thất bại. Vui lòng kiểm tra lỗi.")
    else:
        print("❌ Tất cả test đều thất bại. Vui lòng kiểm tra cài đặt.")
    
    print("\n💡 Để chạy ứng dụng chính, sử dụng:")
    print("   streamlit run main_app.py")

if __name__ == "__main__":
    main() 