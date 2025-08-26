import json
import random
from typing import Dict, List, Optional
import logging
import pandas as pd


class ProductRecommender:
    """Hệ thống tư vấn sản phẩm skincare cá nhân hóa"""

    def __init__(self, product_file: str = "products.xlsx", seed: Optional[int] = None):
        # Load dữ liệu sản phẩm từ Excel
        self.products_database = self._load_products_from_excel(product_file)

        # Khởi tạo quy tắc tư vấn
        self.recommendation_rules = self._initialize_rules()

        if seed is not None:
            random.seed(seed)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_products_from_excel(self, filename: str) -> Dict:
        """Đọc dữ liệu sản phẩm từ file Excel"""
        try:
            df = pd.read_excel(filename)

            required_columns = {"category", "skin_type", "name", "brand", "price", "rating"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Thiếu cột trong file Excel. Cần có: {required_columns}")

            products_db: Dict[str, Dict[str, List[Dict]]] = {}

            for _, row in df.iterrows():
                category = str(row["category"]).strip()
                skin_type = str(row["skin_type"]).strip()

                product = {
                    "name": str(row["name"]),
                    "brand": str(row["brand"]),
                    "price": str(row["price"]),
                    "ingredients": str(row.get("ingredients", "")).split(";") if row.get("ingredients") else [],
                    "benefits": str(row.get("benefits", "")).split(";") if row.get("benefits") else [],
                    "rating": float(row.get("rating", 0)),
                    "image": str(row.get("image", ""))
                }

                products_db.setdefault(category, {}).setdefault(skin_type, []).append(product)

            return products_db
        except Exception as e:
            logging.error(f"Lỗi khi load Excel: {e}")
            return {}

    def _initialize_rules(self) -> Dict:
        """Khởi tạo quy tắc tư vấn"""
        return {
            "skin_concerns": {
                "mụn": ["cleanser", "serum"],
                "vết thâm": ["serum", "moisturizer"],
                "da khô": ["cleanser", "moisturizer", "serum"],
                "da dầu": ["cleanser", "serum", "moisturizer"],
                "lỗ chân lông to": ["cleanser", "serum"],
                "da nhạy cảm": ["cleanser", "moisturizer"],
                "nếp nhăn": ["serum", "moisturizer"]
            },
            "age_groups": {
                "18-25": ["cleanser", "moisturizer", "sunscreen"],
                "26-35": ["cleanser", "serum", "moisturizer", "sunscreen"],
                "36-45": ["cleanser", "serum", "moisturizer", "sunscreen"],
                "45+": ["cleanser", "serum", "moisturizer", "sunscreen"]
            },
            "budget_levels": {
                "thấp": ["cleanser", "moisturizer"],
                "trung bình": ["cleanser", "serum", "moisturizer"],
                "cao": ["cleanser", "serum", "moisturizer", "sunscreen"]
            }
        }

    def get_product_recommendations(
        self,
        skin_type: str,
        skin_condition: Optional[str] = None,
        skin_concerns: Optional[List[str]] = None,
        age_group: str = "26-35",
        budget_level: str = "trung bình",
        max_products: int = 5,
        products_per_category: int = 2
    ) -> Dict:
        """Lấy khuyến nghị sản phẩm dựa trên thông tin cá nhân và tình trạng da"""
        try:
            recommendations = {
                "skin_type": skin_type,
                "skin_condition": skin_condition,
                "recommended_products": [],
                "skincare_routine": [],
                "total_estimated_cost": 0,
                "explanation": ""
            }

            # Xác định danh mục sản phẩm cần thiết
            required_categories = self._get_required_categories(
                skin_type, skin_condition, skin_concerns, age_group, budget_level
            )

            # Lấy sản phẩm cho từng danh mục
            for category in required_categories:
                if category in self.products_database and skin_type in self.products_database[category]:
                    products = self.products_database[category][skin_type]
                    products.sort(key=lambda x: x["rating"], reverse=True)

                    selected_products = self._select_product_by_budget(
                        products, budget_level, category, top_k=max(1, products_per_category)
                    )

                    for selected_product in selected_products:
                        if selected_product:
                            recommendations["recommended_products"].append({
                                "category": category,
                                "product": selected_product
                            })
                            estimated_price = self._extract_price(selected_product["price"])
                            recommendations["total_estimated_cost"] += estimated_price

            # Tạo quy trình skincare
            recommendations["skincare_routine"] = self._create_skincare_routine(
                recommendations["recommended_products"], skin_condition
            )

            # Tạo giải thích
            recommendations["explanation"] = self._create_explanation(
                skin_type, skin_condition, skin_concerns, age_group, budget_level
            )

            return recommendations
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo khuyến nghị: {e}")
            return {"error": str(e)}

    def _get_required_categories(
        self,
        skin_type: str,
        skin_condition: Optional[str],
        skin_concerns: Optional[List[str]],
        age_group: str,
        budget_level: str
    ) -> List[str]:
        """Xác định danh mục sản phẩm cần thiết"""
        base_categories = ["cleanser", "moisturizer"]

        if skin_condition:
            if skin_condition == "Mụn":
                base_categories += ["serum", "spot treatment"]
            elif skin_condition in ["Nám/tàn nhang", "Lỗ chân lông to"]:
                base_categories.append("serum")

        if skin_concerns:
            for concern in skin_concerns:
                if concern in self.recommendation_rules["skin_concerns"]:
                    base_categories.extend(self.recommendation_rules["skin_concerns"][concern])

        if age_group in self.recommendation_rules["age_groups"]:
            base_categories.extend(self.recommendation_rules["age_groups"][age_group])

        if budget_level in self.recommendation_rules["budget_levels"]:
            budget_categories = self.recommendation_rules["budget_levels"][budget_level]
            base_categories = [cat for cat in base_categories if cat in budget_categories]

        # Loại bỏ trùng lặp và giữ thứ tự
        unique_categories = []
        for cat in base_categories:
            if cat not in unique_categories:
                unique_categories.append(cat)

        return unique_categories[:4]

    def _select_product_by_budget(
        self, products: List[Dict], budget_level: str, category: str, top_k: int = 2
    ) -> List[Dict]:
        """Chọn danh sách sản phẩm phù hợp với ngân sách"""
        if not products:
            return []

        top_k = max(1, int(top_k))

        if budget_level == "thấp":
            return sorted(products, key=lambda x: self._extract_price(x["price"]))[:top_k]
        elif budget_level == "cao":
            return sorted(products, key=lambda x: x["rating"], reverse=True)[:top_k]
        else:
            scored = []
            for product in products:
                price = self._extract_price(product["price"])
                score = product["rating"] / (max(price, 1) / 100000)
                scored.append((score, product))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored[:top_k]]

    def _extract_price(self, price_range: str) -> int:
        """Trích xuất giá trung bình từ chuỗi giá"""
        try:
            price_str = str(price_range).replace("VNĐ", "").replace(" ", "")
            if "-" in price_str:
                min_price, max_price = price_str.split("-")
                return (int(min_price.replace(",", "")) + int(max_price.replace(",", ""))) // 2
            return int(price_str.replace(",", ""))
        except:
            return 300000  # Giá mặc định

    def _create_skincare_routine(self, recommended_products: List[Dict], skin_condition: Optional[str]) -> List[Dict]:
        """Tạo quy trình skincare"""
        routine = []
        order = ["cleanser", "serum", "moisturizer", "sunscreen"]

        for step in order:
            for product_info in recommended_products:
                if product_info["category"] == step:
                    routine.append({
                        "step": len(routine) + 1,
                        "category": step,
                        "product_name": product_info["product"]["name"],
                        "brand": product_info["product"]["brand"],
                        "usage": self._get_usage_instructions(step)
                    })
                    break

        if skin_condition == "Mụn":
            routine.append({
                "step": len(routine) + 1,
                "category": "spot treatment",
                "product_name": "Spot Treatment",
                "brand": "Paula's Choice",
                "usage": "Sử dụng sau khi rửa mặt, trước kem dưỡng ẩm"
            })
        elif skin_condition == "Nám/tàn nhang":
            routine.append({
                "step": len(routine) + 1,
                "category": "serum",
                "product_name": "Serum Vitamin C",
                "brand": "Paula's Choice",
                "usage": "Sử dụng sau khi rửa mặt, trước kem dưỡng ẩm"
            })
        elif skin_condition == "Lỗ chân lông to":
            routine.append({
                "step": len(routine) + 1,
                "category": "serum",
                "product_name": "Serum Niacinamide",
                "brand": "The Ordinary",
                "usage": "Sử dụng sau khi rửa mặt, trước kem dưỡng ẩm"
            })

        return routine

    def _get_usage_instructions(self, category: str) -> str:
        return {
            "cleanser": "Sử dụng 2 lần/ngày (sáng và tối)",
            "serum": "Sử dụng sau khi rửa mặt, trước kem dưỡng ẩm",
            "moisturizer": "Sử dụng sau serum, 2 lần/ngày",
            "sunscreen": "Sử dụng vào buổi sáng, thoa lại sau 2-3 giờ"
        }.get(category, "Sử dụng theo hướng dẫn")

    def _create_explanation(
        self, skin_type: str, skin_condition: Optional[str],
        skin_concerns: Optional[List[str]], age_group: str, budget_level: str
    ) -> str:
        explanation = f"Dựa trên phân tích, bạn có loại da {skin_type.lower()}. "
        if skin_condition:
            explanation += f"Với tình trạng da {skin_condition.lower()}. "
        if skin_concerns:
            explanation += f"Với các vấn đề: {', '.join(skin_concerns)}. "
        explanation += f"Ở độ tuổi {age_group}, da cần được chăm sóc đặc biệt. "
        explanation += f"Với ngân sách {budget_level}, chúng tôi đã chọn những sản phẩm chất lượng tốt nhất phù hợp với nhu cầu của bạn."
        return explanation

    def get_alternative_products(self, skin_type: str, category: str, current_product: str) -> List[Dict]:
        """Lấy sản phẩm thay thế"""
        try:
            if category not in self.products_database or skin_type not in self.products_database[category]:
                return []
            products = self.products_database[category][skin_type]
            alternatives = [p for p in products if p["name"] != current_product]
            alternatives.sort(key=lambda x: x["rating"], reverse=True)
            return alternatives[:3]
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy sản phẩm thay thế: {e}")
            return []

    def save_recommendations(self, recommendations: Dict, filename: str = "skincare_recommendations.json") -> bool:
        """Lưu kết quả khuyến nghị vào file JSON"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Đã lưu khuyến nghị vào {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu khuyến nghị: {e}")
            return False
