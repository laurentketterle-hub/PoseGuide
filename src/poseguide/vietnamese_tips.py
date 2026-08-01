"""Vietnamese photography tip pack for PoseGuide poses."""
import json
from typing import List, Dict

VIETNAMESE_TIPS: Dict[str, List[str]] = {
    "standing": [
        "Đứng thẳng lưng, vai thả lỏng tự nhiên",
        "Chân đứng rộng bằng vai, đầu gối hơi chùng",
        "Hướng về phía ánh sáng chính để tạo chiều sâu",
        "Tránh đứng thẳng đơ - xoay nhẹ thân 15-30 độ",
    ],
    "sitting": [
        "Ngồi thẳng lưng, không dựa hoàn toàn vào ghế",
        "Đặt tay nhẹ nhàng lên đùi hoặc thành ghế",
        "Chân bắt chéo tự nhiên, mũi chân hướng xuống",
        "Nghiêng người về phía trước 10 độ tạo cảm giác thân thiện",
    ],
    "portrait": [
        "Mắt nhìn vào ống kính hoặc hơi lệch sang bên",
        "Cười nhẹ tự nhiên, không gượng ép",
        "Đầu hơi nghiêng 5-10 độ tạo cảm giác mềm mại",
        "Ánh sáng từ phía trước hoặc góc 45 độ",
    ],
    "outdoor": [
        "Chụp vào giờ vàng (sáng sớm hoặc chiều muộn)",
        "Tận dụng ánh sáng tự nhiên qua tán lá",
        "Tránh ánh nắng gắt giữa trưa (11h-14h)",
        "Phông nền đơn giản, tránh quá nhiều chi tiết gây rối",
    ],
    "couple": [
        "Đứng gần nhau, thân chạm nhẹ tạo cảm giác gắn kết",
        "Một người hơi nghiêng về phía người kia",
        "Tay đan vào nhau tự nhiên, không gượng ép",
        "Cả hai cùng nhìn về một hướng hoặc nhìn nhau",
    ],
    "group": [
        "Sắp xếp theo hình tam giác hoặc đường chéo",
        "Người cao nhất ở giữa, thấp dần ra hai bên",
        "Đảm bảo không ai bị che khuất",
        "Tất cả cùng nhìn về một điểm",
    ],
}

def get_tips_for_pose_family(family: str, lang: str = "vi") -> List[str]:
    """Get photography tips for a pose family in specified language."""
    return VIETNAMESE_TIPS.get(family, VIETNAMESE_TIPS.get("standing", []))

def get_all_tips(lang: str = "vi") -> Dict[str, List[str]]:
    """Get all tips organized by pose family."""
    return {family: tips for family, tips in VIETNAMESE_TIPS.items()}

def export_tips_json(filepath: str) -> None:
    """Export tips to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(VIETNAMESE_TIPS, f, ensure_ascii=False, indent=2)
