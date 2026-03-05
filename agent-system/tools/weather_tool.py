from langchain_core.tools import Tool
from typing import Optional
import json

def get_weather(city: str) -> str:
    """查询城市天气（模拟）
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息的 JSON 字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴天", "humidity": 45, "wind": "北风3级"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": 65, "wind": "东南风2级"},
        "深圳": {"temperature": 32, "condition": "雷阵雨", "humidity": 80, "wind": "南风4级"},
        "成都": {"temperature": 22, "condition": "阴天", "humidity": 70, "wind": "无持续风向"},
    }
    
    # 如果城市不在数据中，返回默认值
    if city not in weather_data:
        return json.dumps({
            "city": city,
            "message": f"抱歉，暂无 {city} 的天气数据",
            "temperature": None,
            "condition": "未知"
        }, ensure_ascii=False)
    
    data = weather_data[city]
    result = {
        "city": city,
        "temperature": data["temperature"],
        "condition": data["condition"],
        "humidity": data["humidity"],
        "wind": data["wind"]
    }
    
    return json.dumps(result, ensure_ascii=False)

# 创建 LangChain Tool
weather_tool = Tool(
    name="weather_query",
    func=get_weather,
    description="""查询指定城市的实时天气信息。
    
输入格式：城市名称（如："北京"、"上海"）
输出格式：JSON 格式的天气信息，包含温度、天气状况、湿度、风力等

使用示例：
- 输入："北京"
- 输出：{"city": "北京", "temperature": 25, "condition": "晴天", ...}

注意：只接受城市名称，不要输入其他信息。"""
)

# 测试
if __name__ == "__main__":
    print("测试天气工具：")
    print(weather_tool.run("北京"))
    print(weather_tool.run("深圳"))