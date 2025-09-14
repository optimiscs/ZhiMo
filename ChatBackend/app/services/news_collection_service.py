import requests
import re
from datetime import datetime
from app.extensions import db

class NewsCollectionService:
    """简化的新闻采集服务 - 从API获取热搜新闻，热度归一化后保存到数据库"""
    
    @staticmethod
    def fetch_and_save_hot_news():
        """
        从API获取热搜新闻，热度归一化后保存到数据库
        
        Returns:
            dict: 处理结果
        """
        try:
            print("开始获取热搜新闻...")
            
            # API端点
            api_endpoints = {
                "weibo": "https://api-hot.imsyy.top/weibo?cache=true",
                "baidu": "https://api-hot.imsyy.top/baidu?cache=true", 
                "douyin": "https://api-hot.imsyy.top/douyin?cache=true",
                "bilibili": "https://api-hot.imsyy.top/bilibili?cache=true",
                "toutiao": "https://api-hot.imsyy.top/toutiao?cache=true",
                "zhihu": "https://api-hot.imsyy.top/zhihu?cache=true"
            }
            
            # 平台权重
            platform_weights = {
                "weibo": 1.2,
                "baidu": 1.0, 
                "douyin": 0.9,
                "bilibili": 0.8,
                "toutiao": 0.9,
                "zhihu": 0.8
            }
            
            all_news = []
            
            # 获取各平台数据
            for platform_key, api_url in api_endpoints.items():
                try:
                    response = requests.get(api_url, timeout=10)
                    data = response.json()
                    
                    if data.get("code") == 200:
                        for news in data.get("data", []):
                            heat_value = NewsCollectionService._parse_heat(news.get("hot", 0))
                            weighted_heat = heat_value * platform_weights.get(platform_key, 1.0)
                            
                            all_news.append({
                                "title": news.get("title", ""),
                                "url": news.get("url", ""),
                                "platform": platform_key,
                                "raw_heat": heat_value,
                                "weighted_heat": weighted_heat
                            })
                except Exception as e:
                    print(f"获取{platform_key}数据失败: {str(e)}")
                    continue
            
            if not all_news:
                return {"status": "error", "message": "未获取到任何新闻数据"}
            
            # 热度归一化
            max_heat = max(news["weighted_heat"] for news in all_news)
            for news in all_news:
                news["normalized_heat"] = news["weighted_heat"] / max_heat if max_heat > 0 else 0
            
            # 按热度排序
            all_news.sort(key=lambda x: x["normalized_heat"], reverse=True)
            
            # 保存到数据库
            timestamp = datetime.now().isoformat()
            
            # 清空旧数据并插入新数据
            db.hot_news.delete_many({})
            
            for news in all_news:
                news["timestamp"] = timestamp
                db.hot_news.insert_one(news)
            
            print(f"成功保存{len(all_news)}条热搜新闻到数据库")
            
            return {
                "status": "success",
                "count": len(all_news),
                "message": f"成功获取并保存{len(all_news)}条热搜新闻"
            }
            
        except Exception as e:
            print(f"获取热搜新闻失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def _parse_heat(hot_value):
        """
        解析热度值为数字
        
        Args:
            hot_value: 热度值
            
        Returns:
            float: 数字热度值
        """
        if isinstance(hot_value, (int, float)):
            return float(hot_value)
        
        if not hot_value or not isinstance(hot_value, str):
            return 0
            
        # 清理字符串并匹配数字和单位
        cleaned = str(hot_value).strip().replace(',', '')
        match = re.search(r'([\d\.]+)([亿万千])?', cleaned)
        
        if not match:
            return 0
            
        base_num = float(match.group(1))
        unit = match.group(2) if match.group(2) else ''
        
        multipliers = {'': 1, '千': 1000, '万': 10000, '亿': 100000000}
        return base_num * multipliers.get(unit, 1)
    
    @staticmethod
    def get_hot_news_from_db(limit=20):
        """
        从数据库获取热搜新闻
        
        Args:
            limit (int): 限制数量
            
        Returns:
            list: 热搜新闻列表
        """
        try:
            news_list = list(db.hot_news.find(
                {}, 
                {"_id": 0}
            ).sort("normalized_heat", -1).limit(limit))
            
            return news_list
            
        except Exception as e:
            print(f"从数据库获取热搜新闻失败: {str(e)}")
            return []
