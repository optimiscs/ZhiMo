# 新闻分析系统测试指南

## 📋 系统概述

重构后的新闻分析系统采用简化架构：
- **采集**: 从API获取热搜新闻 → 热度归一化 → 保存到`hot_news`表
- **分析**: 从数据库获取标题 → 大模型分析 → 保存到`analyzed_news`表
- **接口**: 前端获取分析结果

## 🚀 启动服务

### 1. 启动基础服务
```bash
# 启动Redis（如果使用Docker）
docker run -d -p 6379:6379 redis:alpine

# 或者启动完整的Docker Compose
docker-compose up -d
```

### 2. 启动Celery Worker
```bash
cd ChatBackend
celery -A celery_app worker --loglevel=info
```

### 3. 启动Celery Beat（定时任务，可选）
```bash
celery -A celery_app beat --loglevel=info
```

### 4. 启动Flask应用
```bash
python run.py
```

## 🧪 手动测试Celery任务

### 使用测试脚本
```bash
cd ChatBackend

# 测试所有任务
python test_celery_tasks.py all

# 测试单个任务
python test_celery_tasks.py heartbeat
python test_celery_tasks.py collect
python test_celery_tasks.py analyze
python test_celery_tasks.py full
```

### 手动执行任务
```python
# 进入Python环境
from app.tasks import *

# 测试心跳
result = heartbeat.delay()
print(result.get())

# 测试采集
result = collect_news_task.delay()
print(result.get())

# 测试分析
result = batch_analyze_news_task.delay(limit=5)
print(result.get())

# 测试完整流程
result = full_process_task.delay(collect_limit=30, analyze_limit=5)
print(result.get())
```

## 🌐 测试前端API接口

### 使用API测试脚本
```bash
cd ChatBackend
python test_api_endpoints.py
```

### 手动测试API接口

#### 1. 登录
```bash
curl -X POST http://localhost:5000/api/login/account \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123","type":"account"}' \
  -c cookies.txt
```

#### 2. 采集新闻
```bash
curl -X POST http://localhost:5000/api/collect_news \
  -H "Content-Type: application/json" \
  -b cookies.txt
```

#### 3. 分析新闻
```bash
curl -X POST http://localhost:5000/api/analyze_hot_news \
  -H "Content-Type: application/json" \
  -d '{"limit":5}' \
  -b cookies.txt
```

#### 4. 完整流程
```bash
curl -X POST http://localhost:5000/api/full_process \
  -H "Content-Type: application/json" \
  -d '{"collect_limit":30,"analyze_limit":5}' \
  -b cookies.txt
```

#### 5. 获取当前新闻
```bash
curl -X GET http://localhost:5000/api/currentnews \
  -b cookies.txt
```

#### 6. 获取已分析新闻
```bash
curl -X GET http://localhost:5000/api/analyze_news \
  -b cookies.txt
```

#### 7. 触发更新流程
```bash
curl -X GET "http://localhost:5000/api/currentnews?update=true" \
  -b cookies.txt
```

## 🧪 通过API测试Celery任务

### 提交任务
```bash
# 测试心跳任务
curl -X POST http://localhost:5000/api/test_celery/heartbeat \
  -H "Content-Type: application/json" \
  -b cookies.txt

# 测试采集任务
curl -X POST http://localhost:5000/api/test_celery/collect \
  -H "Content-Type: application/json" \
  -b cookies.txt

# 测试分析任务
curl -X POST http://localhost:5000/api/test_celery/analyze \
  -H "Content-Type: application/json" \
  -d '{"limit":5}' \
  -b cookies.txt

# 测试完整流程任务
curl -X POST http://localhost:5000/api/test_celery/full \
  -H "Content-Type: application/json" \
  -d '{"collect_limit":30,"analyze_limit":5}' \
  -b cookies.txt
```

### 查看任务结果
```bash
# 使用返回的task_id查看结果
curl -X GET http://localhost:5000/api/task_result/YOUR_TASK_ID \
  -b cookies.txt
```

## 📊 验证数据库

### 检查MongoDB集合
```javascript
// 连接MongoDB
use your_database_name

// 查看热搜新闻数据
db.hot_news.find().limit(5)

// 查看已分析新闻数据
db.analyzed_news.find().limit(5)

// 统计数据量
db.hot_news.countDocuments()
db.analyzed_news.countDocuments()
```

## 🔍 故障排除

### 常见问题

#### 1. Celery连接Redis失败
```bash
# 检查Redis状态
redis-cli ping

# 检查Redis配置
echo $CELERY_BROKER_URL
echo $CELERY_RESULT_BACKEND
```

#### 2. 大模型API调用失败
检查配置文件中的：
- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_MODEL`

#### 3. MongoDB连接失败
检查MongoDB连接字符串和数据库权限

#### 4. 任务执行超时
- 检查网络连接
- 增加任务超时时间
- 减少分析数量限制

### 日志查看
```bash
# Celery worker日志
celery -A celery_app worker --loglevel=debug

# Flask应用日志
python run.py

# 检查任务状态
celery -A celery_app inspect active
celery -A celery_app inspect scheduled
```

## 📈 性能监控

### Celery监控
```bash
# 查看活跃任务
celery -A celery_app inspect active

# 查看任务统计
celery -A celery_app inspect stats

# 查看已注册任务
celery -A celery_app inspect registered
```

### 数据库监控
```javascript
// MongoDB性能统计
db.hot_news.aggregate([{$group: {_id: "$platform", count: {$sum: 1}}}])
db.analyzed_news.aggregate([{$group: {_id: {$dateToString: {format: "%Y-%m-%d", date: {$dateFromString: {dateString: "$analyzed_at"}}}}, count: {$sum: 1}}}])
```

## 🎯 测试建议

1. **逐步测试**: 先测试心跳 → 采集 → 分析 → 完整流程
2. **小量测试**: 开始时使用较小的limit值（如5条新闻）
3. **监控日志**: 密切关注Celery worker和Flask应用的日志输出
4. **数据验证**: 每次测试后检查数据库中的数据是否正确保存
5. **错误处理**: 注意观察错误信息，及时调整配置

## 📝 自动化测试

### 定时任务验证
系统会自动运行以下定时任务：
- 每分钟：心跳检查
- 每30分钟：智能采集新闻
- 每2小时：批量分析新闻
- 每4小时：完整流程（采集+分析）

检查这些任务是否正常运行：
```bash
# 查看定时任务状态
celery -A celery_app inspect scheduled

# 查看beat调度器日志
celery -A celery_app beat --loglevel=info
```
