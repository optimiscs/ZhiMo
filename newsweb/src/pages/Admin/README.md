# 数据库管理界面

这是一个基于 Ant Design 的高级数据库管理界面，支持动态获取数据库表结构并进行 CRUD 操作。

## 功能特性

### 🚀 核心功能
- **动态表检测**：自动获取数据库中可用的表及其结构信息
- **智能表单生成**：根据字段类型自动生成对应的表单控件
- **完整CRUD操作**：支持创建、读取、更新、删除操作
- **高级搜索**：支持跨字段的模糊搜索
- **批量操作**：支持批量删除记录
- **数据导出**：支持将表数据导出为CSV格式

### 📊 界面特性
- **响应式设计**：适配各种屏幕尺寸
- **实时统计**：显示各表的记录数量统计
- **分页支持**：支持大数据量的分页浏览
- **排序功能**：支持按任意字段排序
- **表格配置**：智能列宽和类型识别

## 技术实现

### 后端API设计

#### 1. 获取可用表列表
```http
GET /api/v1/admin/tables
```

响应结构：
```json
{
  "success": true,
  "data": [
    {
      "name": "users",
      "title": "Users",
      "columns": [...],
      "total_columns": 8
    }
  ]
}
```

#### 2. 获取表结构信息
```http
GET /api/v1/admin/tables/{table_name}/schema
```

#### 3. 获取表数据（支持分页、搜索、排序）
```http
GET /api/v1/admin/tables/{table_name}/data
```

参数：
- `page`: 页码
- `pageSize`: 每页大小
- `search`: 搜索关键词
- `sortField`: 排序字段
- `sortOrder`: 排序方向（asc/desc）

#### 4. 创建记录
```http
POST /api/v1/admin/tables/{table_name}/data
```

#### 5. 更新记录
```http
PUT /api/v1/admin/tables/{table_name}/data/{id}
```

#### 6. 删除记录
```http
DELETE /api/v1/admin/tables/{table_name}/data/{id}
```

#### 7. 批量删除
```http
DELETE /api/v1/admin/tables/{table_name}/data/batch
```

### 前端组件架构

#### 主要组件
- `AdminManagement`: 主管理组件
- `TableSelector`: 表选择器
- `DynamicTable`: 动态表格组件
- `DynamicForm`: 动态表单组件
- `StatsCards`: 统计卡片组件

#### 状态管理
使用 React Hooks 进行状态管理：
- `selectedTable`: 当前选中的表
- `tableData`: 表格数据
- `tableColumns`: 表结构信息
- `pagination`: 分页信息
- `searchText`: 搜索文本

#### 表单字段映射
根据数据库字段类型自动映射为对应的表单控件：

| 数据库类型 | 表单控件 | 说明 |
|-----------|---------|-----|
| VARCHAR/TEXT | Input/TextArea | 文本输入框 |
| INTEGER/NUMERIC | InputNumber | 数字输入框 |
| BOOLEAN | Switch | 开关组件 |
| DATETIME/DATE | DatePicker | 日期选择器 |
| email字段 | Input(email) | 邮箱输入框 |
| password字段 | Input.Password | 密码输入框 |

## 使用说明

### 1. 访问管理界面
导航到 `/admin` 路径即可访问数据库管理界面。

### 2. 选择要管理的表
在页面顶部的下拉菜单中选择要管理的数据库表。

### 3. 查看和搜索数据
- 使用搜索框进行全文搜索
- 点击表头进行排序
- 使用分页控件浏览大量数据

### 4. 添加新记录
1. 点击"新增"按钮
2. 在弹出的表单中填写信息
3. 点击确定保存

### 5. 编辑记录
1. 点击记录行的"编辑"按钮
2. 修改表单中的信息
3. 点击确定保存更改

### 6. 删除记录
- **单个删除**：点击记录行的"删除"按钮
- **批量删除**：选中多条记录后，使用"更多操作"菜单中的"批量删除"

### 7. 导出数据
在"更多操作"菜单中点击"导出数据"，系统会生成CSV文件供下载。

## 安全考虑

### 权限控制
- 所有API都需要用户登录认证
- 使用 `@login_required` 装饰器确保安全访问

### 数据验证
- 后端进行严格的数据类型验证
- 前端表单验证确保数据完整性
- SQL注入防护通过ORM实现

### 表访问限制
通过 `AVAILABLE_TABLES` 配置限制可访问的表，防止未授权访问敏感表。

## 扩展开发

### 添加新的可管理表
在后端 `admin.py` 中的 `AVAILABLE_TABLES` 字典中添加新的表映射：

```python
AVAILABLE_TABLES = {
    'users': User,
    'news': News,
    'your_new_table': YourNewModel,  # 添加这里
}
```

### 自定义字段类型处理
在 `get_table_columns` 函数中添加新的字段类型处理逻辑。

### 自定义表单控件
在前端的 `renderFormItem` 函数中添加新的表单控件类型。

## 性能优化

1. **分页加载**：大数据表使用分页避免一次性加载过多数据
2. **延迟加载**：表结构信息按需加载
3. **搜索优化**：使用数据库索引优化搜索性能
4. **缓存策略**：表结构信息可以考虑缓存

## 故障排除

### 常见问题

1. **表不显示**
   - 检查 `AVAILABLE_TABLES` 中是否包含该表
   - 确认数据库连接正常
   - 查看后端日志错误信息

2. **数据加载失败**
   - 检查API权限设置
   - 确认数据库表结构正确
   - 查看网络请求是否成功

3. **表单提交失败**
   - 检查必填字段是否填写
   - 确认数据类型格式正确
   - 查看后端验证错误信息

### 调试建议
- 开启浏览器开发者工具查看网络请求
- 检查后端日志文件
- 使用数据库工具验证数据状态

## 更新日志

### v1.0.0 (2025-09-12)
- 初始版本发布
- 支持基本的CRUD操作
- 实现动态表单生成
- 添加批量操作功能
- 支持数据导出
