import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Input,
  Space,
  Modal,
  Form,
  message,
  Popconfirm,
  Select,
  DatePicker,
  InputNumber,
  Switch,
  Row,
  Col,
  Statistic,
  Tag,
  Tooltip,
  Dropdown,
  Menu,
  Divider,
  Typography,
  Upload,
  Progress,
  Alert
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ExportOutlined,
  ReloadOutlined,
  SearchOutlined,
  SettingOutlined,
  TableOutlined,
  DownloadOutlined,
  ExclamationCircleOutlined,
  UploadOutlined,
  InboxOutlined
} from '@ant-design/icons';
import { PageContainer } from '@ant-design/pro-components';
import { createStyles } from 'antd-style';
import dayjs from 'dayjs';
import { request } from '@umijs/max';

const { Search } = Input;
const { Option } = Select;
const { TextArea } = Input;
const { Title, Text } = Typography;
const { Dragger } = Upload;

// 样式定义
const useStyles = createStyles(({ token }) => {
  return {
    container: {
      padding: '0',
      background: token.colorBgContainer,
    },
    tableCard: {
      marginBottom: 24,
      '& .ant-card-head': {
        borderBottom: `1px solid ${token.colorBorderSecondary}`,
        '& .ant-card-head-title': {
          fontWeight: 600,
          fontSize: 16,
        },
      },
    },
    statsCard: {
      marginBottom: 24,
      textAlign: 'center',
      '& .ant-statistic-title': {
        fontSize: 14,
        color: token.colorTextSecondary,
      },
      '& .ant-statistic-content-value': {
        fontSize: 24,
        fontWeight: 600,
      },
    },
    headerActions: {
      display: 'flex',
      gap: 12,
      alignItems: 'center',
    },
    searchBox: {
      maxWidth: 300,
    },
    formItem: {
      marginBottom: 16,
    },
    tableHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 16,
    },
    tableSelector: {
      minWidth: 200,
    },
  };
});

// 数据类型定义
interface TableColumn {
  key: string;
  title: string;
  dataIndex: string;
  type: string;
  nullable: boolean;
  primary_key: boolean;
  autoincrement: boolean;
  input_type: string;
  default?: string;
}

interface TableInfo {
  name: string;
  title: string;
  columns: TableColumn[];
  total_columns: number;
}

interface TableData {
  [key: string]: any;
}

interface AdminStats {
  [tableName: string]: {
    total: number;
    title: string;
  };
}

const AdminManagement: React.FC = () => {
  const { styles } = useStyles();
  const [form] = Form.useForm();
  
  // 状态管理
  const [loading, setLoading] = useState(false);
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [availableTables, setAvailableTables] = useState<TableInfo[]>([]);
  const [tableData, setTableData] = useState<TableData[]>([]);
  const [tableColumns, setTableColumns] = useState<TableColumn[]>([]);
  const [adminStats, setAdminStats] = useState<AdminStats>({});
  
  // 分页和搜索
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 20,
    total: 0,
  });
  const [searchText, setSearchText] = useState('');
  const [sortField, setSortField] = useState<string>('');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  
  // 模态框状态
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<'create' | 'edit'>('create');
  const [editingRecord, setEditingRecord] = useState<TableData | null>(null);
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  
  // 导入相关状态
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importResult, setImportResult] = useState<any>(null);

  // 初始化加载数据
  useEffect(() => {
    loadAvailableTables();
    loadAdminStats();
  }, []);

  // 当选中表变化时加载表数据
  useEffect(() => {
    if (selectedTable) {
      loadTableData();
    }
  }, [selectedTable, pagination.current, pagination.pageSize, searchText, sortField, sortOrder]);

  // API调用函数
  const loadAvailableTables = async () => {
    try {
      setLoading(true);
      const response = await request('/api/v1/admin/tables');
      if (response.success) {
        setAvailableTables(response.data);
        if (response.data.length > 0 && !selectedTable) {
          setSelectedTable(response.data[0].name);
        }
      } else {
        message.error(response.error || '获取表列表失败');
      }
    } catch (error) {
      console.error('获取表列表失败:', error);
      message.error('获取表列表失败');
    } finally {
      setLoading(false);
    }
  };

  const loadAdminStats = async () => {
    try {
      const response = await request('/api/v1/admin/stats');
      if (response.success) {
        setAdminStats(response.data);
      }
    } catch (error) {
      console.error('获取统计信息失败:', error);
    }
  };

  const loadTableData = async () => {
    try {
      setLoading(true);
      const params = {
        page: pagination.current,
        pageSize: pagination.pageSize,
        search: searchText,
        sortField: sortField,
        sortOrder: sortOrder,
      };
      
      const response = await request(`/api/v1/admin/tables/${selectedTable}/data`, {
        params,
      });
      
      if (response.success) {
        setTableData(response.data);
        setPagination(prev => ({
          ...prev,
          total: response.pagination.total,
        }));
        
        // 获取表结构
        const schemaResponse = await request(`/api/v1/admin/tables/${selectedTable}/schema`);
        if (schemaResponse.success) {
          setTableColumns(schemaResponse.data.columns);
        }
      } else {
        message.error(response.error || '获取数据失败');
      }
    } catch (error) {
      console.error('获取数据失败:', error);
      message.error('获取数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (values: any) => {
    try {
      const response = await request(`/api/v1/admin/tables/${selectedTable}/data`, {
        method: 'POST',
        data: values,
      });
      
      if (response.success) {
        message.success('创建成功');
        setModalVisible(false);
        form.resetFields();
        loadTableData();
        loadAdminStats();
      } else {
        message.error(response.error || '创建失败');
      }
    } catch (error) {
      console.error('创建失败:', error);
      message.error('创建失败');
    }
  };

  const handleUpdate = async (values: any) => {
    if (!editingRecord) return;
    
    try {
      const response = await request(`/api/v1/admin/tables/${selectedTable}/data/${editingRecord.id}`, {
        method: 'PUT',
        data: values,
      });
      
      if (response.success) {
        message.success('更新成功');
        setModalVisible(false);
        form.resetFields();
        setEditingRecord(null);
        loadTableData();
      } else {
        message.error(response.error || '更新失败');
      }
    } catch (error) {
      console.error('更新失败:', error);
      message.error('更新失败');
    }
  };

  const handleDelete = async (recordId: number) => {
    try {
      const response = await request(`/api/v1/admin/tables/${selectedTable}/data/${recordId}`, {
        method: 'DELETE',
      });
      
      if (response.success) {
        message.success('删除成功');
        loadTableData();
        loadAdminStats();
      } else {
        message.error(response.error || '删除失败');
      }
    } catch (error) {
      console.error('删除失败:', error);
      message.error('删除失败');
    }
  };

  const handleBatchDelete = async () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请选择要删除的记录');
      return;
    }
    
    try {
      const response = await request(`/api/v1/admin/tables/${selectedTable}/data/batch`, {
        method: 'DELETE',
        data: { ids: selectedRowKeys },
      });
      
      if (response.success) {
        message.success(response.message || '批量删除成功');
        setSelectedRowKeys([]);
        loadTableData();
        loadAdminStats();
      } else {
        message.error(response.error || '批量删除失败');
      }
    } catch (error) {
      console.error('批量删除失败:', error);
      message.error('批量删除失败');
    }
  };

  const handleExport = async () => {
    try {
      const response = await request(`/api/v1/admin/tables/${selectedTable}/export`);
      
      if (response.success) {
        const data = response.data;
        
        // 检查数据格式，支持新旧两种格式
        let columns: string[] = [];
        let rows: any[] = [];
        
        if (data.columns && Array.isArray(data.columns) && data.rows && Array.isArray(data.rows)) {
          // 新格式
          columns = data.columns;
          rows = data.rows;
        } else if (data.fields && Array.isArray(data.fields) && data.documents && Array.isArray(data.documents)) {
          // 旧格式（向后兼容）
          columns = data.fields;
          rows = data.documents;
        } else {
          message.error('导出数据格式错误：无法识别数据结构');
          return;
        }
        
        if (columns.length === 0) {
          message.error('导出失败：没有可导出的字段');
          return;
        }
        
        // 构建CSV内容
        const csvRows = [];
        
        // 添加表头
        csvRows.push(columns.map(col => `"${col}"`).join(','));
        
        // 添加数据行
        rows.forEach((row: any) => {
          const csvRow = columns.map((col: string) => {
            const value = row[col];
            if (value === null || value === undefined) {
              return '""';
            }
            // 处理复杂对象
            if (typeof value === 'object') {
              return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
            }
            // 处理包含逗号、引号或换行符的值
            const stringValue = String(value);
            if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
              return `"${stringValue.replace(/"/g, '""')}"`;
            }
            return `"${stringValue}"`;
          });
          csvRows.push(csvRow.join(','));
        });
        
        const csvContent = csvRows.join('\n');
        
        // 创建并下载文件
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `${selectedTable}_export_${new Date().toISOString().slice(0, 10)}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // 清理URL对象
        URL.revokeObjectURL(url);
        
        message.success(`导出成功！共导出 ${rows.length} 条记录`);
      } else {
        message.error(response.error || '导出失败');
      }
    } catch (error) {
      console.error('导出失败:', error);
      message.error('导出失败');
    }
  };

  const handleDeleteTable = async () => {
    if (!selectedTable) {
      message.warning('请先选择要删除的表');
      return;
    }

    // 保护重要表
    const protectedTables = ['users'];
    if (protectedTables.includes(selectedTable)) {
      message.error('不能删除系统重要表');
      return;
    }

    Modal.confirm({
      title: '确认删除表',
      icon: <ExclamationCircleOutlined />,
      content: (
        <div>
          <p>您确定要删除表 <strong>"{selectedTable}"</strong> 吗？</p>
          <p style={{ color: '#ff4d4f', marginBottom: 0 }}>
            ⚠️ 此操作将永久删除表中的所有数据，且无法恢复！
          </p>
        </div>
      ),
      okText: '确认删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const response = await request(`/api/v1/admin/tables/${selectedTable}`, {
            method: 'DELETE',
          });
          
          if (response.success) {
            message.success(response.message || '表删除成功');
            
            // 重新加载表列表
            await loadAvailableTables();
            await loadAdminStats();
            
            // 如果删除的是当前选中的表，清空选择
            setSelectedTable('');
            setTableData([]);
            setTableColumns([]);
            
          } else {
            message.error(response.error || '删除表失败');
          }
        } catch (error) {
          console.error('删除表失败:', error);
          message.error('删除表失败');
        }
      },
    });
  };

  const handleImport = async (file: File) => {
    if (!selectedTable) {
      message.warning('请先选择要导入数据的表');
      return false;
    }

    try {
      setImportLoading(true);
      setImportProgress(0);
      setImportResult(null);

      // 模拟进度更新
      const progressInterval = setInterval(() => {
        setImportProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      const formData = new FormData();
      formData.append('file', file);
      
      const response = await request(`/api/v1/admin/tables/${selectedTable}/import`, {
        method: 'POST',
        data: formData,
        headers: {
          // 不设置Content-Type，让浏览器自动设置multipart/form-data边界
        },
      });

      clearInterval(progressInterval);
      setImportProgress(100);

      if (response.success) {
        setImportResult(response);
        message.success(response.message || '导入成功');
        
        // 刷新表数据
        await loadTableData();
        await loadAdminStats();
        
        return true;
      } else {
        setImportResult(response);
        message.error(response.error || '导入失败');
        return false;
      }
    } catch (error: any) {
      console.error('导入失败:', error);
      
      // 尝试获取更详细的错误信息
      let errorMessage = '导入失败，请检查网络连接';
      
      if (error?.response?.data) {
        // 如果后端返回了具体错误信息
        if (typeof error.response.data === 'string') {
          errorMessage = error.response.data;
        } else if (error.response.data.error) {
          errorMessage = error.response.data.error;
        } else if (error.response.data.message) {
          errorMessage = error.response.data.message;
        }
      } else if (error?.message) {
        errorMessage = `请求失败: ${error.message}`;
      }
      
      console.error('详细错误信息:', {
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        data: error?.response?.data,
        message: error?.message,
        config: error?.config
      });
      
      message.error(errorMessage);
      setImportResult({
        success: false,
        error: errorMessage
      });
      return false;
    } finally {
      setImportLoading(false);
    }
  };

  const resetImportModal = () => {
    setImportModalVisible(false);
    setImportLoading(false);
    setImportProgress(0);
    setImportResult(null);
  };

  // 表格列配置
  const getTableColumns = () => {
    if (!tableColumns || tableColumns.length === 0) {
      return [];
    }
    
    const columns = tableColumns.map(col => {
      const column: any = {
        title: col.title,
        dataIndex: col.key,
        key: col.key,
        sorter: true,
        ellipsis: true,
      };

      // 通用渲染函数，处理各种数据类型
      const renderValue = (value: any) => {
        if (value === null || value === undefined) {
          return '-';
        }
        
        // 处理对象类型
        if (typeof value === 'object') {
          if (Array.isArray(value)) {
            return value.length > 0 ? JSON.stringify(value) : '[]';
          } else {
            return JSON.stringify(value);
          }
        }
        
        // 处理基本类型
        return String(value);
      };

      // 根据类型设置渲染方式
      if (col.input_type === 'boolean') {
        column.render = (value: boolean) => (
          <Tag color={value ? 'green' : 'red'}>
            {value ? '是' : '否'}
          </Tag>
        );
      } else if (col.input_type === 'datetime') {
        column.render = (value: string) => 
          value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '-';
      } else if (col.input_type === 'json' || col.input_type === 'tags' || col.type === 'array' || col.type === 'object') {
        column.width = 200;
        column.render = (value: any) => (
          <Tooltip title={renderValue(value)}>
            <div style={{ 
              maxWidth: 200, 
              overflow: 'hidden', 
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap' 
            }}>
              {renderValue(value)}
            </div>
          </Tooltip>
        );
      } else if (col.key === 'id' || col.primary_key) {
        column.width = 80;
        column.fixed = 'left';
        column.render = renderValue;
      } else if (col.input_type === 'textarea') {
        column.width = 200;
        column.render = (text: any) => (
          <Tooltip title={renderValue(text)}>
            <div style={{ 
              maxWidth: 200, 
              overflow: 'hidden', 
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap' 
            }}>
              {renderValue(text)}
            </div>
          </Tooltip>
        );
      } else {
        // 默认渲染函数，确保所有值都被正确处理
        column.render = renderValue;
      }

      return column;
    });

    // 添加操作列
    columns.push({
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 120,
      render: (_: any, record: TableData) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EditOutlined />}
            onClick={() => {
              setModalType('edit');
              setEditingRecord(record);
              
              // 填充表单
              const formValues: any = {};
              if (tableColumns) {
                tableColumns.forEach(col => {
                  let value = record[col.key];
                  
                  // 处理不同类型的值
                  if (col.input_type === 'datetime' && value) {
                    value = dayjs(value);
                  } else if ((col.input_type === 'json' || col.input_type === 'tags') && value) {
                    // 将对象/数组转换为JSON字符串用于编辑
                    if (typeof value === 'object') {
                      value = JSON.stringify(value, null, 2);
                    }
                  }
                  
                  formValues[col.key] = value;
                });
              }
              
              form.setFieldsValue(formValues);
              setModalVisible(true);
            }}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这条记录吗？"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    });

    return columns;
  };

  // 表单项渲染
  const renderFormItem = (col: TableColumn) => {
    const { key, title, input_type, nullable, primary_key, autoincrement } = col;
    
    if (primary_key && autoincrement) {
      return null; // 自增主键不显示在表单中
    }

    const rules = [];
    if (!nullable && !primary_key) {
      rules.push({ required: true, message: `请输入${title}` });
    }

    switch (input_type) {
      case 'number':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
          >
            <InputNumber style={{ width: '100%' }} placeholder={`请输入${title}`} />
          </Form.Item>
        );
      
      case 'boolean':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            valuePropName="checked"
            className={styles.formItem}
          >
            <Switch />
          </Form.Item>
        );
      
      case 'datetime':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
          >
            <DatePicker 
              showTime 
              style={{ width: '100%' }} 
              placeholder={`请选择${title}`}
            />
          </Form.Item>
        );
      
      case 'textarea':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
          >
            <TextArea rows={4} placeholder={`请输入${title}`} />
          </Form.Item>
        );
      
      case 'email':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={[
              ...rules,
              { type: 'email', message: '请输入有效的邮箱地址' }
            ]}
            className={styles.formItem}
          >
            <Input placeholder={`请输入${title}`} />
          </Form.Item>
        );
      
      case 'password':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
          >
            <Input.Password placeholder={`请输入${title}`} />
          </Form.Item>
        );
      
      case 'json':
      case 'tags':
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
            normalize={(value) => {
              if (typeof value === 'string') {
                try {
                  return JSON.parse(value);
                } catch {
                  return value;
                }
              }
              return value;
            }}
            getValueFromEvent={(e) => {
              const value = e.target.value;
              return value;
            }}
          >
            <TextArea 
              rows={3} 
              placeholder={`请输入${title}（JSON格式）`}
              onBlur={(e) => {
                const value = e.target.value;
                if (value) {
                  try {
                    JSON.parse(value);
                  } catch {
                    message.warning(`${title} 格式不正确，请输入有效的JSON格式`);
                  }
                }
              }}
            />
          </Form.Item>
        );
      
      default:
        return (
          <Form.Item
            key={key}
            name={key}
            label={title}
            rules={rules}
            className={styles.formItem}
          >
            <Input placeholder={`请输入${title}`} />
          </Form.Item>
        );
    }
  };

  // 表格操作菜单
  const actionMenuItems = [
    {
      key: 'import',
      icon: <UploadOutlined />,
      label: '导入数据',
      disabled: !selectedTable,
      onClick: () => setImportModalVisible(true),
    },
    {
      key: 'export',
      icon: <DownloadOutlined />,
      label: '导出数据',
      onClick: handleExport,
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'batchDelete',
      icon: <DeleteOutlined />,
      label: `批量删除 (${selectedRowKeys.length})`,
      danger: true,
      disabled: selectedRowKeys.length === 0,
      onClick: handleBatchDelete,
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'deleteTable',
      icon: <ExclamationCircleOutlined />,
      label: '删除整个表',
      danger: true,
      disabled: !selectedTable || ['users'].includes(selectedTable),
      onClick: handleDeleteTable,
    },
  ];

  const currentTable = availableTables.find(t => t.name === selectedTable);

  return (
    <PageContainer
      header={{
        title: '数据库管理',
        subTitle: '动态管理数据库表数据',
      }}
      className={styles.container}
    >
      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {Object.entries(adminStats).map(([tableName, stats]) => (
          <Col span={6} key={tableName}>
            <Card className={styles.statsCard}>
              <Statistic
                title={stats.title}
                value={stats.total}
                suffix="条"
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      {/* 主要内容 */}
      <Card className={styles.tableCard}>
        <div className={styles.tableHeader}>
          <Space>
            <Text strong>选择表:</Text>
            <Select
              value={selectedTable}
              onChange={setSelectedTable}
              className={styles.tableSelector}
              placeholder="请选择要管理的表"
            >
              {availableTables
                .sort((a, b) => a.title.localeCompare(b.title, 'zh-CN', { sensitivity: 'base' }))
                .map(table => (
                  <Option key={table.name} value={table.name}>
                    <Space>
                      <TableOutlined />
                      {table.title}
                      <Tag>{table.total_columns} 列</Tag>
                    </Space>
                  </Option>
                ))}
            </Select>
          </Space>

          <div className={styles.headerActions}>
            <Search
              placeholder="搜索数据..."
              allowClear
              enterButton={<SearchOutlined />}
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              onSearch={() => setPagination(prev => ({ ...prev, current: 1 }))}
              className={styles.searchBox}
            />
            
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => {
                setModalType('create');
                setEditingRecord(null);
                form.resetFields();
                setModalVisible(true);
              }}
            >
              新增
            </Button>
            
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                loadTableData();
                loadAdminStats();
              }}
              loading={loading}
            >
              刷新
            </Button>

            <Dropdown menu={{ items: actionMenuItems }} trigger={['click']}>
              <Button icon={<SettingOutlined />}>
                更多操作
              </Button>
            </Dropdown>
          </div>
        </div>

        {selectedTable && (
          <Table
            columns={getTableColumns()}
            dataSource={tableData}
            rowKey="_id"
            loading={loading}
            pagination={{
              ...pagination,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) => 
                `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
              onChange: (page, pageSize) => {
                setPagination(prev => ({ ...prev, current: page, pageSize }));
              },
            }}
            rowSelection={{
              selectedRowKeys,
              onChange: setSelectedRowKeys,
            }}
            onChange={(pagination, filters, sorter: any) => {
              if (sorter.field) {
                setSortField(sorter.field);
                setSortOrder(sorter.order === 'ascend' ? 'asc' : 'desc');
              }
            }}
            scroll={{ x: 'max-content' }}
          />
        )}
      </Card>

      {/* 新增/编辑模态框 */}
      <Modal
        title={modalType === 'create' ? '新增记录' : '编辑记录'}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
          setEditingRecord(null);
        }}
        onOk={() => form.submit()}
        width={600}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={modalType === 'create' ? handleCreate : handleUpdate}
        >
          <Row gutter={16}>
            {tableColumns && tableColumns.map(col => (
              <Col span={col.input_type === 'textarea' ? 24 : 12} key={col.key}>
                {renderFormItem(col)}
              </Col>
            ))}
          </Row>
        </Form>
      </Modal>

      {/* 数据导入模态框 */}
      <Modal
        title={`导入数据到 "${selectedTable}" 表`}
        open={importModalVisible}
        onCancel={resetImportModal}
        footer={[
          <Button key="cancel" onClick={resetImportModal}>
            关闭
          </Button>,
          ...(importResult?.success ? [
            <Button key="refresh" type="primary" onClick={() => {
              resetImportModal();
              loadTableData();
            }}>
              刷新表格
            </Button>
          ] : [])
        ]}
        width={700}
        destroyOnClose
      >
        <div style={{ padding: '16px 0' }}>
          {/* 文件格式说明 */}
          <Alert
            message="支持的文件格式"
            description={
              <div>
                <p><strong>JSON格式：</strong>支持对象数组 [{"{"}"field1": "value1", "field2": "value2"{"}"}, ...] 或单个对象，也支持JSONL格式（每行一个JSON对象）</p>
                <p><strong>CSV格式：</strong>第一行为列名，后续行为数据</p>
                <p><strong>注意：</strong>文件中的列名必须与数据库表结构匹配，系统会自动验证列名和数据类型。文件大小限制100MB。</p>
              </div>
            }
            type="info"
            showIcon
            style={{ marginBottom: 24 }}
          />

          {/* 文件上传区域 */}
          {!importLoading && !importResult && (
            <Dragger
              name="file"
              multiple={false}
              accept=".json,.csv"
              beforeUpload={(file) => {
                // 检查文件类型
                const isValidType = file.name.toLowerCase().endsWith('.json') || 
                                   file.name.toLowerCase().endsWith('.csv');
                if (!isValidType) {
                  message.error('只支持 JSON 和 CSV 格式文件');
                  return false;
                }

                // 检查文件大小（限制为100MB）
                const isLt100M = file.size / 1024 / 1024 < 100;
                if (!isLt100M) {
                  message.error('文件大小不能超过 100MB');
                  return false;
                }

                // 开始导入
                handleImport(file);
                return false; // 阻止默认上传行为
              }}
              style={{ padding: '20px' }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 JSON 和 CSV 格式，单个文件大小不超过 100MB
              </p>
            </Dragger>
          )}

          {/* 导入进度 */}
          {importLoading && (
            <div style={{ textAlign: 'center', padding: '40px 0' }}>
              <Progress
                type="circle"
                percent={importProgress}
                status={importProgress === 100 ? 'success' : 'active'}
                style={{ marginBottom: 16 }}
              />
              <p>正在导入数据，请稍候...</p>
            </div>
          )}

          {/* 导入结果 */}
          {importResult && (
            <div style={{ marginTop: 24 }}>
              <Alert
                message={importResult.success ? '导入成功' : '导入失败'}
                description={
                  <div>
                    <p>{importResult.message || importResult.error}</p>
                    {importResult.data && (
                      <div style={{ marginTop: 12 }}>
                        <p><strong>导入统计：</strong></p>
                        <ul>
                          <li>总记录数: {importResult.data.total_records}</li>
                          <li>成功导入: {importResult.data.inserted_count}</li>
                          <li>错误数量: {importResult.data.error_count || 0}</li>
                        </ul>
                        
                        {importResult.data.validation && (
                          <div style={{ marginTop: 12 }}>
                            <p><strong>列验证结果：</strong></p>
                            <ul>
                              <li>有效列: {importResult.data.validation.valid_columns?.join(', ') || '无'}</li>
                              {importResult.data.validation.extra_columns?.length > 0 && (
                                <li style={{ color: '#faad14' }}>
                                  多余列: {importResult.data.validation.extra_columns.join(', ')}
                                </li>
                              )}
                              {importResult.data.validation.missing_columns?.length > 0 && (
                                <li style={{ color: '#ff4d4f' }}>
                                  缺失列: {importResult.data.validation.missing_columns.join(', ')}
                                </li>
                              )}
                            </ul>
                          </div>
                        )}

                        {importResult.warnings && importResult.warnings.length > 0 && (
                          <div style={{ marginTop: 12 }}>
                            <p><strong>警告信息：</strong></p>
                            <ul>
                              {importResult.warnings.map((warning: string, index: number) => (
                                <li key={index} style={{ color: '#faad14' }}>{warning}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                }
                type={importResult.success ? 'success' : 'error'}
                showIcon
              />
            </div>
          )}
        </div>
      </Modal>
    </PageContainer>
  );
};

export default AdminManagement;
