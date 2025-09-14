import { request } from '@umijs/max';
import type {
  ApiResponse,
  TableInfo,
  TableSchemaResponse,
  TableDataResponse,
  TableDataParams,
  CreateRecordRequest,
  UpdateRecordRequest,
  ExportDataResponse,
  AdminStats,
  AdminService
} from './types';

const API_BASE = '/api/v1/admin';

/**
 * 数据库管理服务类
 * 封装所有与管理API相关的请求
 */
class AdminManagementService implements AdminService {
  
  /**
   * 获取所有可用的表列表
   */
  async getTables(): Promise<ApiResponse<TableInfo[]>> {
    try {
      const response = await request(`${API_BASE}/tables`, {
        method: 'GET',
      });
      return response;
    } catch (error) {
      console.error('获取表列表失败:', error);
      return {
        success: false,
        error: '获取表列表失败，请检查网络连接'
      };
    }
  }

  /**
   * 获取指定表的结构信息
   */
  async getTableSchema(tableName: string): Promise<ApiResponse<TableSchemaResponse>> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/schema`, {
        method: 'GET',
      });
      return response;
    } catch (error) {
      console.error('获取表结构失败:', error);
      return {
        success: false,
        error: '获取表结构失败'
      };
    }
  }

  /**
   * 获取表数据（支持分页、搜索、排序）
   */
  async getTableData(tableName: string, params: TableDataParams = {}): Promise<ApiResponse<TableDataResponse>> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/data`, {
        method: 'GET',
        params: {
          page: params.page || 1,
          pageSize: params.pageSize || 20,
          search: params.search || '',
          sortField: params.sortField || '',
          sortOrder: params.sortOrder || 'asc',
        },
      });
      return response;
    } catch (error) {
      console.error('获取表数据失败:', error);
      return {
        success: false,
        error: '获取表数据失败'
      };
    }
  }

  /**
   * 创建新记录
   */
  async createRecord(tableName: string, data: CreateRecordRequest): Promise<ApiResponse> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/data`, {
        method: 'POST',
        data,
      });
      return response;
    } catch (error) {
      console.error('创建记录失败:', error);
      return {
        success: false,
        error: '创建记录失败'
      };
    }
  }

  /**
   * 更新记录
   */
  async updateRecord(tableName: string, id: number, data: UpdateRecordRequest): Promise<ApiResponse> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/data/${id}`, {
        method: 'PUT',
        data,
      });
      return response;
    } catch (error) {
      console.error('更新记录失败:', error);
      return {
        success: false,
        error: '更新记录失败'
      };
    }
  }

  /**
   * 删除单条记录
   */
  async deleteRecord(tableName: string, id: number): Promise<ApiResponse> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/data/${id}`, {
        method: 'DELETE',
      });
      return response;
    } catch (error) {
      console.error('删除记录失败:', error);
      return {
        success: false,
        error: '删除记录失败'
      };
    }
  }

  /**
   * 批量删除记录
   */
  async batchDeleteRecords(tableName: string, ids: (string | number)[]): Promise<ApiResponse> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/data/batch`, {
        method: 'DELETE',
        data: { ids },
      });
      return response;
    } catch (error) {
      console.error('批量删除失败:', error);
      return {
        success: false,
        error: '批量删除失败'
      };
    }
  }

  /**
   * 导出表数据
   */
  async exportTableData(tableName: string): Promise<ApiResponse<ExportDataResponse>> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}/export`, {
        method: 'GET',
      });
      return response;
    } catch (error) {
      console.error('导出数据失败:', error);
      return {
        success: false,
        error: '导出数据失败'
      };
    }
  }

  /**
   * 获取管理统计信息
   */
  async getStats(): Promise<ApiResponse<AdminStats>> {
    try {
      const response = await request(`${API_BASE}/stats`, {
        method: 'GET',
      });
      return response;
    } catch (error) {
      console.error('获取统计信息失败:', error);
      return {
        success: false,
        error: '获取统计信息失败'
      };
    }
  }

  /**
   * 删除表/集合
   */
  async deleteTable(tableName: string): Promise<ApiResponse> {
    try {
      const response = await request(`${API_BASE}/tables/${tableName}`, {
        method: 'DELETE',
      });
      return response;
    } catch (error) {
      console.error('删除表失败:', error);
      return {
        success: false,
        error: '删除表失败'
      };
    }
  }

  /**
   * 导入数据到表（支持JSON和CSV格式）
   */
  async importTableData(tableName: string, file: File): Promise<ApiResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await request(`${API_BASE}/tables/${tableName}/import`, {
        method: 'POST',
        data: formData,
        headers: {
          // 不设置Content-Type，让浏览器自动设置multipart/form-data边界
        },
      });
      return response;
    } catch (error) {
      console.error('导入数据失败:', error);
      return {
        success: false,
        error: '导入数据失败'
      };
    }
  }
}

// 创建服务实例
const adminService = new AdminManagementService();

// 导出服务实例和类
export default adminService;
export { AdminManagementService };

// 导出便捷的API函数
export const {
  getTables,
  getTableSchema,
  getTableData,
  createRecord,
  updateRecord,
  deleteRecord,
  batchDeleteRecords,
  exportTableData,
  getStats,
  deleteTable,
  importTableData
} = adminService;

/**
 * 下载文件的工具函数
 */
export const downloadFile = (content: string, filename: string, contentType: string = 'text/csv') => {
  const blob = new Blob([content], { type: `${contentType};charset=utf-8;` });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // 清理URL对象
  URL.revokeObjectURL(url);
};

/**
 * 将表格数据转换为CSV格式
 */
export const convertToCSV = (data: any[], columns: string[]): string => {
  const csvRows = [];
  
  // 添加表头
  csvRows.push(columns.map(col => `"${col}"`).join(','));
  
  // 添加数据行
  for (const row of data) {
    const values = columns.map(col => {
      const value = row[col];
      if (value === null || value === undefined) {
        return '""';
      }
      // 处理包含逗号、引号或换行符的值
      const stringValue = String(value);
      if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
        return `"${stringValue.replace(/"/g, '""')}"`;
      }
      return `"${stringValue}"`;
    });
    csvRows.push(values.join(','));
  }
  
  return csvRows.join('\n');
};

/**
 * 格式化错误消息
 */
export const formatErrorMessage = (error: any): string => {
  if (typeof error === 'string') {
    return error;
  }
  
  if (error?.message) {
    return error.message;
  }
  
  if (error?.error) {
    return error.error;
  }
  
  return '操作失败，请重试';
};

/**
 * 验证表单数据
 */
export const validateFormData = (data: any, columns: any[]): string[] => {
  const errors: string[] = [];
  
  for (const column of columns) {
    const value = data[column.key];
    
    // 检查必填字段
    if (!column.nullable && !column.autoincrement && (value === null || value === undefined || value === '')) {
      errors.push(`${column.title} 是必填字段`);
    }
    
    // 检查邮箱格式
    if (column.input_type === 'email' && value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      errors.push(`${column.title} 格式不正确`);
    }
    
    // 检查数字类型
    if (column.input_type === 'number' && value && isNaN(Number(value))) {
      errors.push(`${column.title} 必须是有效的数字`);
    }
  }
  
  return errors;
};

/**
 * 防抖函数
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * 格式化日期时间
 */
export const formatDateTime = (dateTime: string | Date): string => {
  if (!dateTime) return '-';
  
  const date = new Date(dateTime);
  if (isNaN(date.getTime())) return '-';
  
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

/**
 * 格式化文件大小
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};
