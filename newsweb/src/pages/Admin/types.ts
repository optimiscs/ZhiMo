// 数据库管理相关的类型定义

export interface TableColumn {
  key: string;
  title: string;
  dataIndex: string;
  type: string;
  nullable: boolean;
  primary_key: boolean;
  autoincrement: boolean;
  input_type: 'text' | 'email' | 'password' | 'textarea' | 'number' | 'boolean' | 'datetime';
  default?: string;
}

export interface TableInfo {
  name: string;
  title: string;
  columns: TableColumn[];
  total_columns: number;
}

export interface TableData {
  id: number;
  [key: string]: any;
}

export interface AdminStats {
  [tableName: string]: {
    total: number;
    title: string;
  };
}

export interface PaginationInfo {
  current: number;
  pageSize: number;
  total: number;
  totalPages?: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  pagination?: PaginationInfo;
}

export interface TableDataResponse {
  data: TableData[];
  pagination: PaginationInfo;
}

export interface TableSchemaResponse {
  table_name: string;
  columns: TableColumn[];
}

export interface ExportDataResponse {
  table_name: string;
  columns: string[];
  rows: TableData[];
  total: number;
}

export interface CreateRecordRequest {
  [key: string]: any;
}

export interface UpdateRecordRequest {
  [key: string]: any;
}

export interface BatchDeleteRequest {
  ids: (string | number)[];
}

export interface TableDataParams {
  page?: number;
  pageSize?: number;
  search?: string;
  sortField?: string;
  sortOrder?: 'asc' | 'desc';
}

// 表单相关类型
export interface FormFieldConfig {
  key: string;
  label: string;
  type: TableColumn['input_type'];
  required: boolean;
  rules?: any[];
  placeholder?: string;
  options?: { label: string; value: any }[];
}

// 操作类型枚举
export enum ModalType {
  CREATE = 'create',
  EDIT = 'edit',
}

export enum SortOrder {
  ASC = 'asc',
  DESC = 'desc',
}

// 组件Props类型
export interface AdminManagementProps {
  className?: string;
}

export interface DynamicTableProps {
  columns: TableColumn[];
  dataSource: TableData[];
  loading?: boolean;
  pagination: PaginationInfo;
  selectedRowKeys: React.Key[];
  onSelectionChange: (keys: React.Key[]) => void;
  onPaginationChange: (page: number, pageSize: number) => void;
  onSortChange: (field: string, order: SortOrder) => void;
  onEdit: (record: TableData) => void;
  onDelete: (recordId: number) => void;
}

export interface DynamicFormProps {
  columns: TableColumn[];
  initialValues?: TableData;
  onSubmit: (values: any) => void;
  loading?: boolean;
}

export interface StatsCardsProps {
  stats: AdminStats;
  loading?: boolean;
}

export interface TableSelectorProps {
  tables: TableInfo[];
  selectedTable: string;
  onTableChange: (tableName: string) => void;
  loading?: boolean;
}

// 错误处理相关
export interface ErrorInfo {
  message: string;
  code?: string | number;
  details?: any;
}

export interface ValidationError {
  field: string;
  message: string;
}

// 服务层接口
export interface AdminService {
  getTables(): Promise<ApiResponse<TableInfo[]>>;
  getTableSchema(tableName: string): Promise<ApiResponse<TableSchemaResponse>>;
  getTableData(tableName: string, params: TableDataParams): Promise<ApiResponse<TableDataResponse>>;
  createRecord(tableName: string, data: CreateRecordRequest): Promise<ApiResponse<TableData>>;
  updateRecord(tableName: string, id: number, data: UpdateRecordRequest): Promise<ApiResponse<TableData>>;
  deleteRecord(tableName: string, id: number): Promise<ApiResponse>;
  batchDeleteRecords(tableName: string, ids: (string | number)[]): Promise<ApiResponse>;
  exportTableData(tableName: string): Promise<ApiResponse<ExportDataResponse>>;
  getStats(): Promise<ApiResponse<AdminStats>>;
}

// Hook相关类型
export interface UseAdminManagementState {
  loading: boolean;
  selectedTable: string;
  availableTables: TableInfo[];
  tableData: TableData[];
  tableColumns: TableColumn[];
  adminStats: AdminStats;
  pagination: PaginationInfo;
  searchText: string;
  sortField: string;
  sortOrder: SortOrder;
  selectedRowKeys: React.Key[];
  modalVisible: boolean;
  modalType: ModalType;
  editingRecord: TableData | null;
}

export interface UseAdminManagementActions {
  setSelectedTable: (table: string) => void;
  setPagination: (pagination: Partial<PaginationInfo>) => void;
  setSearchText: (text: string) => void;
  setSorting: (field: string, order: SortOrder) => void;
  setSelectedRowKeys: (keys: React.Key[]) => void;
  openModal: (type: ModalType, record?: TableData) => void;
  closeModal: () => void;
  loadTables: () => Promise<void>;
  loadTableData: () => Promise<void>;
  loadStats: () => Promise<void>;
  handleCreate: (values: any) => Promise<void>;
  handleUpdate: (values: any) => Promise<void>;
  handleDelete: (id: number) => Promise<void>;
  handleBatchDelete: () => Promise<void>;
  handleExport: () => Promise<void>;
  refresh: () => Promise<void>;
}

// 常量定义
export const DEFAULT_PAGE_SIZE = 20;
export const DEFAULT_PAGINATION: PaginationInfo = {
  current: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  total: 0,
};

export const INPUT_TYPE_MAP: Record<string, TableColumn['input_type']> = {
  'VARCHAR': 'text',
  'TEXT': 'textarea',
  'INTEGER': 'number',
  'NUMERIC': 'number',
  'BOOLEAN': 'boolean',
  'DATETIME': 'datetime',
  'DATE': 'datetime',
};

export const VALIDATION_RULES = {
  required: { required: true, message: '此字段为必填项' },
  email: { type: 'email' as const, message: '请输入有效的邮箱地址' },
  number: { type: 'number' as const, message: '请输入有效的数字' },
  minLength: (min: number) => ({ min, message: `至少输入${min}个字符` }),
  maxLength: (max: number) => ({ max, message: `最多输入${max}个字符` }),
};

// 工具类型
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// 扩展Ant Design的表格列类型
export interface ExtendedColumnType extends Omit<any, 'dataIndex'> {
  dataIndex: string;
  inputType?: TableColumn['input_type'];
  editable?: boolean;
  searchable?: boolean;
  exportable?: boolean;
}
