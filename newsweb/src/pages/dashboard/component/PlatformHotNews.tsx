import { ProList } from '@ant-design/pro-components';
import { Skeleton, Typography, Space, Flex, Badge, Dropdown, Empty } from 'antd';
import { useState, useEffect } from 'react';
import {
  FireOutlined,
  RiseOutlined,
  SyncOutlined,
  CaretDownOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

// 平台头像映射
const platformAvatars: { [key: string]: string } = {
  知乎热榜: './icons/知乎.svg',
  哔哩哔哩: './icons/哔哩哔哩.svg',
  今日头条: './icons/今日头条.svg',
  抖音: './icons/抖音.svg',
  微博: './icons/微博.svg',
  百度热点: './icons/百度.svg',
};

// 颜色映射：为前三名设置不同的颜色
const rankColors = ['#f5222d', '#fa8c16', '#faad14'];

// 支持的平台列表
const SUPPORTED_PLATFORMS = ['微博', '知乎热榜', '百度热点', '今日头条', '抖音', '哔哩哔哩'];

// 定义数据结构类型
interface HotNewsItem {
  key: string;
  name: string;
  image: string;
  desc: string;
  hotValue: number;
  url: string;
  rank: number;
}

// 平台映射表 - 将中文平台名称映射到API端点
const PLATFORM_API_MAPPING = {
  '微博': 'weibo',
  '百度热点': 'baidu',
  '抖音': 'douyin',
  '哔哩哔哩': 'bilibili',
  '今日头条': 'toutiao',
  '知乎热榜': 'zhihu'
};

// API端点配置 - 使用后端代理
const API_ENDPOINTS = {
  weibo: '/api/proxy/hotnews/weibo',
  baidu: '/api/proxy/hotnews/baidu',
  douyin: '/api/proxy/hotnews/douyin',
  bilibili: '/api/proxy/hotnews/bilibili',
  toutiao: '/api/proxy/hotnews/toutiao',
  zhihu: '/api/proxy/hotnews/zhihu'
};

const PlatformHotNews: React.FC = () => {
  const [platformData, setPlatformData] = useState<{[key: string]: HotNewsItem[]}>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activePlatform, setActivePlatform] = useState<string>(SUPPORTED_PLATFORMS[0]);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [dataSource, setDataSource] = useState<string>('imsyy'); // 记录数据来源

  // 从imsyy API获取单个平台数据
  const fetchPlatformData = async (platformKey: string): Promise<any> => {
    const apiKey = PLATFORM_API_MAPPING[platformKey as keyof typeof PLATFORM_API_MAPPING];
    if (!apiKey || !API_ENDPOINTS[apiKey as keyof typeof API_ENDPOINTS]) {
      console.warn(`未找到平台 ${platformKey} 的API配置`);
      return null;
    }

    try {
      const response = await fetch(API_ENDPOINTS[apiKey as keyof typeof API_ENDPOINTS], {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.code === 200 && data.data) {
        return {
          name: platformKey,
          subtitle: "热榜",
          update_time: new Date().toLocaleString(),
          data: data.data.map((item: any, index: number) => ({
            type: `${apiKey}Hot`,
            title: item.title,
            hot: item.hot, // 保持原始数字格式
            url: item.url,
            mobil_url: item.mobileUrl || item.url,
            index: index + 1
          }))
        };
      } else {
        throw new Error(`API返回错误: ${data.code}`);
      }
    } catch (error) {
      console.error(`获取 ${platformKey} 数据失败:`, error);
      return null;
    }
  };

  // 获取单个平台数据的函数
  const fetchSinglePlatformData = async (platformKey: string): Promise<HotNewsItem[]> => {
    const platformData = await fetchPlatformData(platformKey);
    
    if (platformData && platformData.data && platformData.data.length > 0) {
      return platformData.data
        .filter((news: any) => news.title && news.url)
        .map((news: any, index: number) => {
          // 处理热度值 - 现在 news.hot 是数字类型
          const hotValue = typeof news.hot === 'number' ? news.hot : 0;
          const hotText = formatHotValue(hotValue);

          return {
            key: `${platformKey}-${index}`,
            name: news.title,
            image: platformAvatars[platformKey],
            desc: hotText,
            hotValue: hotValue,
            url: news.url,
            rank: index + 1
          };
        })
        .slice(0, 8); // 取前8条
    }
    
    return [];
  };

  // 格式化热度值
  const formatHotValue = (hotValue: number): string => {
    if (hotValue >= 100000000) {
      return `${(hotValue / 100000000).toFixed(1)}亿`;
    } else if (hotValue >= 10000) {
      return `${(hotValue / 10000).toFixed(1)}万`;
    } else {
      return hotValue.toString();
    }
  };

  // 从imsyy API获取所有平台数据
  const fetchImsyyData = async (): Promise<any> => {
    const allPlatformData = [];
    
    for (const platform of SUPPORTED_PLATFORMS) {
      const platformData = await fetchPlatformData(platform);
      if (platformData) {
        allPlatformData.push(platformData);
      }
    }

    if (allPlatformData.length > 0) {
      return {
        success: true,
        data: allPlatformData
      };
    } else {
      throw new Error('所有平台数据获取失败');
    }
  };

  // 从vvhan API获取数据（备用方案）
  const fetchVvhanData = async (): Promise<any> => {
    const response = await fetch('https://api.vvhan.com/api/hotlist/all');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      let data;
      let source = 'imsyy';

      // 首先尝试使用imsyy API
      try {
        console.log('尝试从imsyy.top API获取数据...');
        data = await fetchImsyyData();
        setDataSource('imsyy');
        console.log('成功从imsyy.top API获取数据');
      } catch (imsyyError) {
        console.warn('imsyy API获取失败，尝试备用API:', imsyyError);
        
        // 备用方案：使用vvhan API
        try {
          console.log('尝试从vvhan API获取数据...');
          data = await fetchVvhanData();
          setDataSource('vvhan');
          console.log('成功从vvhan API获取数据');
        } catch (vvhanError) {
          throw new Error(`所有API都失败: imsyy=${imsyyError}, vvhan=${vvhanError}`);
        }
      }

      if (data.success && data.data) {
        const allPlatformData: {[key: string]: HotNewsItem[]} = {};

        // 处理每个平台的数据
        SUPPORTED_PLATFORMS.forEach((platform) => {
          const platformSource = data.data.find(
            (item: any) => item.name === platform,
          );

          if (platformSource && platformSource.data && platformSource.data.length > 0) {
            const validNews = platformSource.data
              .filter((news: any) => news.title && news.url)
              .map((news: any, index: number) => {
                // 处理热度值 - 现在 news.hot 是数字类型
                const hotValue = typeof news.hot === 'number' ? news.hot : 0;
                const hotText = formatHotValue(hotValue);

                return {
                  key: `${platform}-${index}`,
                  name: news.title,
                  image: platformAvatars[platform],
                  desc: hotText,
                  hotValue: hotValue,
                  url: news.url,
                  rank: index + 1
                };
              })
              .slice(0, 8); // 取前8条

            allPlatformData[platform] = validNews;
          } else {
            allPlatformData[platform] = [];
          }
        });

        setPlatformData(allPlatformData);
      } else {
        throw new Error(data.message || 'Failed to fetch valid data structure');
      }
    } catch (err: any) {
      console.error('获取热门列表数据失败:', err);
      setError(err.message || '加载数据时发生错误');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // 生成下拉菜单项
  const platformMenuItems = SUPPORTED_PLATFORMS.map(platform => ({
    key: platform,
    label: (
      <Space>
        <img
          src={platformAvatars[platform]}
          alt={platform}
          style={{ width: 14, height: 14 }}
        />
        <span>{platform.replace('热榜', '')}</span>
      </Space>
    ),
  }));

  // 处理下拉菜单选择
  const handleMenuClick = async ({ key }: { key: string }) => {
    setActivePlatform(key);
    setRefreshing(true);
    
    try {
      // 只获取当前选中平台的数据
      const platformData = await fetchSinglePlatformData(key);
      setPlatformData(prev => ({
        ...prev,
        [key]: platformData
      }));
    } catch (error) {
      console.error(`获取 ${key} 平台数据失败:`, error);
      setError(`获取 ${key} 数据失败`);
    } finally {
      setRefreshing(false);
    }
  };

  // 获取当前平台的简短名称
  const getShortPlatformName = (platform: string) => {
    return platform.replace('热榜', '');
  };

  if (loading && !refreshing) {
    return (
      <div style={{ padding: 16, height: '100%' }}>
        <Skeleton active paragraph={{ rows: 10 }} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={<Text type="danger">加载失败: {error}</Text>}
        />
      </div>
    );
  }

  const activeData = platformData[activePlatform] || [];

  // 即使没有数据也显示顶部栏和空状态
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 标题和平台选择 */}
      <Flex justify="space-between" align="center" style={{ padding: '8px 12px', borderBottom: '1px solid #f0f0f0' }}>
        <Space>
          <FireOutlined style={{ color: '#ff4d4f', fontSize: 16 }} />
          <Text strong style={{ fontSize: '16px' }}>热门榜单</Text>
          {refreshing && <SyncOutlined spin style={{ color: '#1890ff', marginLeft: 8 }} />}
     
        </Space>

        <Dropdown
          menu={{
            items: platformMenuItems,
            onClick: handleMenuClick,
            selectedKeys: [activePlatform]
          }}
          trigger={['click']}
        >
          <Space style={{ cursor: 'pointer' }}>
            <img
              src={platformAvatars[activePlatform]}
              alt={activePlatform}
              style={{ width: 16, height: 16 }}
            />
            <Text strong>{getShortPlatformName(activePlatform)}</Text>
            <CaretDownOutlined />
          </Space>
        </Dropdown>
      </Flex>

      {/* 内容区域 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '4px 0' }}>
        {activeData.length === 0 ? (
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            padding: '20px'
          }}>
            <Empty description="暂无热门新闻数据" />
          </div>
        ) : (
          <ProList<HotNewsItem>
            rowKey="key"
            dataSource={activeData}
            split={true}
            metas={{
              title: {
                dataIndex: 'name',
                render: (dom, entity) => (
                  <a
                    href={entity.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'block',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      color: '#000000',
                      textDecoration: 'none',
                      fontSize: '13px',
                      maxWidth: '220px',
                      transition: 'color 0.3s'
                    }}
                    className="hot-news-link"
                  >
                    <Badge
                      count={entity.rank}
                      style={{
                        backgroundColor: entity.rank <= 3 ? rankColors[entity.rank - 1] : '#ccc',
                        marginRight: 8
                      }}
                      size="small"
                    />
                    {dom}
                  </a>
                ),
              },

              subTitle: {
                dataIndex: 'desc',
                search: false,
                render: (_, entity) => (
                  <Flex align="center">
                    <Text style={{ fontSize: '11px', color: '#ff7a45' }}>
                      {entity.desc}
                    </Text>
                    {entity.rank <= 3 && <RiseOutlined style={{ fontSize: '11px', color: '#ff4d4f', marginLeft: 4 }} />}
                  </Flex>
                ),
              },
            }}
            pagination={false}
            style={{
              backgroundColor: '#ffffff',
            }}
            cardProps={{
              className: 'hot-news-item'
            }}
            loading={refreshing}
          />
        )}
      </div>

      <style>
        {`
          .hot-news-item:hover {
            background-color:rgb(255, 255, 255);
          }
          .hot-news-link:hover {
            color: #1890ff !important;
          }
          .platform-list .ant-pro-list-row-header {
            padding: 0 !important;
          }
        `}
      </style>
    </div>
  );


};

export default PlatformHotNews;
