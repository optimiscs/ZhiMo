import React, { useEffect, useState } from 'react';
import { history } from 'umi';
import './index.less';

const IndexPage: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuActive, setMenuActive] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const toggleMenu = () => {
    setMenuActive(!menuActive);
  };

  return (
    <div>
      {/* 导航栏 */}
      <header className={scrolled ? "scrolled" : ""}>
        <div className="container">
          <nav>
            <a href="#" className="logo">
              <div>
                <img src="/Logo.png" alt="智模万象Logo" width="48" height="48" />
              </div>
              <span className="logo-text">智模万象</span>
            </a>
            <ul className={`nav-links ${menuActive ? "active" : ""}`}>
              <li><a href="#features">核心功能</a></li>
              <li><a href="#technology">技术特点</a></li>
              <li><a href="#application">应用场景</a></li>
              <li><a href="#roadmap">发展路线</a></li>
            </ul>
            <div className={`nav-buttons ${menuActive ? "active" : ""}`}>
              <a onClick={() => history.push('/user/login?redirect=/dashboard')} className="btn btn-outline">登录</a>
              <a href="#contact" className="btn btn-primary">联系我们</a>
            </div>
            <div className="menu-toggle" onClick={toggleMenu}>☰</div>
          </nav>
        </div>
      </header>

      {/* 英雄区域 */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <h1 className="hero-title">多模态舆情<br />智能分析系统</h1>
            <p className="hero-subtitle">融合多模态大模型技术，构建专业舆情分析新标准，为用户提供实时、高效、精准的舆情洞察服务。</p>
            <div className="hero-buttons">
              <a href="#features" className="btn btn-primary">了解更多</a>
              <a onClick={() => history.push('/user/login?redirect=/dashboard')} className="btn btn-outline">开始使用</a>
            </div>
          </div>
        </div>
        <div className="hero-image">
          <img src="/chat.jpg" alt="智模万象舆情分析系统展示" />
        </div>
        <div className="hero-shape hero-shape-1"></div>
        <div className="hero-shape hero-shape-2"></div>
      </section>

      {/* 产品特点 */}
      <section className="section" id="features">
        <div className="container text-center">
          <h2 className="section-title">核心功能</h2>
          <p className="section-subtitle">智模万象秉承&quot;数据-模型-服务&quot;三层架构理念，构建从多源数据采集、智能分析到可视化服务的完整技术闭环。</p>

          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M12,12H19C18.47,16.11 15.72,19.78 12,20.92V12H5V6.3L12,3.19M12,1L3,5V11C3,16.55 6.84,21.74 12,23C17.16,21.74 21,16.55 21,11V5L12,1Z" />
                </svg>
              </div>
              <h3 className="feature-title">多源数据采集</h3>
              <p className="feature-description">依托Tornado、Request、Selenium等技术，实现百万级日处理能力，覆盖多平台、多结构化内容。</p>
            </div>

            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M12,17A2,2 0 0,0 14,15C14,13.89 13.1,13 12,13A2,2 0 0,0 10,15A2,2 0 0,0 12,17M18,8A2,2 0 0,1 20,10V20A2,2 0 0,1 18,22H6A2,2 0 0,1 4,20V10C4,8.89 4.9,8 6,8H7V6A5,5 0 0,1 12,1A5,5 0 0,1 17,6V8H18M12,3A3,3 0 0,0 9,6V8H15V6A3,3 0 0,0 12,3Z" />
                </svg>
              </div>
              <h3 className="feature-title">智能分析引擎</h3>
              <p className="feature-description">融合Transformer、BERT、CNN及Deepseek R1、Qwen2.5-VL等模型，驱动多模态分析与谣言识别。</p>
            </div>

            {/* 其他特点项可以按照相同的模式添加 */}
            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M10,17L6,13L7.41,11.59L10,14.17L16.59,7.58L18,9M12,1L3,5V11C3,16.55 6.84,21.74 12,23C17.16,21.74 21,16.55 21,11V5L12,1Z" />
                </svg>
              </div>
              <h3 className="feature-title">可视化服务</h3>
              <p className="feature-description">通过Vue、Echarts等技术构建交互式可视化分析大屏，提供直观的数据洞察体验。</p>
            </div>

            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M12,10.5A1.5,1.5 0 0,1 13.5,12A1.5,1.5 0 0,1 12,13.5A1.5,1.5 0 0,1 10.5,12A1.5,1.5 0 0,1 12,10.5M7.5,10.5A1.5,1.5 0 0,1 9,12A1.5,1.5 0 0,1 7.5,13.5A1.5,1.5 0 0,1 6,12A1.5,1.5 0 0,1 7.5,10.5M16.5,10.5A1.5,1.5 0 0,1 18,12A1.5,1.5 0 0,1 16.5,13.5A1.5,1.5 0 0,1 15,12A1.5,1.5 0 0,1 16.5,10.5Z" />
                </svg>
              </div>
              <h3 className="feature-title">多模态语义增强</h3>
              <p className="feature-description">融合多模态大模型，构建&quot;语义对齐-领域特征增强&quot;协同架构，全面理解文本、图像及视频内容。</p>
            </div>

            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M21,11C21,16.55 17.16,21.74 12,23C6.84,21.74 3,16.55 3,11V5L12,1L21,5V11M12,21C15.75,20 19,15.54 19,11.22V6.3L12,3.18L5,6.3V11.22C5,15.54 8.25,20 12,21Z" />
                </svg>
              </div>
              <h3 className="feature-title">谣言检测技术</h3>
              <p className="feature-description">基于Transformer Encoder构建核心模型，结合大模型赋能推理，提供高效精准的自动化内容审核能力。</p>
            </div>

            <div className="feature-item">
              <div className="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M13,9H18.5L13,3.5V9M6,2H14L20,8V20A2,2 0 0,1 18,22H6C4.89,22 4,21.1 4,20V4C4,2.89 4.89,2 6,2M15,18V16H6V18H15M18,14V12H6V14H18Z" />
                </svg>
              </div>
              <h3 className="feature-title">舆情策略生成</h3>
              <p className="feature-description">支持点对点对话，实现问答查询和多模态分析，自动评估舆情风险，生成危机公关策略。</p>
            </div>
          </div>
        </div>
      </section>

      {/* 技术优势部分 */}
      <section className="section" id="technology">
        <div className="container">
          <h2 className="section-title text-center">技术特点</h2>
          <p className="section-subtitle text-center">智模万象基于先进的多模态大模型技术打造，为舆情分析提供强大的技术支撑。</p>

          <div className="tech-container">
            <div className="tech-item">
              <div className="tech-image">
                <img src="/duotech.png" alt="多模态大模型技术展示" />
                <div className="tech-shape"></div>
              </div>
              <div className="tech-content">
                <h3 className="tech-title">多模态语义增强分析</h3>
                <p className="tech-description">针对舆情数据的多源异构特性，系统以多模态语义增强为核心技术亮点，融合Deepseek R1、Qwen2.5-VL等多模态大模型，构建&quot;语义对齐-领域特征增强&quot;协同架构。文本模态运用双向Transformer结合注意力机制进行深度语义理解；视觉模态采用改进型Vision Transformer与时序空间编码，解析图像及视频内容。</p>
                <div className="tech-stats">
                  <div className="stat">
                    <div className="stat-value">70%+</div>
                    <div className="stat-label">响应时间降低</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">秒级</div>
                    <div className="stat-label">舆情分析</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">百级节点</div>
                    <div className="stat-label">传播链路分析</div>
                  </div>
                </div>
              </div>
            </div>

            {/* 其他技术项可以按照相同的模式添加 */}
            <div className="tech-item">
              <div className="tech-content">
                <h3 className="tech-title">大模型微调优化</h3>
                <p className="tech-description">采用LoRA（Low-Rank Adaptation）低秩适配对预训练大模型进行高效微调。相较于全参数微调，LoRA仅调整极少部分参数（约0.003%），显著降低了模型优化的复杂度与资源需求。显存需求从全参数微调的1.2TB锐减至350GB，仅为原先的三分之一，推理速度提升25%以上，性能损失极小，控制在仅1-3%的范围内。</p>
                <div className="tech-stats">
                  <div className="stat">
                    <div className="stat-value">25%+</div>
                    <div className="stat-label">推理速度提升</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">53%</div>
                    <div className="stat-label">Token成本降低</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">3B</div>
                    <div className="stat-label">蒸馏模型</div>
                  </div>
                </div>
              </div>
              <div className="tech-image">
                <img src="/lora.png" alt="大模型微调技术展示" />
                <div className="tech-shape"></div>
              </div>
            </div>

            <div className="tech-item">
              <div className="tech-image">
                <img src="/yaoyan.png" alt="谣言检测技术展示" />
                <div className="tech-shape"></div>
              </div>
              <div className="tech-content">
                <h3 className="tech-title">基于Transformer的谣言检测</h3>
                <p className="tech-description">系统首先进行数据获取与预处理，包括读取数据、文本编码、填充/截断及标签转换。核心模型构建于Transformer Encoder之上，其结构包含位置编码层和全连接层。模型训练采用批次输入和Adam优化器，评估阶段关注F1 Score、准确率和召回率等关键指标，结合大模型进行赋能推理，提供高效精准的自动化内容审核能力。</p>
                <div className="tech-stats">
                  <div className="stat">
                    <div className="stat-value">5秒内</div>
                    <div className="stat-label">单点舆情分析</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">90%+</div>
                    <div className="stat-label">谣言检测准确率</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">实时</div>
                    <div className="stat-label">舆情预警</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 应用场景部分 */}
      <section className="section" id="application">
        <div className="container">
          <h2 className="section-title text-center">智慧舆情分析平台</h2>
          <p className="section-subtitle text-center">智模万象舆情分析系统为不同场景提供全面的舆情监测与分析解决方案。</p>

          <div className="app-container">
            {/* 智慧中枢场景 */}
            <div className="app-item">
              <div className="app-content">
                <h3 className="app-title">智慧中枢</h3>
                <p className="app-description">平台核心界面集成了热门词汇、地理热点定位、舆情监测、情感分析、新闻热度趋势展示及热门关键词提取等功能模块，全面掌握舆情动态。</p>
                <ul className="app-features">
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    热门词汇实时追踪，捕捉舆论焦点
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    地理热点可视化，洞察区域舆情分布
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    多维度情感分析，全面理解公众情绪
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    热度趋势监测，预测舆情发展方向
                  </li>
                </ul>
              </div>
              <div className="app-image">
                <img src="/map.jpg" alt="智慧中枢界面" />
                <div className="app-shape"></div>
              </div>
            </div>

            {/* 新闻列表场景 */}
            <div className="app-item">
              <div className="app-content">
                <h3 className="app-title">新闻列表</h3>
                <p className="app-description">提供图表联动和多维度筛选功能，支持复合查询、兼具数据持久化和热度可视化，使用户能高效洞察舆情动态。</p>
                <ul className="app-features">
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    多条件组合筛选，精准定位目标信息
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    事件卡片式展示，直观了解事件概况
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    数据持久化存储，支持历史追溯分析
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    热度指标可视化，快速识别重点事件
                  </li>
                </ul>
              </div>
              <div className="app-image">
                <img src="/list.jpg" alt="新闻列表界面" />
                <div className="app-shape"></div>
              </div>
            </div>

            {/* 多模态分析场景 */}
            <div className="app-item">
              <div className="app-content">
                <h3 className="app-title">多模态分析</h3>
                <p className="app-description">支持多平台数据采集，对图像、视频、文本等交互工具进行深度分析，全面理解各类信息载体所传递的舆情信号。</p>
                <ul className="app-features">
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    图像内容识别分析，洞察视觉舆情信息
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    视频关键帧分析，把握视频核心内容
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    实时情感分析，精准捕捉情绪变化
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    传播路径评估，追踪信息流动轨迹
                  </li>
                </ul>
              </div>
              <div className="app-image">
                <img src="/duomotai.jpg" alt="多模态分析界面" />
                <div className="app-shape"></div>
              </div>
            </div>

            {/* 舆情策略生成场景 */}
            <div className="app-item">
              <div className="app-content">
                <h3 className="app-title">舆情策略生成</h3>
                <p className="app-description">可开启点对点对话，实现问答查询和多模态分析，对热点事件的舆情走势进行分析，并自动生成相应的危机公关策略。</p>
                <ul className="app-features">
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    舆情风险评估，预警潜在危机
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    谣言信息识别，澄清不实信息
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    情感倾向分析，掌握公众态度
                  </li>
                  <li>
                    <svg viewBox="0 0 24 24">
                      <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
                    </svg>
                    智能策略生成，提供危机应对方案
                  </li>
                </ul>
              </div>
              <div className="app-image">
                <img src="/chat.jpg" alt="舆情策略生成界面" />
                <div className="app-shape"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 发展路线部分 */}
      <section className="section" id="roadmap">
        <div className="container">
          <h2 className="section-title text-center">发展路线</h2>
          <p className="section-subtitle text-center">智模万象从初期探索到智能化部署，不断迭代升级，持续提升舆情分析能力。</p>

          <div className="timeline">
            <div className="timeline-item">
              <div className="timeline-date">2024.12</div>
              <div className="timeline-content">
                <h3 className="timeline-title">构建舆情分析系统基础能力</h3>
                <p className="timeline-text">基于Vue+Flask+Celery架构，成功部署网络爬虫技术与可视化图表的热点追踪与分析系统。系统初步具备网络信息聚合能力，能自动识别潜在风险（如谣言），实现舆情数据的基础采集、存储与展示。此阶段奠定了平台的技术基础，为后续功能扩展与性能优化打下坚实基础。</p>
              </div>
            </div>

            <div className="timeline-item">
              <div className="timeline-date">2025.01</div>
              <div className="timeline-content">
                <h3 className="timeline-title">引入大语言模型与性能优化</h3>
                <p className="timeline-text">集成DeepSeek-R1大语言模型，显著提升文本分析深度与广度。通过多线程并行处理与优化的分布式架构，系统单节点可在5秒内完成基于七个主流平台数据的全面舆情分析，包括词云生成、情感立场分析、核心观点摘要和传播溯源。将LLM与传统机器学习模型结合，大幅提升分析准确率同时有效降低计算资源开销，响应时间降低70%。</p>
              </div>
            </div>

            <div className="timeline-item">
              <div className="timeline-date">2025.02</div>
              <div className="timeline-content">
                <h3 className="timeline-title">探索多模态分析与挑战识别</h3>
                <p className="timeline-text">整合Qwen2.5VL，开发多模态混合处理框架。该框架通过提取音频流、时间戳对齐与噪声抑制后转换为文本，利用DeepSeek定位关键帧，再由Qwen-VL进行视觉分析。此混合策略将多模态分析开销降低95%，大幅提升关键帧定位性能，实现成本与效率双重优化，Token使用成本降低53%，为下一阶段智能体开发奠定技术基础。</p>
              </div>
            </div>

            <div className="timeline-item">
              <div className="timeline-date">2025.04</div>
              <div className="timeline-content">
                <h3 className="timeline-title">舆情分析智能体</h3>
                <p className="timeline-text">基于新开源的DeepSeek v3开发&quot;舆情智能体&quot;应用。相较于DeepSeek-R1，新模型将Token成本降低约53%，端到端响应时间缩短近70%，极适合大规模高效部署。用户只需提供公司背景、垂直领域、舆情事件概要等关键信息，智能体便能自动整合内部知识库和实时网络数据，生成专业级舆情分析PDF报告，极大提升分析效率和报告质量，实现从数据采集到舆情策略生成的全流程智能化。</p>
              </div>
            </div>
          </div>

          <div className="future-plans">
            <h3 className="future-title">未来规划</h3>
            <div className="future-grid">
              <div className="future-item">
                <div className="future-icon">
                  <svg viewBox="0 0 24 24">
                    <path d="M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8M12,10A2,2 0 0,0 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12A2,2 0 0,0 12,10M10,22C9.75,22 9.54,21.82 9.5,21.58L9.13,18.93C8.5,18.68 7.96,18.34 7.44,17.94L4.95,18.95C4.73,19.03 4.46,18.95 4.34,18.73L2.34,15.27C2.21,15.05 2.27,14.78 2.46,14.63L4.57,12.97L4.5,12L4.57,11L2.46,9.37C2.27,9.22 2.21,8.95 2.34,8.73L4.34,5.27C4.46,5.05 4.73,4.96 4.95,5.05L7.44,6.05C7.96,5.66 8.5,5.32 9.13,5.07L9.5,2.42C9.54,2.18 9.75,2 10,2H14C14.25,2 14.46,2.18 14.5,2.42L14.87,5.07C15.5,5.32 16.04,5.66 16.56,6.05L19.05,5.05C19.27,4.96 19.54,5.05 19.66,5.27L21.66,8.73C21.79,8.95 21.73,9.22 21.54,9.37L19.43,11L19.5,12L19.43,13L21.54,14.63C21.73,14.78 21.79,15.05 21.66,15.27L19.66,18.73C19.54,18.95 19.27,19.04 19.05,18.95L16.56,17.95C16.04,18.34 15.5,18.68 14.87,18.93L14.5,21.58C14.46,21.82 14.25,22 14,22H10M11.25,4L10.88,6.61C9.68,6.86 8.62,7.5 7.85,8.39L5.44,7.35L4.69,8.65L6.8,10.2C6.4,11.37 6.4,12.64 6.8,13.8L4.68,15.36L5.43,16.66L7.86,15.62C8.63,16.5 9.68,17.14 10.87,17.38L11.24,20H12.76L13.13,17.39C14.32,17.14 15.37,16.5 16.14,15.62L18.57,16.66L19.32,15.36L17.2,13.81C17.6,12.64 17.6,11.37 17.2,10.2L19.31,8.65L18.56,7.35L16.15,8.39C15.38,7.5 14.32,6.86 13.12,6.62L12.75,4H11.25Z" />
                  </svg>
                </div>
                <h4>优化用户体验</h4>
                <p>持续优化用户界面设计，简化操作流程，打造直观便捷的交互体验。引入VR/AR等新兴技术，实现数据沉浸式大屏展示和3D数据可视化，增强数据呈现的直观性与吸引力。</p>
              </div>

              <div className="future-item">
                <div className="future-icon">
                  <svg viewBox="0 0 24 24">
                    <path d="M3,11H11V3H3M3,21H11V13H3M13,21H21V13H13M13,3V11H21V3" />
                  </svg>
                </div>
                <h4>扩大应用范围</h4>
                <p>通过开放API接口，与社交平台、市场营销、客户关系管理等系统深度集成，将舆情分析功能融入更多业务场景。针对金融、医疗、教育等不同行业特点，开发定制化解决方案，拓展应用边界。</p>
              </div>

              <div className="future-item">
                <div className="future-icon">
                  <svg viewBox="0 0 24 24">
                    <path d="M18,14A4,4 0 0,1 22,18A4,4 0 0,1 18,22A4,4 0 0,1 14,18A4,4 0 0,1 18,14M18,16A2,2 0 0,0 16,18A2,2 0 0,0 18,20A2,2 0 0,0 20,18A2,2 0 0,0 18,16M6,14A4,4 0 0,1 10,18A4,4 0 0,1 6,22A4,4 0 0,1 2,18A4,4 0 0,1 6,14M6,16A2,2 0 0,0 4,18A2,2 0 0,0 6,20A2,2 0 0,0 8,18A2,2 0 0,0 6,16M6,2A4,4 0 0,1 10,6A4,4 0 0,1 6,10A4,4 0 0,1 2,6A4,4 0 0,1 6,2M6,4A2,2 0 0,0 4,6A2,2 0 0,0 6,8A2,2 0 0,0 8,6A2,2 0 0,0 6,4M18,2A4,4 0 0,1 22,6A4,4 0 0,1 18,10A4,4 0 0,1 14,6A4,4 0 0,1 18,2M18,4A2,2 0 0,0 16,6A2,2 0 0,0 18,8A2,2 0 0,0 20,6A2,2 0 0,0 18,4Z" />
                  </svg>
                </div>
                <h4>国际市场拓展</h4>
                <p>积极拓展国际合作与推广，将系统推向全球市场，为国际用户提供优质服务。支持多语言舆情分析，实现跨文化、跨语言的舆情监测与分析能力，满足全球化企业的需求。</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 页脚 */}
      <footer>
        <div className="container">
          <div className="footer-content">
            <div className="footer-logo">
              <a href="#" className="logo">
                <div>
                  <img src="/Logo.png" alt="智模万象Logo" width="48" height="48" />
                </div>
                <span className="logo-text">智模万象</span>
              </a>
              <p className="footer-description">多模态舆情智能分析系统</p>
            </div>

            <div className="footer-links">
              <div className="footer-links-column">
                <h3>产品</h3>
                <ul>
                  <li><a href="#features">核心功能</a></li>
                  <li><a href="#technology">技术特点</a></li>
                  <li><a href="#application">应用场景</a></li>
                  <li><a href="#roadmap">发展路线</a></li>
                </ul>
              </div>

              <div className="footer-links-column">
                <h3>资源</h3>
                <ul>
                  <li><a href="#">技术白皮书</a></li>
                  <li><a href="#">用户手册</a></li>
                  <li><a href="#">API文档</a></li>
                  <li><a href="#">常见问题</a></li>
                </ul>
              </div>

              <div className="footer-links-column">
                <h3>关于</h3>
                <ul>
                  <li><a href="#">关于我们</a></li>
                  <li><a href="#">新闻动态</a></li>
                  <li><a href="#">加入我们</a></li>
                  <li><a href="#contact">联系我们</a></li>
                </ul>
              </div>

              <div className="footer-links-column">
                <h3>联系方式</h3>
                <p><strong>地址：</strong>北京市海淀区中关村南大街5号</p>
                <p><strong>电话：</strong>010-12345678</p>
                <p><strong>邮箱：</strong>contact@zhimowx.com</p>
                <div className="social-icons">
                  <a href="#" className="social-icon">
                    <svg viewBox="0 0 24 24" width="24" height="24">
                      <path fill="currentColor" d="M8.58,17.25L9.5,13.36L6.5,10.78L10.45,10.41L12,6.8L13.55,10.45L17.5,10.78L14.5,13.36L15.42,17.25L12,15.19L8.58,17.25M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2Z" />
                    </svg>
                  </a>
                  <a href="#" className="social-icon">
                    <svg viewBox="0 0 24 24" width="24" height="24">
                      <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12C2,16.42 4.87,20.17 8.84,21.5C9.34,21.58 9.5,21.27 9.5,21C9.5,20.77 9.5,20.14 9.5,19.31C6.73,19.91 6.14,17.97 6.14,17.97C5.68,16.81 5.03,16.5 5.03,16.5C4.12,15.88 5.1,15.9 5.1,15.9C6.1,15.97 6.63,16.93 6.63,16.93C7.5,18.45 8.97,18 9.54,17.76C9.63,17.11 9.89,16.67 10.17,16.42C7.95,16.17 5.62,15.31 5.62,11.5C5.62,10.39 6,9.5 6.65,8.79C6.55,8.54 6.2,7.5 6.75,6.15C6.75,6.15 7.59,5.88 9.5,7.17C10.29,6.95 11.15,6.84 12,6.84C12.85,6.84 13.71,6.95 14.5,7.17C16.41,5.88 17.25,6.15 17.25,6.15C17.8,7.5 17.45,8.54 17.35,8.79C18,9.5 18.38,10.39 18.38,11.5C18.38,15.32 16.04,16.16 13.81,16.41C14.17,16.72 14.5,17.33 14.5,18.26C14.5,19.6 14.5,20.68 14.5,21C14.5,21.27 14.66,21.59 15.17,21.5C19.14,20.16 22,16.42 22,12A10,10 0 0,0 12,2Z" />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© 2025 智模万象 - 多模态舆情智能分析系统. 保留所有权利.</p>
            <div className="footer-bottom-links">
              <a href="#">隐私政策</a>
              <a href="#">使用条款</a>
              <a href="#">法律声明</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default IndexPage;
