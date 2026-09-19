# 刘江丽

**政治面貌**：中共党员  
**邮箱**：liujiangli0301@163.com  
**电话**：18722034165

---

## 个人总结
具备扎实的算法功底与端到端工程化落地能力  
- **计算机视觉**：深入掌握目标检测、多目标跟踪与扩散模型，具备从模型选型、数据构建到性能优化的全链路研发经验；自研 RGB-D跨模态融合模块（DG-CSPStage/DGA/DA-FPN）实现检测精度显著突破。
- **语音识别与生成**：熟练掌握 Wav2Vec2/Conformer架构及 CTC+LM联合解码，能够针对低资源方言场景设计热词注入与动态适配策略；熟悉视频插帧扩散模型（RIFE）原理与可控调优。
- **强化学习**：系统学习 PPO、GRPO等主流算法，具备将 RL思想应用于实际业务场景的建模能力。
- **多模态与 Agent系统**：深入多模态大模型原理，独立构建基于 ReAct+ Function Calling的 LLM Agent系统，集成 Qwen-VL 多模态校验与 RAG检索增强（BGE Embedding+ ChromaDB）；熟悉 LangChain架构与知识库构建方法。
- **工程部署**：掌握 ONNX/TensorRT/NPU异构平台模型转换与加速优化，具备 GPU/NPU多硬件环境下的全流程部署经验，熟练运用 Docker容器化与 AI Coding工具赋能开发提效。

---

## 教育背景

**北京化工大学** | 计算机技术 | 硕士 | 2024.8 - 2027.6
- 特等奖学金*1、万集科技奖学金*1
- 《LogiTrack: Effective Warehouse Multi-Object Tracking via Adaptive Orthogonal Attention and Motion-Compensated OC-SORT》 CAC在投
- 《DSG: Dual-Semantic Guidance from LLM to Token Distillation for Few-Shot Incremental Learning》 ICASSP 2026已录用

**天津科技大学** | 数字媒体技术 | 本科 | 2019.9 - 2023.6
- 一等奖学金*2、创新创业奖学金*2、校三好学生*1、校先进个人*1

---

## 实习工作经历

### 三六零安全科技股份有限公司 | 垂直业务搜索部初级算法工程师 | 2025.11 - 2026.5
- **线上违规图像分类系统**
  - **项目简介**：主导线上违规图像分类（52个细分类别、拆分为 3个子模型）与违规目标检测模块研发，独立完成基线调研、数据构建、模型优化与生产部署全链路。
  - **技术栈**：Python / PyTorch / Focal Loss / Self-Supervised Learning / ONNX / Triton Server
  - **技术亮点**：系统开展难负样本挖掘方法对比实验，涵盖三种主流策略——Focal Loss+监督对比损失（SupCon）+类别权重动态调配、Focal Loss+类别权重、OHEM+监督对比损失+类别权重——通过消融实验综合评估各方案在困难样本特征空间区分度与类别平衡上的表现，最终选定方案 1并结合自监督学习策略实现模型 Precision提升 3.1%、Recall提升 1.9%，误召回从 392例降至 168例；基于大模型自动化标注管线实现低成本高效数据生产；模型后处理接入大模型纠错模块进行二次校验。
- **语音识别数据处理与标注流水线**
  - **项目简介**：独立完成从原始音频数据到结构化标注数据全流程流水线，覆盖音频切割、ASR识别、JSONL格式化、多进程并行处理与人工质检环节。
  - **技术栈**：Python / SenseVoice / JSONL / Multi-Process / Shell / nsys
  - **技术亮点**：从 JSONL提取时间戳索引对应音频路径，支持.wav后缀与无后缀双模式映射，以 bytes流形式切割避免中间文件写入；设计多进程并发方案（单卡 30并发），5卡并行处理 160个 JSONL文件；切割音频 170万+条，生成 JSONL条目 700万+条。
- **图片特效处理模型（Triton推理服务）**
  - **项目简介**：独立完成图片特效处理模型的调研、开发与部署上线，支持 7种处理模式，封装为 Triton Inference Server推理服务供业务调用。
  - **技术栈**：Python / OpenCV / PIL / NumPy / Triton Inference Server / Loguru / Docker
  - **技术亮点**：支持 7种处理模式（压缩/涂改/遮罩/线条涂鸦/模糊/色块/镜像），采用 cv2+ PIL双库协同保障各模式效果；严格保持输出尺寸一致性，高斯模糊与运动模糊核大小随强度动态映射；设计 bytes→处理→bytes全链路流式接口，支持 base64编解码直接入出；统一错误码体系（-1-8）规范异常处理。

### 作业帮教育科技有限公司 | 智联业务部算法实习生 | 2025.9 – 2025.11
- **项目简介**：参与智能客服与模块化 RAG系统设计与实现，涵盖敏感词过滤、同义词替换、关键词正则、FAQ相似度检索、向量索引、意图识别及检索增强生成全流程。
- **技术栈**：Python / FastAPI / LangChain / vLLM / PyTorch / Transformers / Elasticsearch / MinIO / Redis / Docker
- **技术亮点**：借助 LLM构建数据集，微调 BERT-small并集成 Qwen-3 7B/ Qwen-Turbo进行复杂意图识别；部署 simBERT-4L与 BGE-M3-Embedding，使用 Redis缓存热点回答，FAQ匹配准确率提升约 15%，响应时间降低 40%；基于 FastAPI 和 LangChain框架实现多格式文档解析、文本分块、向量化、向量检索与 LLM生成的模块化 API，阅读 LangChain源码编写自定义向量化类以兼容本地部署模型；设计 Elasticsearch向量/全文/混合检索结构，集成 Reranker优化结果相关性；异步任务队列处理大文件与批量入库，性能提升 40%+；使用 Dockerfile和 Docker Compose完成容器化部署，支持多行业快速扩展与多格式文档（PDF/Markdown/Office）处理。

### 二六三网络通信股份有限公司 | 终端开发部 AI智能研究实习生 | 2025.5 - 2025.8
- **Linux环境与工程基础**：系统掌握 Linux开发环境与 Shell脚本操作，建立规范化项目工程结构与 Git版本管理流程，熟练使用 Docker容器化部署测试环境，为后续算法研发与部署奠定工程基础。
- **论文调研与复现能力**：深入培养论文筛查、算法调研与项目复现能力，能够快速定位关键模块（Backbone/Neck/Head/Loss），理解核心创新点并完成 PyTorch架构下的完整复现，掌握从理论到可执行代码的转化方法。
- **核心算法理论与实践**：系统学习 Transformer架构原理与 PyTorch框架实现，深入理解自注意力机制、位置编码与多层堆叠设计；同步学习强化学习算法（PPO/GRPO等）核心思想与训练范式，具备将 RL思想应用于实际业务场景的建模能力。
- **端到端项目实战**：在政务方言 ASR与视频插帧项目中实践 Wav2Vec2/Conformer微调、RIFE插帧调优及 GPU/NPU多平台部署，贯通”论文调研→代码复现→模型训练→评测优化→工程落地”全流程，积累从算法研究到业务交付的完整闭环经验。

---

## 实验室项目

### 学术文献综述多智能体自动生成系统 AI应用开发 | 项目负责人 | 2026.5 – 2026.7
- **项目简介**：设计并实现学术文献综述多智能体自动生成系统，基于 Multi-Agent Pipeline架构实现”检索→解析→生成→审计”全链路自动化，支持用户输入模糊中文主题自动完成 arXiv论文检索、MinerU PDF解析、分块生成与零幻觉引用审计，输出可直接投稿的 LaTeX格式文献综述文档。
- **技术栈**：Python / Multi-Agent Pipeline / Plan-and-Execute / Query Rewriting / Hybrid Retrieval（arXiv+ FAISS） / MinerU / BGE Embedding & Reranker / ChromaDB / LaTeX / LLM-as-a-Judge
- **技术亮点**：
  - **Multi-Agent Pipeline编排与自恢复机制**：设计中心化 Coordinator+ 5个单一职责子 Agent架构，以 Coordinator维护 PipelineState全局状态并控制条件重试与容错降级，子 Agent以函数调用方式受控执行，不移交控制权；支持 Searcher自动遍历 4种备选检索式重试、Reader断点续传跳过已解析 PDF，实现管线级降级。
  - **四角度 Query Rewriting与 Hybrid Search**：将用户模糊中文主题转化为 4条 arXiv检索式（精确/同义/相关/综述），构建同义词 OR扩展规则配合 ti:标题级前缀提升检索精度；同时设计在线关键词搜索（arXiv+引用量排序）与离线语义检索（FAISS+ BGE本地 Embedding）双路混合架构，双路结果标题去重合并，形成”检索→解析→存入→复用”数据飞轮。
  - **分块生成与零幻觉引用审计**：设计”分类→逐节生成→节间摘要传递”分层上下文压缩机制，每节只传入对应论文信息，通过前节末尾摘要维持连贯性，解决上下文窗口超限问题；Critic基于正则提取\cite标签逐一核对引用白名单，审查链路完全独立于 LLM，实现引用 100%可信度。
  - **Pipeline可观测性与断点续传**：设计 Coordinator+ PipelineState+ Checkpoint三层可观测架构，实时记录每步执行状态与 Token消耗；文件级断点机制确保每篇 PDF解析完成后即时写入 checkpoint，重启自动跳过已完成文件；结合 Planner 2 次自动重试与兜底策略保障管线鲁棒性。
  - **Query改写质量评估体系**：设计人工标注基准+ LLM-as-a-Judge双维度评估，从语义质量（4条检索式与标准答案对比打分，每题 1-5分）与语法质量（正则校验前缀/括号平衡/引号匹配）综合评分，建立”发现→定位→修改→验证”迭代闭环，实现 Prompt改进的量化可追溯。

### 北京欣奕华科技有限公司 | 物流场景下 RGB-D多模态多任务视觉系统研发 | 2024.10 - 至今
- **项目简介**：针对物流场景密集遮挡与目标尺度多变问题，构建 LogiMOT-RGB仓储数据集并提出 LogiTrack方法，创新性设计自适应正交注意力（Adaptive Orthogonal Region Attention, AORA）模块与运动补偿 OC-SORT跟踪算法，实现检测端 mAP50提升 5.24%、mAP50-95提升 7.79%，跟踪端 MOTA提升 1.6、HOTA提升 3.1；同时构建 LLM Agent智能监控系统打通”感知→跟踪→语义理解”全链路闭环。
- **技术栈**：Python / PyTorch / YOLO(v8–v11) / OC-SORT / DeepSeek / Qwen-VL / ChromaDB / BGE Embedding & Reranker / Streamlit / OpenCV / Loguru
- **技术亮点**：
  - **自适应正交注意力（AORA）**：提出自适应正交区域注意力机制，通过正交空间分区建模目标多尺度形变与遮挡关系，动态建模目标结构信息，有效缓解密集场景下的漏检与误检问题。
  - **RGB-D跨模态融合检测**：设计三大深度融合模块——DG-CSPStage（静态通道级门控融合 Neck层 P3/P4/P5，推理期结构重参数化零额外开销）、DGA（深度引导空间与通道双重注意力）、DA-FPN（根据深度距离动态调整多尺度特征层权重），并采用 RGB-D四通道联合训练显著提升检测精度。
  - **运动补偿跟踪优化**：提出运动补偿卡尔曼滤波算法对 OC-SORT进行适配重构，引入 Ghost Target瞬现误检剔除机制，有效应对相机运动与视角切换导致的身份跳变与跟踪漂移问题，在动态巡检场景下鲁棒性显著提升。
  - **LLM Agent智能监控系统**：基于 DeepSeek ReAct+ Function Calling构建智能分析系统，集成轨迹统计、单目标详情、多模态视觉校验（Qwen-VL）等工具；本地部署 BGE Embedding+ BGE Reranker+ ChromaDB构建工业级 RAG管线，结合仓储机器人 SOP文档实现安全规则实时查询与违规判定；基于 Streamlit构建仓库监控 Copilot系统，支持视频时间戳联动与多轮业务问答。

### 中国科学院自动化所*中日友好医院 | 基于扩散模型的多模态医学影像生成技术研究 | 2025.2 - 2025.10
- **扩散模型理论基础**：系统学习扩散概率模型（DDPM）核心原理，深入推导前向加噪过程与反向去噪建模，理解 KL散度在变分推断中的数学意义与推导链路；对比学习 VAE、Flow-Based Model等生成式模型，掌握各生成范式的理论基础、优劣势与演化关系，构建完整的生成模型知识体系。
- **论文调研与复现能力**：深入研读扩散模型领域核心论文（DDPM、Classifier-Free Guidance、Stable Diffusion等），能够快速定位关键模块（U-Net Backbone/ Condition Embedding/ Noise Scheduler），理解创新点并完成 PyTorch架构下的完整复现，掌握从理论公式到可执行代码的转化方法。
- **核心算法理论实践**：以垂体腺瘤医学影像生成为载体，实践扩散模型的条件控制机制与可控生成策略，深入理解生成分布与真实分布间的度量方法；结合医学影像严苛的生理解剖约束，探索生成质量与结构一致性的平衡方案，积累扩散模型在下游分割任务中的数据增强经验。
- **端到端项目实战**：在医学影像生成项目中贯通”论文调研→理论推导→模型复现→生成调优→下游验证”全流程，将生成的影像数据反哺下游分割模型训练，在病灶分割对比评估中显著补充先验特征并提升分割精度，实现从学术研究到下游任务的完整闭环。

---

## 专业技能
- **编程语言与工具**：精通 Python，熟练 C++；熟悉 Linux/ Shell、Git、GitHub；熟练使用 Docker容器化部署；掌握 PyTorch、PyTorch Lightning、HuggingFace框架。
- **深度学习算法**：深入掌握目标检测（YOLO系列）、多目标跟踪（OC-SORT/ByteTrack）、扩散模型（DDPM/RIFE）；系统学习强化学习（PPO/GRPO）算法原理与训练范式。
- **大模型与 Agent**：熟练掌握 Transformer、LLM、Agent（ReAct+ Function Calling）架构；熟悉 RAG检索增强（LangChain / BGE Embedding+ ChromaDB）、Multi-Agent Pipeline、Skill分层路由机制；具备多模态大模型（Qwen-VL）开发经验。
- **语音识别**：熟练掌握 Wav2Vec2、Conformer架构及 CTC+LM联合解码，具备低资源方言 ASR热词优化与模型微调能力。
- **模型部署**：掌握 ONNX、TensorRT模型转换与加速优化；熟练使用 Triton Inference Server部署；具备 GPU/ NPU多硬件平台异构部署经验。
- **英语水平**：CET-6