# Write Agent Liujiangli

## 项目说明

本项目是一个AI写作助手项目，包含文档处理、模型推理等功能。

## ⚠️ 重要提示：未上传到GitHub的大文件

由于GitHub对单个文件和仓库大小有限制（单个文件不超过100MB），以下文件和文件夹**没有被上传到GitHub**：

### 1. MinerU 文件夹
- **路径**: `MinerU/`
- **原因**: 整个MinerU项目文件夹包含大量依赖文件和模型，总体积超过100MB
- **说明**: MinerU是一个文档解析和处理工具，需要从官方仓库单独下载

### 2. 模型文件
以下类型的模型文件由于体积过大（通常几百MB到几GB），未上传到GitHub：

- **`*.bin`** - 二进制模型文件（如BERT、GPT等预训练模型）
- **`*.safetensors`** - 安全的张量文件格式（HuggingFace模型常用）
- **`*.pth`** - PyTorch模型权重文件
- **`models/`** - 整个models文件夹，包含所有下载的预训练模型

### 3. 输出文件夹
- **`output_*/`** - 程序运行产生的输出文件
- **`output/`** - 输出目录
- **`papers_*/`** - 论文处理产生的临时文件

### 4. Python缓存文件
- **`__pycache__/`** - Python字节码缓存
- **`*.pyc`** - 编译的Python文件

## 如何获取未上传的文件

### 获取MinerU
```bash
# 从官方仓库克隆MinerU
git clone https://github.com/opendatalab/MinerU.git
```

### 获取模型文件
模型文件需要根据项目需求从以下来源下载：
- **HuggingFace Hub**: https://huggingface.co/models
- **ModelScope**: https://modelscope.cn/models
- 或参考项目文档中的模型下载说明

### 运行模型下载脚本（如果项目提供）
```bash
# 检查项目是否有模型下载脚本
python -m mineru.cli.models_download
```

## .gitignore 配置

本项目已配置`.gitignore`文件，自动排除以下大文件类型：

```gitignore
# 输出文件夹
output_*/
output/
papers_*/

# MinerU文件夹（超过100MB）
MinerU/

# Python缓存
__pycache__/
*.pyc

# 模型文件（超过100MB）
models/
*.bin
*.safetensors
*.pth
```

## 项目结构

```
write_agent_liujiangli/
├── MinerU/                    # [未上传] 文档解析工具（>100MB）
├── models/                    # [未上传] 模型权重文件（>100MB）
├── output/                    # [未上传] 输出文件夹
├── .env                       # 环境配置文件
├── .gitignore                 # Git忽略配置
└── README.md                  # 本文件
```

## 注意事项

1. **克隆本项目后**，需要单独下载MinerU和模型文件才能正常运行
2. **模型文件较大**，建议使用高速网络或镜像源下载
3. **MinerU**可以参考其官方文档进行安装和配置
4. 如有任何问题，请参考项目文档或提交Issue

## 许可证

请参考项目中的LICENSE文件（如有）