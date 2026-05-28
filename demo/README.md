# 自动化评测 · 本地网页（真实数据 + 真实 LLM）

## 启动

```powershell
cd demo
pip install -r requirements-demo.txt
python app.py
```

浏览器打开：**http://127.0.0.1:5050**

## 文档

| 文档 | 说明 |
|------|------|
| [`../docs/README.md`](../docs/README.md) | 文档中心 |
| [`../docs/平台建设方案.md`](../docs/平台建设方案.md) | 建设路线与优先级 |
| [`../docs/流程与限制说明.md`](../docs/流程与限制说明.md) | 当前流程、API、限制 |
| [`../产品基石.txt`](../产品基石.txt) | 快速总览 |

## 流程（三步）

1. **上传** Excel/CSV，填写数据集说明与期望输出  
2. **澄清** — 列映射、Prompt、确认并准备数据  
3. **运行** — 小样本 / 全量跑批，查看结果与导出  

## 原任务样本

- 任务元数据：`demo/task_registry.json`  
- 可从任务列表 **「加载样本」** 拉取工作区内已有 CSV  
- 典型业务样例见 [`../示例_V0/`](../示例_V0/)

## 目录

| 目录 | 用途 |
|------|------|
| `uploads/` | 用户上传 |
| `backups/` | 准备前自动备份 |
| `prepared/` | 规范化后的跑批输入 |
| `templates/` | 澄清用 Jinja 模板 |
