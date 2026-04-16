# JSONL 标注工具

用终端浏览 `data/` 里的 JSONL 数据，并把标注结果写到 sidecar 文件。详情面板默认展示自然化内容，按 `r` 可以切回原始 JSON。

## 安装

```bash
uv sync
```

## 启动

```bash
python -m annotations
```

直接打开单个文件：

```bash
python -m annotations --file data/parallel_200.jsonl
```

导出合并结果：

```bash
python -m annotations export --file data/parallel_200.jsonl
```

## 快捷键

- `g`: 标记为好
- `b`: 标记为不好，并填写原因
- `e`: 编辑原因，保存为 `不好：原因`
- `n` / `p`: 下一条 / 上一条
- `j`: 跳转到 ID 或行号
- `f`: 切换过滤视图
- `r`: 在自然视图和原始 JSON 之间切换
- `o`: 查看原始来源
- `q`: 退出

## 输出

- sidecar: `annotations/<dataset>.feedback.jsonl`
- 导出: `data/<dataset>.annotated.jsonl`
