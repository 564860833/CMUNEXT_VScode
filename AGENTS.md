# Agent Workspace Rules

- 临时代码、实验脚本、截图、生成图片、日志、导出文件等临时产物必须放在仓库根目录的 `.codex-scratch/`。
- 除非用户明确要求修改正式项目文件，否则不要在源码、配置、数据或训练输出目录中创建无关文件。
- 如果需要临时验证实现，先使用 `.codex-scratch/`；完成后不要把其中内容加入 Git。
- 正式代码改动仍按用户请求在项目对应位置进行，不能把需要提交的功能代码放进 `.codex-scratch/`。

## Scratch Directory Hygiene

- `.codex-scratch/` 只用于临时产物，不得直接在 `.codex-scratch/` 根目录散放文件。
- 每次任务必须在 `.codex-scratch/` 下创建独立子目录，格式为：
  `.codex-scratch/YYYYMMDD-HHMM-简短任务名/`
  例如：
  `.codex-scratch/20260628-1430-model-shape-check/`
- 子目录内文件名必须能说明用途，避免使用 `test.py`、`tmp.txt`、`1.png` 这类含义不清的名字。
- 推荐按用途进一步分类：
  - `scripts/`：临时验证脚本
  - `logs/`：运行日志
  - `outputs/`：导出结果
  - `screenshots/`：截图
  - `cache/`：测试缓存或中间缓存
- pytest 缓存、临时日志、实验输出必须写入当前任务目录或 `.codex-scratch/.pytest_cache/`，不得污染仓库根目录或源码目录。
- 临时验证完成后，应删除无用的大文件、重复文件和失败实验残留；只保留对复现或说明问题有价值的产物。
- 在写入 `.codex-scratch/` 前，必须先确认本次任务应使用哪个子目录；如果不存在，则先创建规范命名的任务目录。
- `.codex-scratch/` 中内容不得加入 Git，除非用户明确要求。

## Python/Test Environment

- 在本仓库运行 Python 脚本、单元测试、训练/推理验证前，必须优先使用仓库根目录的项目虚拟环境 `.venv`。
- Windows 下优先使用：
  `D:\DeepLearning\CMUNeXt\.venv\Scripts\python.exe`
  例如：
  `D:\DeepLearning\CMUNeXt\.venv\Scripts\python.exe -m pytest tests/test_best0616_models.py -o cache_dir=.codex-scratch/.pytest_cache`
- 不要优先使用系统 `python`、全局 conda/base 环境或其他解释器；只有当 `.venv` 不存在、损坏或缺少必要依赖时，才允许回退到其他环境，并必须在回复中说明原因。
- pytest 缓存、临时测试输出、日志等仍必须放到 `.codex-scratch/`，不要在仓库根目录或源码目录创建无关临时产物。
- 判断“环境缺少依赖”前，必须先检查 `.venv`，例如：
  `D:\DeepLearning\CMUNeXt\.venv\Scripts\python.exe -c "import torch, pytest"`
