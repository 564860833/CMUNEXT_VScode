# Agent Workspace Rules

- 临时代码、实验脚本、截图、生成图片、日志、导出文件等临时产物必须放在仓库根目录的 `.codex-scratch/`。
- 除非用户明确要求修改正式项目文件，否则不要在源码、配置、数据或训练输出目录中创建无关文件。
- 如果需要临时验证实现，先使用 `.codex-scratch/`；完成后不要把其中内容加入 Git。
- 正式代码改动仍按用户请求在项目对应位置进行，不能把需要提交的功能代码放进 `.codex-scratch/`。
- 

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
