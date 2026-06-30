# Claude Code 项目规则

通用规则见下方导入的 AGENTS.md；本文件追加 Claude Code 专属覆盖，优先级高于导入内容。

@AGENTS.md

---

## 覆盖：scratch 根目录用 `.claude-scratch/`

- 导入内容中出现的 `.codex-scratch/`，对 Claude Code 一律按 `.claude-scratch/` 理解并执行；其余规则（venv、命名、分类、清理、不入 Git 等）完全不变。
- `.codex-scratch/` 是 Codex 专用，Claude Code 禁止向其写入、创建或删除任何文件（避免两个工具互相污染）。
