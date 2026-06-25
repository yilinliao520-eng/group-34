# Overleaf 编译说明

推荐做法：

1. 在 Overleaf 新建一个 blank project，或使用官方 ACL template。
2. 上传本目录中的 `final_report_group34_zh.tex`、`acl.sty`、`references.bib` 和 `figures/`。
3. 将 Overleaf compiler 设为 `XeLaTeX`。
4. 如果 template 默认主文件不是 `final_report_group34_zh.tex`，请在 Menu 中把 Main document 改成该文件，或把内容复制到 `main.tex`。

注意：

- 正文是中文，因此不要用 pdfLaTeX。
- `figures/workflow.pdf` 和 `figures/main_results.pdf` 是给 LaTeX 使用的版本。
- 本目录的 `acl.sty` 参考了课程同学仓库中的 lightweight ACL-like 格式，并把页眉改为本项目主题；若老师严格要求官方 ACL template，也可以删除本地 `acl.sty`，改用官方 ACL template 自带的 style。
