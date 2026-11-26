# 本科毕业论文（设计）开题报告 —— LaTeX 版

基于 [hithesis](https://github.com/hithesis/hithesis) 模板（`examples/hitart/reports`，哈尔滨校区本科生开题报告结构）重写。

题目：**基于记忆与想像建模的视觉-语言目标跟踪方法研究**
学生：刘明卓（2023111648）　指导教师：马丁

## 文件结构

```
tex/
├── report.tex                          # 主文件（编译入口）
├── reference.bib                       # 参考文献（34 条）
├── front/coverart.tex                  # 封面信息（题目/姓名/学号/导师）
├── body/report_harbin_bachelor_opening.tex   # 正文（第 1~8 章）
├── figures/                            # 图片目录
├── hithesis.dtx / hithesis.ins         # 模板源（生成 .cls 用）
├── *.cls *.cfg *.bst                   # 已生成的格式文件（勿删）
└── report.pdf                          # 【产出】编译好的 PDF
```

## 编译方法

本机可用 MiKTeX（`E:\MiKTeX\miktex\bin\x64`）。编译顺序：

```bash
export PATH="/e/MiKTeX/miktex/bin/x64:$PATH"
xelatex report.tex
bibtex  report
xelatex report.tex
xelatex report.tex
```

或直接用 `latexmk -xelatex report.tex`（若已安装 latexmk）。

> 注意：必须是 **xelatex**，不能用 pdflatex。

若在其它机器上首次编译，需要先用 `xelatex hithesis.ins` 重新生成 `hithesisart.cls` 等格式文件。

## 插图说明

原 Word 文档中嵌入的 4 张图已自动提取并插入到位，无需手工处理。图片已存放在 `figures/` 目录：

| 图号 | 内容 | 文件名 | 尺寸 |
|------|------|--------|------|
| 图 1-1 | 视觉-语言目标跟踪任务示意 | `fig1-1-vlt-task.png` | 638×353 |
| 图 3-1 | 基线 MVLM 跟踪框架 | `fig3-1-mvlm.png` | 1421×622 |
| 图 3-2 | 加入记忆库与世界模型的总体跟踪框架 | `fig3-2-framework.png` | 1421×540 |
| 图 4-1 | 总体技术路线与阶段推进 | `fig4-1-roadmap.png` | 2509×811 |

若要替换某张图，直接用同名文件覆盖即可；若要调整显示宽度，改正文中对应 `\includegraphics[width=...]` 的比例值。

## 表格

4 个表格已用 `booktabs` 三线表写好（表 3-1、表 4-1、表 5-1、表 6-1），内容与原始 Word 文档一致，无需手工处理。
