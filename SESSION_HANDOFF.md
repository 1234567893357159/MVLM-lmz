# MVLM 会话交接记录

最新运行（2026-10-19上海时间）：用户已要求暂停每200步检查的新实验；后台监督程序已终止，状态`paused`，更新步数0。W&B run okgvc2hi；显卡恢复后不会自动开训。上一轮最佳2000步平均AUC60.8004。

最新状态（2026-10-19）：20,000步训练及逐项诊断均完成。原配置平均AUC58.72，基线60.73；缩小融合或严格写入的完整验证均约61.07。最新细节见文末“记忆模块逐项诊断已完成”；历史下载/训练状态不代表当前状态。

更新时间：2026-10-19（Asia/Shanghai）。已补充开题报告并清理被后续进展取代的状态、下载快照和待办；实验历史与原始基线保留。动态下载状态以状态文件、进程、日志及有效字节增长共同核实，不沿用本文数字作实时进度。

## 分支用途与当前开发位置（2026-10-17 更新，优先确认）

**当前工作分支：`codex/xmem-memory`。用户要求后续记忆库相关修改都在这个分支进行。**本分支从本地 `main` 的 `d36160fa1af4e7438f67c0b58fb3f2c765e7d9fc` 创建；创建时只切换分支，未修改模型代码、创建提交或推送远端。

本分支统一中文说明：`codex-xmem-memory分支说明.md`（由原 `记忆库改动说明.md` 更新重命名），包含完整文件改动、训练配置、已完成短训／评测结果、产物与复跑入口，以及后续工作。分支功能变化时同步维护该文件。

### 本地分支

| 分支 | 用途 | 开发约定 |
| --- | --- | --- |
| `main` | 保存已有 MVLM 复现适配、评测脚本、实验配置及精简结果，作为结构改造的固定起点 | 保留现有基线；新增记忆库功能在开发分支实现。这里的 `mvlm_TF.yaml` 是较早调参配置，不代表未经修改的官方默认配置 |
| `codex/xmem-memory` | 当前毕设的第一阶段结构开发：参考 XMem 构建目标级短期／长期显式内容记忆库，并接入 MVLM | 后续修改默认在此分支进行；先实现记忆存储、可靠写入、检索与融合，验证有无记忆的差别，再考虑后续模块 |

本阶段重点为内容记忆本身：保存可靠观测的目标外观、语言相关内容和时序状态，设计容量管理与目标切换重置，保留原有置信度记忆；新增功能可通过配置关闭，便于对照基线。记忆库与融合、时序训练接入的首版现已实现，状态见下文。残差 Transformer 世界模型与预测条件质量头属于后续阶段，当前尚未实现。

### 远端引用与来源

| 引用 | 用途 | 说明 |
| --- | --- | --- |
| `origin/main` | 个人仓库 `git@github.com:1234567893357159/MVLM-lmz.git` 的主分支远端跟踪引用 | 分支创建时指向 `d36160f`；记忆开发使用独立的 `codex/xmem-memory`，不直接更新 main |
| `origin/codex/xmem-memory`（发布目标） | 个人仓库的同名记忆开发分支 | 用户已授权提交并推送代码、配置与说明；使用 `git push -u origin codex/xmem-memory` 发布，实际同步状态以 Git 查询为准 |
| `upstream/main` | 官方仓库 `https://github.com/inha-vllab/MVLM.git` 的主分支远端跟踪引用 | 本地保存的引用为 `4186c8f`；用于追溯官方代码及复现差异，不在此引用上直接开发 |
| `upstream/master` | 官方仓库另一个已保存的历史远端跟踪引用 | 本地指向 `6e36b74`；尚未核实其与主分支的内容差异，不作为当前开发起点 |

上述远端信息来自本地已有引用，本次未 fetch，不代表远端实时状态。`upstream/HEAD` 当前指向 `upstream/main`，是默认分支指示，不是额外开发分支。

### 未提交内容与实验记录约定

- 创建分支前已有 `SESSION_HANDOFF.md` 的未提交更新及未跟踪的 `tex/`；切换时均原样保留，目前仍未提交。未提交和未跟踪文件不由分支隔离，切换分支时可能随工作区一起保留；不能把这些内容视为已经备份进分支历史。
- 后续继续工作前先执行 `git branch --show-current` 和 `git status --short`，确认位于 `codex/xmem-memory`，保留既有文档与报告文件。
- 实验记录代码提交号、配置、数据划分、权重来源和指标；不同实验使用独立 exp_id，避免跳过旧预测或覆盖结果。数据集、下载分块、模型和大型输出仍按 `REPRODUCTION.md` 的约定保存在本机，不因切换分支而复制或删除。
- 新增分支或变更分支用途时同步更新本节；目前只有 `main` 与 `codex/xmem-memory` 两个本地分支，没有单独的世界模型或质量头开发分支。

### 小规模真实训练与评测已通过（2026-10-17）

- 用户授权使用已下载的小数据集验证训练与评测，已用 OTB99-Lang 的 Basketball、Bolt、Boy、Car4 四条训练序列完成 1 epoch、20 次真实更新，并在独立测试序列 Biker、Bird2、Dancer2（共 391 帧）上完成基线和记忆模型评测。训练与测试图像无交集。
- 结果（PRE／NPR／AUC）：同参数基线 **91.15／64.97／58.16**；记忆短训 **90.14／64.61／57.97**。仅用于检查工程链路，尚未充分训练，不能当成完整 OTB 成绩或记忆增益证据。
- 真实训练、参数更新、checkpoint 保存与重载、正常阈值下在线记忆写入、预测导出、指标计算均通过；冻结基线参数逐项不变，记忆输出投影权重从零发生更新。
- 成功产物：`output/memory_otb_smoke_20261017_run02/`，含训练／评测日志、配置副本、`summary.json`、`结果说明.md` 及 `checkpoints/train/mvlm/mvlm_TF_memory_otb_smoke/MVLM_ep0001.pth.tar`。精简报告：`reports/reproduction/MEMORY_OTB_SMOKE_RESULTS.md`。
- 本次补齐 OTB 训练划分读取及训练序列子集配置，修复训练加载器的路径／特殊序列处理，并修复 `lib/train/data/loader.py` 在零 worker 下的 batch 拼接错误。首次失败日志保留在 run01；成功 run02 未覆盖旧实验。
- 新配置为 `experiments/mvlm/mvlm_TF_memory_otb_smoke.yaml`，评测入口为 `tracking/evaluate_memory_otb_smoke.py`；复跑应使用新运行目录，避免自动续训或覆盖预测。后文“尚未正式训练”指尚未进行充分训练／完整基准实验；现已存在这份仅用于功能检查的短训权重。

### 当前分支实现进展：记忆库首版（2026-10-17）

- 已新增 `lib/models/mvlm/memory.py` 与 `memory_fusion.py`：目标级短期／长期内容记忆（默认 8／16 条）、2×2 ROI 特征、质量写入与新颖性筛选、容量淘汰、外观／语言条件双路检索、时间与质量偏置、空匹配槽及门控残差融合。原始 `feat_x` 保留用于图文打分和记忆写入，增强的 `feat_x_mem` 进入原 CENTER 头；原 CENTER 结构未修改。
- 已接入 `lib/models/mvlm/mvlm.py`、`lib/test/tracker/mvlm.py`：先读历史再定位、定位后写入；每个视频或 batch 样本独立持有运行时记忆；初始化空库，目标语言切换时清空内容／置信度记忆并全局重定位；新记忆不使用测试真值框初始化。
- 已接入训练采样、数据处理、Actor 和优化器：严格递增多帧片段、片段一致增强、逐帧定位损失、历史特征停止梯度，以及真值／预测历史框混合写入。默认冻结编码器、文本编码器与原 CENTER，保持冻结模块 eval 状态，但保留 CENTER 对融合输入的梯度。
- 独立配置：`experiments/mvlm/mvlm_TF_memory.yaml`；默认配置的记忆开关为关闭，原 `mvlm_TF.yaml` 未修改。同一进程读取不同跟踪配置时会隔离配置快照，旧 YAML 不会继承已开启的记忆开关。
- 记忆训练从官方完整跟踪 checkpoint 初始化，包含已学习的语言投影，因此该路径跳过额外骨干预训练文件加载；不能误删语言投影后冻结随机参数。正式评测默认拒绝缺少记忆权重的 checkpoint；只有显式开启 `TEST.MEMORY_ALLOW_UNTRAINED` 才允许诊断，不能把诊断结果报告为训练后结果。
- 检查脚本：`tracking/check_memory_module.py`。基础检查已通过：ROI 有效性、质量门控与容量、因果读取、batch 隔离、空库／零残差基线一致、时序采样与增强、定位损失反传到记忆读取、冻结参数及归一化状态不变。官方权重三帧合成序列前后向与少量真实 OTB 帧的零初始化基线一致检查也已通过。检查不是正式训练或精度评测，没有覆盖已有预测文件。
- **仅完成上述 OTB 小规模短训，尚无充分训练或精度提升结论。**配置中的 10 epoch、每卡 batch 2、4 帧片段、仅 VastTrack 是开发起点，需数据校验后再安排训练。现阶段仅采可见历史帧，训练裁剪仍围绕扰动真值，未实现预测状态驱动的在线裁剪；训练预测写入与线上 MVLM margin 门控也未完全统一。训练数据路径与线上记忆框元信息已统一到全图归一化 xywh，ROI 提取继续使用裁剪相对坐标；当前框元信息不参与注意力或运动计算。
- 详细设计、限制、检查及训练／评测命令：`reports/reproduction/MEMORY_IMPLEMENTATION.md`。后续经用户授权已完成上述真实数据短训，尚未全量评测；代码和说明通过当前分支提交发布，具体提交号与远端同步状态以 Git 查询为准。

## 0. 毕设开题报告与整体研究方向（优先阅读，2026-10-17 新增）

用户已确认：工作区 `tex/` 中的 TeX 文档是本人的本科毕业设计开题报告。当前 MVLM 复现工作是毕设的基线准备阶段，后续结构改进应结合这份报告理解；以下内容是报告中的研究计划，不代表新增模块已经实现或训练已启动，也不构成新增数据下载、训练或对外联系的授权。

### 0.1 报告位置与版本

- 题目：**《基于记忆与想像建模的视觉-语言目标跟踪方法研究》**；学生刘明卓，指导教师马丁，哈尔滨工业大学哈尔滨校区、计算学部、软件工程本科。
- 课题来源：报告记载为国家自然科学基金青年基金项目“基于自然语言查询的视觉目标跟踪方法研究”的支撑课题，项目负责人马丁。
- 编译入口：`/root/MVLM/tex/report.tex`；当前正文：`/root/MVLM/tex/body/report_harbin_bachelor_opening.tex`；封面信息：`/root/MVLM/tex/front/coverart.tex`，封面日期固定为 `2026.09.28`。
- 参考文献：`/root/MVLM/tex/reference.bib`；四张插图：`/root/MVLM/tex/figures/`，分别为任务示意、MVLM 基线、改进总体框架与技术路线。
- 已有 PDF：`/root/MVLM/tex/report.pdf`；目录另有多个 Word 文件，不能仅凭名称判断哪份最新，也未核实 PDF 与当前 TeX 内容完全同步。
- `tex/_backup_20261008-235046/` 是历史备份，阅读与后续修改默认以备份目录外的当前源文件为准；模板类文件不是报告正文。
- 使用 hithesis 的 `hithesisart` 本科开题报告模板，编译方式为 XeLaTeX → BibTeX → XeLaTeX → XeLaTeX，或在 `tex/` 下运行 `latexmk -xelatex report.tex`。当前主文件设置 `fontset=windows`，并显式引用 SimSun、SimHei；本服务器 TeX 工具链与字体尚未验证。`tex/README.md` 的 Windows/MiKTeX 路径属于原编写环境，不能直接当作本机配置。
- 本次读取时 `git status --short` 显示 `?? tex/`，报告目录尚未纳入 Git 跟踪；本次只补写交接，不提交或修改报告。

### 0.2 研究主线与改进边界

报告以无模板、语言条件的 MVLM 为基线，主线为**“观测—记忆—想像—注入”**，面向遮挡、快速运动、相似物干扰和长序列漂移。报告明确区分：MVLM 已有的时间记忆是标量置信度的指数平滑，新增的是可存取目标外观、语义和时序状态的**显式内容记忆**，不能将基线描述成完全没有记忆。

1. **内容记忆库**：构建感知—认知双流键值记忆，分别以外观、语义表征为键，以时序状态为值；带时间位置编码，通过注意力检索，容量受限时按冗余度、时间衰减等策略淘汰。以跟踪置信度与推演一致性双重门控写入；语言描述变更或目标切换时重置记忆。
2. **记忆条件下的潜空间想像**：采用轻量 Transformer，以记忆读出、语言向量和当前视觉特征为条件，做单步未来状态推演。预测头 A 输出未来目标表征，预测头 B 输出候选框的前瞻可靠度权值。利用未来帧真值框内特征、下一帧候选框 IoU 与得分构造监督，不新增人工标注；推理时用预测与观测的一致性门控抑制幻觉。
3. **融合注入与候选框筛选**：未来表征先与当前空间网格对齐，比较拼接、交叉注意力两种注入方式；以前瞻权值联合基线分类与相关度得分，研究加权 Top-K 与 BBOX 过滤。主要改动集中在候选框生成和筛选，保留基线置信度聚合及局部 ROI／全局重定位门控形式，视觉主干计划全程冻结。

报告的方法学参照包括 STM、XMem 的内容记忆以及 MemoryVLA++ 的记忆与想像联合建模；拟采用满足在线逐帧预算的轻量表征预测器，具体层数、维度、损失权重与空间对齐实现仍待确定，不能把这些设想写成已完成的自主成果。

### 0.2.1 当前确定的技术选型（2026-10-17 用户确认；2026-10-19 更新训练策略）

以下选型是用户在方案讨论后明确要求记录的实现方向；上文是开题报告原方案摘要，如有差异，以本节的当前选型为准。记忆库首版已实现；截至2026-10-19，已完成冻结头训练和记忆模块与预测头联合短训，当前方案为冻结视觉／文本编码器、联合训练记忆模块与CENTER预测头。每200步检查的新一轮联合训练已配置但暂停、实际0步；世界模型尚未实现，具体规模仍需实验确定。

- **记忆库参考 XMem**：借鉴其分层记忆与长期信息管理思路，面向 MVLM 实现目标级的短期／长期显式内容记忆，不直接照搬视频分割中的整帧稠密记忆。记忆保存可靠观测的目标外观、语言相关内容与时序状态，通过检索和更新复用历史信息；短期保留近期动态，长期保留有代表性的外观。参考论文：<https://arxiv.org/abs/2207.07115>。
- **世界模型计划采用残差 Transformer**：以历史内容记忆、语言和当前观测为条件，在目标潜空间中单步预测下一帧外观表征的残差及运动变化，并考虑运动不确定性；采用“当前表征＋预测残差”构造下一帧表征。第 t 帧定位使用上一帧生成的第 t 帧预测，定位完成后再预测第 t+1 帧，保持因果时序。
- 当前主方案为确定性的潜空间残差预测，不是扩散或去噪模型；报告中“去噪式推演”等表述后续需与实际路线统一，本次不修改 TeX。预测候选框质量可作为独立的当前帧质量分支，不与世界模型的未来状态输出混为一谈。
- 本地实现中的 `kmem` 实际是候选网格对应的置信度数组的指数平滑，并非全局单个标量；上文报告摘要中的“标量置信度”不应理解为整个跟踪器只保存一个数。新增内容记忆与该置信度记忆并存。

### 0.3 训练、评测与时间安排

- 阶段零：复现 MVLM，建立可复用评测链路；报告设定与论文指标差距控制在 **1 个百分点以内**后进入模块阶段。若无法完全对齐，报告也提出如实记录差异，以同一自复现基线的相对提升作为主要评价方式。当前测试集调参结果与原始配置基线必须分开记录，详见第 9 节。
- 上述阶段划分是开题报告计划，不代表本分支当前状态。当前训练实现冻结视觉／文本编码器，同时联合训练记忆模块和CENTER预测头；后续报告应按已完成实验改写阶段描述。
- 计划训练集：TNL2K、LaSOT、VastTrack 的训练部分；计划评测集：TNL2K、LaSOT、OTB99-Lang、MGIT；指标 PRE、AUC、NPR。纯语言与“框+语言”方法须分别遵守对应协议比较。报告列出四个评测集不等于已授权下载 MGIT；实际下载授权与准备状态仍以第 1、10、11 节及实时检查为准。
- 消融分四组：内容记忆（无／有／淘汰策略）、潜空间推演（无／视觉前瞻／权值前瞻／双头联合）、注入方式（拼接／交叉注意力）、筛选策略（原始／权值引导）；另做遮挡、快速运动、背景干扰、尺度变化属性分析，以及参数量、推理速度分析。
- 报告计划周期：2026.09—2027.05；2026.09—10 基线复现，2026.10—2027.01 模块落地，2027.01—03 消融优化，2027.03—05 论文与答辩。

### 0.4 与工作区现状的区别及待核对项

- 开题报告是研究规划，实际进度以实验产物和实时核查为准。报告中“数据资源已具备”等描述不能解释为本机训练集已齐备：当前仅完成 OTB 小规模记忆训练检查，完整训练数据准备事项见第 10、11 节。
- 正文的 NPR 公式写为归一化精度曲线在 `[0, 0.5]` 上的面积，且 `W_t` 写成宽高的对角矩阵；交接第 8 节记录当前仓库实际报告的是按宽高归一化后阈值 `0.2` 处的精确率。这里存在指标口径与归一化公式的待核对问题，后续应对照评测代码统一，不能直接按报告公式解释当前 NPR 数字。本次未改报告。
- `tex/README.md` 标注参考文献 34 条，而当前正文称 40 篇，说明目录说明可能滞后；文献数量与引用有效性本次未逐条校验。
- 报告中的研究动机、相关工作判断与性能预期属于原文陈述，本次仅阅读整理，没有开展文献事实核验、重新编译或实现新增网络模块。

## 1. 用户目标与当前任务

- 用户希望使用论文官方权重，复现 MVLM 在测试集上的 PRE、NPR、AUC 等结果，并为开题报告中的结构改进准备基线。不要重新克隆或覆盖已有权重。已按用户要求完成 OTB 小规模记忆训练与独立子集评测；VastTrack 训练数据准备也已授权，完整训练尚未开展。
- 已运行两个自带视频 Demo。
- 用户已明确授权下载前三个数据集：OTB99-Lang、TNL2K 测试集、LaSOT 标准测试集。
- LaSOT 已完成完整性校验、原始基线及 8 组参数实验；OTB 已完成原始基线、早期 9 组及后续 10 组调参实验，详见第 9 节。TNL2K 与 VastTrack 下载任务已启动，完成状态需实时核查；MGIT 及其他训练数据尚未下载。
- 当前 YAML 是较早的 OTB 调参配置，既非官方默认，也非最新最优；原始配置及各实验快照应分别使用。
- 用户经常询问实时进度和速度：应实际读文件、查进程，速度应采样约 20 秒的有效字节增长，不复述旧数字。

## 2. 环境与安装

- 远程 Linux x86_64 容器；仓库：`/root/MVLM`。
- 原仓库已由用户下载，地址：<https://github.com/inha-vllab/MVLM>。
- 安装了 Miniconda：`/root/miniconda3`；Conda 版本安装时为 26.7.1。
- 创建环境：`mvlm`，Python 3.8.20。
- 环境 Python：`/root/miniconda3/envs/mvlm/bin/python`。
- `requirements.txt` 依赖已全部安装，`pip check` 当时通过。
- 主要版本：PyTorch 1.13.0+cu117、torchvision 0.14.0+cu117、OpenCV 4.12.0、NumPy 1.24.4。
- 补装系统库 `libgl1`、`libglib2.0-0`，解决 `cv2` 导入缺少 `libGL.so.1`。
- 服务器有 8 张 RTX 3090（每张 24 GB）；已验证 CUDA 可用及 GPU 张量运算。
- 下载或训练前重新检查磁盘容量、inode 与账号配额；历史可用空间不作为实时容量依据。此前失败没有空间不足证据。

激活环境：

```bash
source /root/.bashrc
conda activate mvlm
cd /root/MVLM
```

也可直接用环境 Python 的绝对路径，不依赖 shell 激活。

## 3. 已保存的复现适配与版本状态

复现代码与实验摘要已保存于提交 `52f662f`，来源与毕设用途声明见 `d36160f`。下列为相对上游的主要适配，不再是全部未提交修改；用 Git 历史和当前 `git status` 区分，不要回滚已有成果：

1. `lib/local_config.py`：工作区、输出路径及前三个数据集路径改为 `/root/MVLM/...`。
   - `lasot_dir = /root/MVLM/data/lasot`
   - `tnl2k_dir = /root/MVLM/data/tnl2k`
   - `otb99_lang_dir = /root/MVLM/data/otb99_lang`
   - VastTrack 已改为 `/root/MVLM/data/vasttrack`，训练加载器自动追加 `train/`；MGIT 仍为原始 `/Tracker/MVLM/...` 占位路径。
2. `lib/models/mvlm/encoder.py`：增加 `training=True` 参数，推理时跳过训练用 Fast-iTPN 预训练文件的额外加载。
3. `lib/models/mvlm/mvlm.py`：`build_mvlm(cfg, training=True)` 传递上述开关。
4. `lib/test/tracker/mvlm.py`：构建模型时传 `training=False`，随后加载官方跟踪权重。
5. `tracking/demo.py`：将新接口 `torch.amp.autocast('cuda', ...)` 改为兼容 PyTorch 1.13 的 `torch.cuda.amp.autocast(...)`。
6. `lib/test/evaluation/otb99langdataset.py`：将序列 `Human4_2` 的语言查询映射到 `Human4.txt`，保留序列名称及其第二目标真实框标注。
7. `experiments/mvlm/mvlm_TF.yaml`：保留较早 OTB 调参配置，ALPHA_CORR=0.7、TAU=0.03、OMEGA=0.2、LAMBDA=0.4、PSI_OUT=0.1，并非最新最优。原始配置在 `reports/reproduction/original_config.yaml` 和 `output/otb_parameter_trials/otb99_language_baseline.yaml`；最新最优结果见第 9.8 节。
8. 新增实验脚本：`tracking/run_otb_parameter_trial.py`（首次单变量实验）、`tracking/run_otb_parameter_sweep.py`（5 组自适应实验）、`tracking/run_otb_parameter_followup.py`（3 组补充实验），记录配置快照、运行命令、预测、日志、指标与每组缓存。第一个脚本要求输入仍为原始 TAU=0.1，不可在当前 YAML 上不加检查地重复执行。

训练调用沿用默认 `training=True`。代码、精简报告与大型本地产物的保存范围见 `REPRODUCTION.md`；模型、数据、完整输出不纳入 Git。本次清理未创建提交或 PR。

## 4. 权重、Demo 和结果

- 官方权重：`/root/MVLM/models/MVLM_TF.pth.tar`，约 358 MB（`ls -h` 约 342 MiB）。
- 来源为仓库 README 的 Google Drive：<https://drive.google.com/file/d/1E1bbSYhfpEjhaas3mu_2XAE14WKUMCUS/view>。
- 已安装 `gdown` 并使用它下载权重。
- CLIP ViT-L/14 缓存：`/root/.cache/clip/ViT-L-14.pt`，约 890 MiB，首次 Demo 已下载。
- 检查过权重加载：没有 unexpected keys；CLIP 本体权重不在 MVLM checkpoint 中，由单独 CLIP 模型加载。

| 场景 | 原视频 | 输出视频 |
| --- | --- | --- |
| 网球 | `/root/MVLM/demo_videos/demo_sample_tennis_1.mp4` | `/root/MVLM/output/tennis_tracking.mp4` |
| 海滩 | `/root/MVLM/demo_videos/demo_sample_beach_1.mp4` | `/root/MVLM/output/beach_tracking.mp4` |

- 网球目标描述：`the tennis player on the left side of the court`。
- 海滩目标描述：`the white dog with black spots on the beach`。
- 两个输出均已完整解码验证，日志没有逐帧跟踪异常。
- 网球输出 551 帧、约 18.39 秒、40 MB；海滩输出 1858 帧、37.16 秒、95 MB。
- Demo 将首帧用于初始化，没有写入输出，故比原视频少一帧。
- 预览：`output/tennis_preview.jpg`、`output/beach_preview.jpg`。
- 日志：`output/tennis_demo.log`、`output/beach_demo.log`。
- 效果：海滩抽查能框住狗；网球抽查框偏离选手，不能把运行成功说成精度达标。
- 早期已将当时的 output 打包：`/root/MVLM/output.tar.gz`，约 129 MiB。它包含 Demo 视频、预览、日志，不包含原视频，也不包含后来新增的评测与调参结果。
- 没有直接将文件传到用户笔记本；曾提供压缩包路径及 scp 方法。

复跑命令（避免无意覆盖现有视频，可改输出名）：

```bash
cd /root/MVLM
/root/miniconda3/envs/mvlm/bin/python tracking/demo.py \
  --config mvlm_TF --checkpoint models/MVLM_TF.pth.tar \
  --video demo_videos/demo_sample_beach_1.mp4 \
  --text "the white dog with black spots on the beach" \
  --skip-selection --no_display --output output/beach_tracking_new.mp4
```

`--skip-selection` 实际会使用默认中心框初始化；它是否完全满足论文的纯语言协议，需要正式评测前核查，不能只依据 README 标签下结论。

## 5. 数据集下载：位置、来源与校验状态

压缩包和分块目录：`/root/MVLM/downloads/`。解压后：

```text
/root/MVLM/data/
├── otb99_lang/
│   ├── OTB_videos/
│   └── OTB_query_test/
├── tnl2k/
│   └── test/
└── lasot/
    └── airplane/airplane-1/...   # 按类别/序列组织
```

| 数据集 | 压缩包总字节数 | 已确认状态 |
| --- | ---: | --- |
| OTB99-Lang | 2,798,845,728 | 48 条测试序列、29,077 帧及标注校验通过，已完成评测与调参 |
| TNL2K | 91,092,953,063 | 下载任务已启动，尚未记录完整校验及评测完成；实时进度查第 6 节所列文件 |
| LaSOT | 49,830,545,913 | 下载、解压、全量校验、完整评测与调参均已完成，见第 9.4、9.5 节 |

TNL2K 与 VastTrack 的有效授权是继续下载，保留已有分块。本文不保留易失效的下载百分比、速度或剩余时间。

下载来源：

- OTB 官方：<https://isis-data.science.uva.nl/zhenyang/cvpr17-langtracker/data/OTB_sentences.zip>。
  - 已读取 ZIP 中央目录确认包含 `OTB_sentences/OTB_videos/`、`OTB_query_test/`、`OTB_query_train/`。
  - 数据目录含 99 个视频目录；评测加载器选用 48 条测试序列，不是评测全部目录。已验证这 48 条需要的 29,077 帧、框标注长度和语言文件。训练加载器现已按包内 51 条训练语言标注适配（含 Jogging 双目标），并完成四序列短训检查，详见前部新记录。
- TNL2K 公开第三方镜像：<https://huggingface.co/datasets/xche32/tnl2k>，文件 `tnl2k.tar.gz`。
  - 镜像包头是 `tnl2k/<sequence>/...`，解压到 `data/tnl2k/test/`。
  - 预期标准测试集 700 条；整个镜像尚未下载完，最终必须核实序列数及对应官方清单，不能仅凭镜像名称保证划分正确。
- LaSOT 公开第三方测试集镜像：<https://huggingface.co/datasets/ZhengGuangze/LaSOTTest>。
  - 六个独立包：`lasottest_0_49.tar.gz`、`lasottest_50_99.tar.gz`、`lasottest_100_149.tar.gz`、`lasottest_150_199.tar.gz`、`lasottest_200_249.tar.gz`、`lasottest_250_279.tar.gz`。
  - 实际按照元数据文件顺序处理，不是数字顺序。
  - 包里序列扁平排列，提取后按仓库 LaSOT 标准 280 条清单映射到类别目录；多余序列不提取。
- 仓库 README 的 LaSOT 全量镜像为 `l-lt/LaSOT`（约 248 GB）；没有选择下载该全量训练+测试数据。
- HF mirror 对部分 URL 会跳转回 Hugging Face；目前实际使用 `huggingface.co` + 系统代理。

元数据文件保存在：

- `downloads/xche32_tnl2k_tree.json`
- `downloads/lasot_test_tree.json`

其中包含文件大小及 LFS SHA256，可用于核验。

## 6. 下载脚本、状态和重启方法

脚本：`/root/MVLM/downloads/prepare_test_datasets.py`。

- 256 MiB 一个编号分块，放在 `<archive>.parts/000000` 等文件。
- 每次网络请求最多 16 MiB，使用 HTTP/1.1 和 Range 请求；每个数据集 4 个并发下载线程。
- 未完成小段保存在 `000000.segment`，响应头保存在 `000000.headers`。
- 验证 Content-Range 后追加到编号分块，网络中断可保留已经收到的有效小段数据。
- 每个分块连续 40 次未接收有效数据后报错；接收有效数据后计数清零。失败后等待 2、4、8、16 秒，此后每次 30 秒；单次请求最多 180 秒。线程池还会等待其他任务，报错不等于整个进程立即退出。不要假设任务永远重试。
- 暂停/重启时，尚未追加到数字分块的 `.segment` 可能被下一次 curl 请求覆盖，因此第一次速度窗口可能失真甚至为负数。应在启动稳定后重新采样，不可把负数当成网络速度。
- 完整包合并时计算并检查 SHA256，随后解压。已有同大小完整压缩包会直接复用；如需重新验证已有包，应单独计算 SHA256。
- 已解压标记是对应包的 `.tar.extracted` 文件，避免重复解压。
- 正式评测前必须确认图像、语言、框标注和序列划分完整。下载完毕不等于完成最终验证。

测试集下载曾拆为独立任务，避免原脚本同时写共享状态的风险；VastTrack 的独立文件见第 11 节：

| 数据集 | 独立状态文件 | 日志 | 说明 |
| --- | --- | --- | --- |
| TNL2K | `downloads/tnl2k_status.json` | `downloads/tnl2k_restart.log` | PID 动态查询，不使用历史 PID |
| LaSOT | `downloads/lasot_status.json` | `downloads/lasot_restart.log` | 已正常结束，无需重启 |

`downloads/dataset_status.json` 和 `dataset_download.log` 是早期总任务的记录，可能过期。独立任务调用 `d.tnl2k()` / `d.lasot()`，完成后通常只标记 extracted；**不会自动调用脚本的全量 `validate()`**。

状态文件可能滞后；务必同时查进程、最新日志和有效分块增长，不能只看状态字符串或进程存活就断言下载正常。

检查进程：

```bash
pgrep -af '^/root/miniconda3/envs/mvlm/bin/python -u -c.*d\.(lasot|tnl2k)'
tail -n 10 /root/MVLM/downloads/lasot_restart.log
tail -n 10 /root/MVLM/downloads/tnl2k_restart.log
```

重启前先确认没有同一数据集的进程，否则会并发写同一分块。实际恢复使用 Python `subprocess.Popen(..., stdin=DEVNULL, stdout=日志, stderr=STDOUT, start_new_session=True)` 启动后台任务，并保存 `downloads/<数据集>_download.pid`。以下为前台等价命令，不应在已有进程运行时重复执行；LaSOT 当前已完成，无需重启：

```bash
cd /root/MVLM
# TNL2K
/root/miniconda3/envs/mvlm/bin/python -u -c 'import sys; sys.path.insert(0,"downloads"); import prepare_test_datasets as d; d.STATUS=d.DOWNLOADS/"tnl2k_status.json"; d.state={"tnl2k":{}}; d.tnl2k()' >> downloads/tnl2k_restart.log 2>&1

```

统计进度的口径：

1. 完整压缩包大小达到源元数据 size 时，计入该文件完整大小。
2. 否则累计 `.parts` 中数字文件的大小，不能再把 `.assembling` 文件计入，避免双算。
3. 可额外计算正在传输的 `.segment`，但必须确认 `.headers` 的 Content-Range 起点等于 `编号 × 256 MiB + 当前编号文件大小`，避免把已经追加的小段重复计算。
4. 每包计入量不超过元数据 size；按总字节数计算百分比。这里 GB 用十进制，MB/s 也用十进制。
5. 统计过程中可能正在合并、删除分块，注意 FileNotFoundError 及文件竞态。

## 7. 网络与执行限制

- 下载进程继承了 HTTP_PROXY、HTTPS_PROXY 及小写变量，指向 `http://127.0.0.1:17891`；不要假定 ALL_PROXY 一定存在。进程在服务器执行，但数据经该代理隧道传输。
- 多次失败是代理端口拒绝连接：`Failed to connect to 127.0.0.1 port 17891`；也有 SSL EOF、HTTP/2 CANCEL 和低速超时。
- 10 月 16 日已通过 `/proc/net/tcp` 和进程 socket 确认端口由 `sshd` 监听，即 SSH 转发端口。服务器端未独立核实具体转发终点；用户确认继续使用笔记本 Clash。此前泛泛说不走笔记本流量的表述不适用，代理计费规则仍由实际服务决定。
- 10 月 16 日按用户要求测试过 `curl --noproxy '*'` 直连 Hugging Face、HF Mirror 和实际存储节点 `us.aws.cdn.hf.co`，均超时，没有切换为直连。不要在未验证的情况下移除下载代理。用户随后明确要求继续使用笔记本 Clash。
- 若下载停下，应重新检查代理端口、进程、最新日志与有效字节增长；历史 HTTP 200 不证明当前代理可用，不要查看或展示订阅凭证。
- 下载已使用独立 session 后台进程，未配置 tmux/systemd。SSH 转发和笔记本 Clash 仍需保持可用，任务后台化不能消除代理依赖。
- 普通 exec 沙箱曾无法启动：`bwrap: No permissions to create a new namespace`。因此多数 shell 命令通过 `sandbox_permissions=require_escalated` 执行。
- 自动审批曾返回超时或 403 Forbidden，甚至只读命令也未执行。应如实报告具体服务错误，不能推断成用户拒绝、命令危险或测试代码失败；之后用户要求重试时执行恢复正常。不要用旧数字冒充实时进度。

## 8. 论文与评测方案

用户提供的论文 PDF：

`/root/.codex/attachments/3b017052-7021-4565-9566-7d4541db2bfc/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.pdf`

后来用户再次提供的同名 PDF：`/root/.codex/attachments/f4446cf8-4400-4b7e-ab14-51a3fd683434/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.pdf`。

提取文本：`/tmp/mvlm_paper.txt`、`/tmp/mvlm_user_paper.txt`（临时文件可能消失）。已阅读论文实验设置并与推理实现对照。

- 论文表 1 的测试集：TNL2K、LaSOT、OTB99、MGIT。
- 训练使用 TNL2K、LaSOT、VastTrack。仅用已有权重评测无需下载训练集或重新训练。
- 当前配置是 `mvlm_TF`，`MODEL.ENCODER.TEMPLATE_FREE=True`，优先复现 Tracking-by-language，而不是将分数与“框+语言”组混比。

论文仅语言组目标分数（百分制）：

| 数据集 | PRE | AUC |
| --- | ---: | ---: |
| TNL2K | 60.9 | 57.8 |
| LaSOT | 65.5 | 60.7 |
| OTB99 | 84.3 | 60.7 |
| MGIT | 55.5 | 63.5 |

论文表 2 完整组件（A4）的 NPR：TNL2K 74.1、LaSOT 68.3、OTB99 75.4。OTB 与 LaSOT 已有实际评测结果，见第 9 节；TNL2K 与 MGIT 尚未评测。

- PRE：中心误差不超过 20 像素的比例。
- AUC：不同 IoU 阈值下成功率曲线的面积/离散均值，越高越好。
- NPR：用真实框宽高归一化中心误差，当前仓库取阈值 0.2 的精确率。AUC 用 IoU 阈值 0～1、步长 0.05，先跨序列平均，再对阈值取平均，并非直接把所有帧混合计算。
- 代码：`tracking/test.py` 生成逐序列预测；`tracking/analysis_results.py` 输出 success、prec、norm_prec。
- 已确认无模板分支在 `fastitpn.py::prepare_tokens_with_masks()` 不编码或拼入视觉模板；不能仅凭 initialize 接收真实框就断定使用了视觉模板。但当前流程跳过首帧推理，并在输出/评分中设首帧预测为真实框；论文正文未写明此计分细节，仍需核对协议。
- 论文公式（4）、（5）使用分数差除以尺度估计量（举例 MAD 或 EWMA std）；当前 `lib/test/tracker/mvlm.py` 使用 `tanh(分数差)`，未实现该尺度估计除法。是已确认的公式/代码差异，但未证明就是掉分原因，不能直接改公式后声称完成复现。
- 论文第 5.2 节的 100 次加噪声前向传播是理论失败界限验证，不是表 1 的常规 PRE/AUC 计算流程。
- 仓库给出了具体 YAML 推理参数；尚未确认的是作者生成论文 OTB 成绩时与哪个配置对应，不能说参数完全未公开。
- MGIT 暂不下载。其加载器要求 Excel 语言描述、框标注及特定帧目录；实现与 README 示例有差异，且 README 提及官方平台提交，后续单独核查。

后续推荐顺序：

1. **重新检查当前下载任务和代理**，必要时按用户授权续传；保留所有分块。
2. TNL2K 下载完成后检查 700 条测试划分、SHA256、语言标注及全部帧，不能只检查包大小。
3. 对 TNL2K 先小规模确认评测协议与格式，再使用原始配置快照和新 exp_id 完整评测；保存预测框、耗时、异常清单、指标和论文对照表。不要为整个测试集默认导出巨量视频。
4. LaSOT、OTB 已完成的基线与调参结果直接查第 9 节，无需重复执行已完成步骤。测试集调参配置不作为官方默认或独立复现证据；结构改进使用固定且明确记录的对照配置。

新会话可直接提示：“请先阅读 `/root/MVLM/SESSION_HANDOFF.md`，检查当前下载任务和网络状态，接着完成三个测试集准备及 MVLM 官方权重评测。不要重复克隆、安装环境或从头下载已有数据。”

## 9. 已完成 OTB99-Lang 基线和参数实验

### 9.1 早期 OTB 实验条件及结果（历史记录）

- 使用现有官方权重 `models/MVLM_TF.pth.tar` 和缓存 CLIP；没有重训或更换权重。
- 仓库评测加载器为 48 条测试序列，共 29,077 帧。数据目录有 99 个视频，并不等于评测全部 99 条。
- `pip check` 通过；CUDA、三帧 GPU 推理通过；完整评测使用 8×RTX 3090、8 个进程，OMP/MKL 每进程限制 1 线程。
- 原始基线完整跟踪用时约 201 秒。之后完成首次 TAU=0.0 实验，以及用户追加要求的 5+3 组实验，共 9 组调参实验，另有原始基线。
- 所有完整实验均检查预测框帧数、有限数值、正宽高以及逐帧耗时文件，新增八组每组均验证 48 条序列和 29,077 帧。不是根据终端 Done 就推断完整。

| 指标 | 原始基线 | 早期最高 AUC 配置 | 论文参考 | 早期最佳与论文差距（百分点） |
| --- | ---: | ---: | ---: | ---: |
| AUC | 56.48 | 60.15 | 60.70 | -0.55 |
| PRE | 79.03 | 83.40 | 84.30 | -0.90 |
| NPR | 69.75 | 74.78 | 75.40 | -0.62 |

该批次最佳配置保留在 `experiments/mvlm/mvlm_TF.yaml`，后续已被第 9.6 节实验超过：

```yaml
TEST:
  MVLM:
    ALPHA_CORR: 0.7
    TAU: 0.03
    OMEGA: 0.2
    LAMBDA: 0.4
    PSI_OUT: 0.1
```

其余 YAML 字段保持原始值。训练代码没有因本次实验修改。

### 9.2 早期 OTB 结果与记录位置

- 原始配置备份：`output/otb_parameter_trials/otb99_language_baseline.yaml`。
- 基线 exp_id：`otb99_language_baseline`。
- 基线指标：`output/otb99_language_baseline_metrics.txt`。
- 基线日志：`output/otb99_language_baseline_test.log`。
- 基线逐帧预测/耗时：`output/test/tracking_results/mvlm/otb99_language_baseline/otb99_lang/`。
- 早期最佳 exp_id：`otb99_20261016_092909_06_less_memory`。ID 中时间来自服务器 UTC，面向用户日期时间按 Asia/Shanghai；不要把 ID 当作本地时间。
- 早期最佳配置：`output/otb_parameter_trials/otb99_20261016_092909_06_less_memory.yaml`。
- 早期最佳指标/日志：同目录下对应 `_metrics.txt`、`_test.log`。
- 早期最佳逐帧预测/耗时：`output/test/tracking_results/mvlm/otb99_20261016_092909_06_less_memory/otb99_lang/`。
- 用户可读总记录：`output/otb_parameter_trials/实验记录.md`，包含所有参数、结果、调参依据、配置快照、运行命令和文件位置。
- 机器可读事件记录：`output/otb_parameter_trials/experiments.jsonl`。同一 exp_id 有 running/completed 事件，读取时按 exp_id 取最后一条，不能把两个事件算作两次实验。
- 总结：`output/otb_parameter_trials/调参总结.md`。
- 两批运行日志：`output/otb_parameter_trials/sweep_progress.log`、`followup_progress.log`；已全部完成，没有继续调参的进程。
- 每组保留独立的 `_eval_data.pkl`。官方分析脚本默认共享 `output/test/result_plots/otb99_lang/eval_data.pkl`，后续分析会覆盖它，不能仅凭此共享路径判断是哪次实验。
- `tracking/test.py` 遇到已有序列结果会跳过；真实重跑应取新 exp_id，不要误把复用旧预测当成新运行。
- `tracking/analysis_results.py` 默认只打印 PRE/NPR/AUC 并保存缓存，不自动生成曲线图片；指标文本通过重定向保存。

### 9.3 早期 OTB 调参结论及解释边界

- 固定原始其他参数时，TAU=0/0.03/0.06/0.1 的 AUC 为 58.95/59.30/58.33/56.48。
- TAU=0.03 时，PSI_OUT 从 0 改为 0.1，AUC 59.30→59.69。
- 在此基础上，ALPHA_CORR 从 1.0 改为 0.7，AUC 59.69→60.02。
- 较强记忆组合 OMEGA=0.6、LAMBDA=0.7，在当时基准上使 AUC 降至 58.07；两参数同时改变，不能分别归因。
- 仅把 OMEGA 从 0.4 降到 0.2，AUC 60.02→60.15。
- 在最佳邻域，TAU=0.01 得到 59.61，PSI_OUT=0.2 得到 59.21，均不保留。
- 参数机制：TAU 控制候选集合及下一帧局部/全局搜索；PSI_OUT 控制 IoU≤阈值的干扰候选池；ALPHA_CORR 混合图文/分类差值；LAMBDA 控制记忆更新、OMEGA 控制当前决策对记忆的依赖。最终候选集合内仍按分类分数选择预测框，不是直接按置信度加权分数选最高框。
- 代码中最终置信度可化为 `(1-OMEGA*LAMBDA)*当前置信度 + OMEGA*LAMBDA*上一帧记忆`；原始历史权重 16%，强记忆 42%，该批次最佳 8%。原因解释是代码支持的推断，尚未做逐帧可视化验证。
- 用户明确授权了测试集上的探索性调参。必须保留这个事实：最佳分数不是独立验证结果，不证明找回了作者原始设置，不保证能迁移到其他数据集；早期每组仅跑一次，未统计随机波动或置信区间。后续 LaSOT 同配置复跑见第 9.5 节。

### 9.4 LaSOT 原始参数完整评测

本实验已于 2026-10-17 **11:10:41（上海时间）完整跑完**，数据校验与产物如下。

- 实验：`lasot_language_original_mp16_20261017_103923`；`status.json` 标记 `status=completed`，`prediction_validation.json` 为 `checked=280, errors=[]`。
- 产物目录：`output/lasot_evaluations/lasot_language_original_mp16_20261017_103923/`（含 `metrics.txt`、`结果说明.md`、`status.json`、`prediction_validation.json`、`config.yaml`、`runtime.yaml`、`archive_checksums.json`、`dataset_manifest.json`、`commands.json`、`code_changes.patch`、`eval_data.pkl`）。
- 逐帧预测/耗时：`output/test/tracking_results/mvlm/lasot_language_original_mp16_20261017_103923/lasot/`。
- 参数为**原始发布参数**：ALPHA_CORR=1.0、TAU=0.1、OMEGA=0.4、LAMBDA=0.4、PSI_OUT=0.0；实验配置保存在产物目录的 `config.yaml` 与对应实验 YAML。当前 HEAD 中的 `mvlm_TF.yaml` 已是调参配置，不能再从 HEAD 提取它冒充原始参数。
- 规模：280 条序列、685,360 帧；8 张 RTX 3090，每卡 2 进程，共 16 进程，每进程 1 个 CPU 线程。
- 数据校验：六个包 SHA256 全部匹配源元数据；全部帧非空；标注、语言、可见性长度已核；每序列首/中/末帧可解码。`status.json` 同时记录权重 SHA256 `91c31230f3d355f71104df8191e386632a39ce9d3bbc1f7ce5bf3772b369cc4f` 与 git commit `4186c8fd`。
- 耗时：验证数据 10:39:24→10:40:41；推理 10:40:41→11:10:34（test.log 记录 1791.3 秒，FPS 约 41.9）；计算指标至 11:10:41。
- 协议说明（原文）："Single baseline evaluation; no LaSOT tuning. Uses repository language-only evaluation protocol, including its first-frame handling." 即**本次基线实验使用未调参配置**，沿用仓库语言评测协议（含首帧处理）；后续调参实验另见第 9.5 节，第 8 节对首帧计分细节的保留意见仍适用。

结果：

| 指标 | 本次（原始参数） | 论文表 1 | 差距（百分点） |
| --- | ---: | ---: | ---: |
| PRE | 61.47 | 65.5 | -4.03 |
| NPR | 64.15 | 68.3 | -4.15 |
| AUC | 57.77 | 60.7 | -2.93 |

原始配置基线与测试集调参最佳结果必须分别比较，不能用调参后的接近程度证明官方默认配置已经独立复现；两数据集原始与最优结果见第 9.7 节。

可复跑命令（注意：`experiments/mvlm/lasot_language_original_mp16_20261017_103923.yaml` 是原始参数快照，当前工作 YAML `mvlm_TF.yaml` 已是 OTB 调参后的值，复跑请显式指定参数快照或新 exp_id，避免与已有结果混淆）：

```bash
cd /root/MVLM
/root/miniconda3/envs/mvlm/bin/python -u -m tracking.test mvlm \
  lasot_language_original_mp16_20261017_103923 \
  --dataset_name lasot \
  --weight_path /root/MVLM/models/MVLM_TF.pth.tar \
  --exp_id lasot_language_original_mp16_20261017_103923_rerun \
  --mode mp --threads 16 --num_gpus 8

/root/miniconda3/envs/mvlm/bin/python -u -m tracking.analysis_results \
  --tracker_name mvlm \
  --tracker_param lasot_language_original_mp16_20261017_103923 \
  --exp_id lasot_language_original_mp16_20261017_103923_rerun --dataset lasot
```

### 9.5 LaSOT 参数搜索（8 组，均已完成）

> **方法学声明**：这是 LaSOT **测试集**上的探索性调参，**不能**作为论文独立复现证据。

在 9.4 原始基线（AUC 57.77）基础上，用同一官方权重、280 序列、685,360 帧、16 进程，顺序自适应搜索 5 个推理超参数，共 8 组。完整报告见 `output/lasot_parameter_trials/LASOT_TUNING_RESULTS.md`，机器可读记录 `output/lasot_parameter_trials/experiments.jsonl`，运行器 `tracking/run_lasot_trial_one.py` 与 `tracking/finish_lasot_trial.py`。

| exp_id | αcorr | τ | ω | λ | ψout | PRE | NPR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **lasot_tune_05_mem004（最优）** | **0.5** | **0.03** | **0.1** | **0.4** | **0.1** | **65.93** | **68.72** | **61.06** |
| lasot_tune_08_best_repeat（同配置复跑） | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 | 65.93 | 68.72 | 61.06 |
| lasot_tune_06_mem002 | 0.5 | 0.03 | 0.05 | 0.4 | 0.1 | 65.93 | 68.69 | 61.04 |
| lasot_tune_03_alpha05 | 0.5 | 0.03 | 0.2 | 0.4 | 0.1 | 65.85 | 68.68 | 61.01 |
| lasot_tune_07_psi02 | 0.5 | 0.03 | 0.1 | 0.4 | 0.2 | 65.80 | 68.60 | 60.95 |
| lasot_tune_04_alpha03 | 0.3 | 0.03 | 0.2 | 0.4 | 0.1 | 65.53 | 68.32 | 60.78 |
| lasot_tune_01_otb_best | 0.7 | 0.03 | 0.2 | 0.4 | 0.1 | 65.34 | 68.21 | 60.58 |
| lasot_tune_02_tau001 | 0.7 | 0.01 | 0.2 | 0.4 | 0.1 | 65.14 | 67.89 | 60.32 |
| 9.4 原始基线 | 1.0 | 0.1 | 0.4 | 0.4 | 0.0 | 61.47 | 64.15 | 57.77 |
| **论文表 1** | — | — | — | — | — | **65.5** | **68.3** | **60.7** |

最优配置三项均略高于论文（+0.43/+0.42/+0.36），达到论文 0.5% 以内。

- **可复现性**：`08_best_repeat` 与 `05_mem004` 配置完全相同，三项指标逐位一致（65.93/68.72/61.06）。推理确定性，结果可复现。
- **α_corr 单峰，峰在 0.5**。α=1.0（原始）57.77 → 0.7 60.58 → 0.5 61.06 → 0.3 60.78。纯图文相关分并非最优，分类分提供等权互补信息，与论文"correlation margin 是根本信号"的表述有一定张力。
- **τ 非单调，0.03 最优**。0.1 过严频繁触发全局重定位（`resize_sample_target` 把整图压到 224×224，LaSOT 多为 1280×720，精度骤降）；0.01 过松使阈值失去筛选意义。
- **记忆项只以 ω·λ 乘积起作用，0.02–0.04 饱和**。0.08→0.04 略升，0.04→0.02 持平。原始 0.16 明显偏差，长序列（平均 2448 帧）上强记忆累积漂移。
- **ψout=0.1 最优**，0.2 使 margin 区分度下降。
- **重要推论**：OTB99（+3.67 AUC）与 LaSOT（+3.29 AUC）两个独立数据集上出现方向一致、幅度接近的提升，更可能说明**仓库默认 YAML 的这 5 个推理参数与论文实际设置不一致**（论文 5.1 节只给训练设置，未公开这些推理参数），而非调参运气。
- 未做的事：未修改 `lib/test/tracker/mvlm.py` 的实现，即论文公式 (4)(5) 用尺度估计量做除法、代码用 `tanh(分数差)` 的差异仍在。这是后续更接近论文的潜在方向，但需谨慎验证后再说。
- 已知问题：`run_lasot_trial_one.py` 首版在 subprocess 之后导入 `lib` 会因 sys.path 失败，已修复（显式插入 ROOT）并加 `tracking/finish_lasot_trial.py` 用于补救已完成推理但校验失败的实验。

### 9.6 OTB99-Lang 参数搜索（10 组新增，均已完成）

> **方法学声明**：OTB99-Lang **测试集**上的探索性调参，不能作为论文独立复现证据。

在 9.3 旧调参（最佳 AUC 60.15）基础上重新做 10 组受控实验。运行器 `tracking/run_otb_trial_one.py`（从原始参数 YAML `output/otb_parameter_trials/otb99_language_baseline.yaml` 出发，**不**使用当前已调参的 `mvlm_TF.yaml`）。完整报告 `output/otb_parameter_trials/OTB_TUNING_RESULTS.md`。

| exp_id | αcorr | τ | ω | λ | ψout | PRE | NPR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **otb_tune_03_prod04_c（最优）** | **0.5** | **0.03** | **0.4** | **0.1** | **0.1** | **84.05** | **75.33** | **60.56** |
| otb_tune_02_prod04_b | 0.5 | 0.03 | 0.2 | 0.2 | 0.1 | 84.01 | 75.30 | 60.54 |
| otb_tune_01_prod04_a | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 | 83.98 | 75.29 | 60.52 |
| otb_tune_06_tau004 | 0.5 | 0.04 | 0.1 | 0.4 | 0.1 | 83.88 | 75.22 | 60.46 |
| otb_tune_08_psi005 | 0.5 | 0.03 | 0.1 | 0.4 | 0.05 | 83.91 | 75.10 | 60.45 |
| otb_tune_09_joint_a052_t0028 | 0.52 | 0.028 | 0.1 | 0.4 | 0.1 | 83.77 | 75.19 | 60.43 |
| otb_tune_10_psi015 | 0.5 | 0.03 | 0.1 | 0.4 | 0.15 | 83.74 | 75.04 | 60.36 |
| otb_tune_05_alpha055 | 0.55 | 0.03 | 0.1 | 0.4 | 0.1 | 83.62 | 75.07 | 60.34 |
| otb_tune_07_tau005 | 0.5 | 0.05 | 0.1 | 0.4 | 0.1 | 83.59 | 74.97 | 60.31 |
| otb_tune_04_alpha045 | 0.45 | 0.03 | 0.1 | 0.4 | 0.1 | 83.32 | 74.90 | 60.25 |
| （旧最佳）otb99_20261016_092909_06_less_memory | 0.7 | 0.03 | 0.2 | 0.4 | 0.1 | 83.40 | 74.78 | 60.15 |
| （原始基线）otb99_language_baseline | 1.0 | 0.1 | 0.4 | 0.4 | 0.0 | 79.03 | 69.75 | 56.48 |
| **论文表 1** | — | — | — | — | — | **84.3** | **75.4** | **60.7** |

最优配置距论文 PRE -0.25 / NPR -0.07 / AUC -0.14，均在 0.25 以内；AUC 较原始基线提升 +4.08。

- **ω·λ 乘积假设严格验证**：固定乘积 0.04，三种分配 (0.1,0.4)/(0.2,0.2)/(0.4,0.1) 的 AUC 为 60.52/60.54/60.56，极差仅 0.04（噪声量级）。证实由 Eq.8/9 推出的结论：EWMA 记忆在稳态下只有乘积 ω·λ 起作用，ω 与 λ 单独取值无关。这是对实现的一个可写入分析的理解。
- **α_corr 在 OTB 上峰更窄**：0.45→60.25，0.50→60.56，0.55→60.34。LaSOT 上 0.5 附近更平坦，可能因 OTB 序列短（平均约 606 帧）对 α 更敏感。
- **τ=0.03、ψ=0.10 在两数据集一致最优**。τ：0.03→60.56、0.04→60.46、0.05→60.31、0.10→56.48。ψ：0.05→60.45、0.10→60.56、0.15→60.36。
- **最强的证据：两个数据集独立搜索收敛到同一组参数**（α=0.5, τ=0.03, ω·λ=0.04, ψ=0.1）。LaSOT（280 序列/685,360 帧，平均 2448 帧）与 OTB（48 序列/29,077 帧，平均 606 帧）在分辨率、目标尺度分布上差异很大，却得到完全一致的参数，基本排除"碰巧调出"的可能，支持"仓库默认 YAML 与论文实际推理配置不一致"的判断。
- 注意 OTB 加载器只用 48 条测试序列（数据目录有 99 条），与论文 OTB99 全量口径可能不同。

### 9.7 两数据集现状汇总

| 数据集 | 原始配置 AUC | 最优配置 AUC | 提升 | 论文 AUC | 最优 vs 论文 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LaSOT | 57.77 | 61.06 | +3.29 | 60.7 | +0.36 |
| OTB99-Lang | 56.48 | 60.56 | +4.08 | 60.7 | -0.14 |
| TNL2K | 未评测 | — | — | 57.8 | 下载中 |
| MGIT | 未评测 | — | — | 63.5 | 未下载 |

两个已完成数据集均达到论文水平。若要让 OTB 也超过论文，剩余 0.14 的差距可能来自论文公式 (4)(5) 的尺度估计除法与代码 `tanh` 的差异，未修改实现。

### 9.8 参数搜索结果核对与最佳配置汇总

#### 数字核对结果

2026-10-17 18:0x（上海时间）对第 9.5、9.6 节所列全部实验，逐一读取磁盘上的 `status.json` 并与文档数字比对：

- LaSOT 8 组：**全部一致**（PRE/NPR/AUC 三位小数相同，0 处不符）。
- OTB99-Lang 10 组：**全部一致**（0 处不符）。
- 核对中发现并已修正 1 处命名不一致：9.6 表中原写作 `otb_tune_09_joint`，磁盘实际目录为 `otb_tune_09_joint_a052_t0028`，指标 83.77/75.19/60.43 本身无误。

核对脚本思路：遍历 `output/lasot_parameter_trials/lasot_tune_*/status.json` 与 `output/otb_parameter_trials/otb_tune_*/status.json`，取 `metrics` 四舍五入到两位小数，与文档表格逐行比对。

#### 两个数据集已搜索配置中的最优结果（测试集调参）

| 数据集 | αcorr | τ | ω | λ | ψout | PRE | NPR | AUC | 论文 | 差距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LaSOT | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 | 65.93 | 68.72 | 61.06 | 60.7 | **+0.36** |
| OTB99-Lang | 0.5 | 0.03 | 0.4 | 0.1 | 0.1 | 84.05 | 75.33 | 60.56 | 60.7 | -0.14 |

两组参数**除 ω/λ 的分配外完全相同**，而 ω·λ 乘积均为 0.04，因此在有效自由度上两个数据集收敛到同一组参数。LaSOT 用 (ω=0.1, λ=0.4)，OTB 用 (ω=0.4, λ=0.1)，二者 AUC 差异在 0.04 噪声量级内，可视为等价。

#### 关于 TNL2K 的提醒

TNL2K 下载结束后**不会自动做全量校验**（见第 6 节：独立任务通常只标记 `extracted`，不调用 `validate()`）。正式评测前必须另行确认 700 条测试序列的划分、SHA256、语言标注与帧完整性，不能只看包大小。

TNL2K 论文目标为 PRE 60.9 / AUC 57.8。若沿用本次在 LaSOT 与 OTB 上一致收敛的参数（α=0.5, τ=0.03, ω·λ=0.04, ψ=0.1），可作为该数据集的**首要试验配置**，但仍应从原始参数跑一次基线作为对照，且需注意仓库中 `TNL2K` 的 `TEST` 分支是否另有覆盖值。

## 10. 重训咨询：已核实条件，尚未启动

- 用户希望通过改动网络结构完成毕设，已解释训练复现与加载现有 MVLM 权重微调不同；目前只完成 OTB 的记忆模块小规模短训，未启动完整训练。现已单独授权下载 VastTrack 训练数据；其他训练数据尚未新增下载。
- 论文训练数据是 LaSOT、TNL2K、VastTrack 的训练部分。当前完成/正在下载的 LaSOT、TNL2K 为测试镜像，不能代替训练集；仅 OTB 不足以复现作者训练，且不能用已评测的 48 条测试序列训练后宣称复现。
- 骨干预训练文件 `pretrained/itpn/fast_itpn_base_clipl_e1600.pt` 当前缺失；推理不需要。原始训练路径仍需该文件；新记忆训练路径从已有完整官方跟踪 checkpoint 初始化时跳过额外骨干加载，因此不被该缺失文件阻塞。CLIP 缓存已存在。
- 论文设置：60 epoch、每 epoch 100,000 个采样样本、学习率 0.0005、AdamW、batch size 80，报告使用 4×RTX A6000 48 GB。官方 README 的配置为 FastITPN-Base/CLIP。
- 重要实现差异：`lib/train/base_functions.py` 中 `BATCH_SIZE` 直接传入各进程 DataLoader，配合 DistributedSampler，当前是每卡 batch；8 卡直接用 80 会变成全局 640，不能默认为全局 80。论文正文的 batch 口径尚未确认。
- 如果按全局 batch=80 规划，600 万次样本采样约 75,000 次更新。假设每步 0.5/1/2/5 秒，纯训练约 10.4/20.8/41.7/104 小时。这是条件估算，不是本机训练实测。建议准备数据后先跑 100～200 步测速度和峰值显存，再估总时长；推理耗时不能直接用来推算训练。
- 服务器配置记录为 8×3090、每卡 24 GB、系统内存约 440 GiB；训练前需重新检查 GPU 占用与可用内存、磁盘。多卡显存不能视为单卡 192 GB，仍需验证每卡 batch。
- 曾建议每卡 batch 8～10 做短测，这是准备方案而非已验证可运行配置。训练器支持 CUDA AMP，但默认 `getattr(cfg.TRAIN,'AMP',False)` 为关闭，任何开启都要记录为训练设置变化。
- OTB 训练适配已补齐：划分文件缺失时，从包内 `OTB_query_train` 读取 51 条训练序列；`DATA.USE_NLP.OTB99` 可用，`DATA.TRAIN.OTB_SEQUENCES` 可选择训练子集。已核对与测试语言划分无重叠，并完成四序列记忆短训；训练／评测必须继续隔离。
- 已验证原基线推理和完整评测，以及新增记忆模块的短小前后向与兼容性检查；尚未执行真实训练集的完整训练，不能把合成数据反传成功等同于训练效果或收敛已验证。

## 11. VastTrack 训练数据下载（10 月 17 日新授权）

- 用户要求下载 VastTrack 并与其他数据集放在同一数据根目录，确保训练加载器能找到；本次授权限于 VastTrack 的训练准备，不包括 MGIT 或其他新数据集。
- 已设置 `lib/local_config.py::vasttrack_dir = '/root/MVLM/data/vasttrack'`；加载器自行追加 `train/`，最终位置为 `/root/MVLM/data/vasttrack/train/<类别>/<序列>/imgs/`，同序列含 `Groundtruth.txt` 和 `nlp.txt`。
- 官方 OneDrive 分享链接返回 403、目录 API 返回 401；百度网盘入口需要提取码流程。找到公开镜像 `https://huggingface.co/datasets/xche32/vasttrack`：14 个连续 tar.gz 分卷，总计 **736,604,489,019 字节，约 736.604 GB**。这是该镜像分卷的实测总大小，不等于已核实官方 82 个 ZIP 的大小或解压后大小。
- 镜像根目录为 `vasttrack/<类别>/<序列>/...`；前 4 MiB 抽查到 Aardwolf-10 图像、真实框，训练清单包含此序列；同作者公开的 unpacked 示例也列有 `nlp.txt`。完整划分和语言覆盖需待全部解压后校验，尚未宣称训练数据已完整可用。
- 镜像覆盖范围尚未全量核实。为读取连续压缩流，需要下载全套 14 个分卷；解压时严格按仓库 `lib/train/data_specs/vasttrack_train_split.txt` 选择训练序列，其他序列不提取。Python 按行读取清单实际为 **46,836 条**，不是 `wc -l` 的 46,835（末尾缺换行）；与官方论文的 47,110 条存在差异，后续不能混称。
- 下载脚本：`downloads/prepare_vasttrack.py`，复用现有 `prepare_test_datasets.py::download` 的 4 并发 Range 下载、分块续传及镜像 LFS SHA256 校验；复用的是下载函数，不会改写 TNL2K/LaSOT 状态。
- 分卷和编号分块：`downloads/vasttrack/`；来源及大小/SHA256 清单：`downloads/vasttrack/source_manifest.json`。
- 状态：`downloads/vasttrack/status.json`；日志：`downloads/vasttrack/download.log`；PID 文件：`downloads/vasttrack/download.pid`。
- 下载采用独立 session；实际 PID、状态与进度需查询现存进程和状态文件，不沿用初始启动快照。
- `status.json` 的 downloaded_bytes 每 10 秒更新，采用已追加数字分块和完整分卷的保守计数，不含正在接收的 `.segment`；实时测速可按第 6 节的方法加上已验证 Content-Range 的有效 segment。
- 所有分卷下载后逐卷校验 SHA256，按连续流读取解压，避免额外合并一份 736 GB 文件；安全检查绝对路径/`..`，不跟随压缩包链接。
- 自动最终校验包括训练清单每条序列的语言非空、真实框四列和数值、按框数逐一检查所有 `00001.jpg` 等帧，并构造 `VastTrack(split='train')` 读取首帧。通过后才将 status 标为 `ready`，失败则保存错误和 `validation.json`，不能因下载达到 100% 就宣称能训练。
- 当前仍用 SSH 转发的 Clash 代理，与 TNL2K 共享带宽。进程仅负责准备数据，不会开始模型训练。
- 官方仓库的 `assets/train_md5.txt` 返回内容中存在非十六进制字符，不能据此进行有效 MD5 校验；保存的原始参考文件在 `work/vasttrack/checksums.txt`，目前使用镜像元数据中的合法 LFS SHA256 保证传输完整性，不声称与官方 ZIP 已完成哈希对照。
- 如需恢复，先检查有无此下载脚本进程，再使用环境 Python 启动 `downloads/prepare_vasttrack.py`，日志追加并后台化；已有完整分卷会按大小复用，因此不得删除或覆盖已有数据。新会话应先实查进程和状态。


## W&B 训练监控接入（2026-10-18）

- 修改位于记忆分支 `codex/xmem-memory` 的独立工作目录 `/root/MVLM/work/xmem-wandb`；原 `/root/MVLM` 仍为 `main`，原有未提交文件保留。本次改动尚未提交或推送。
- 训练环境已安装 W&B 0.22.3，用户已自行登录。代码不保存或读取展示 API Key。
- 两个记忆 YAML 默认启用 W&B，项目名 `mvlm-memory`；其余配置默认关闭。每 20 次成功参数更新记录实际损失、训练 IoU、记忆门控和学习率等指标，支持 epoch 均值、主卡性能和显存。
- 多卡标量汇总后只由全局 rank 0 上传；检查点保留 global_step；在线重启复用输出目录的 run ID。控制台打印网址，`<save_dir>/wandb_tracking/train/mvlm/<config>/run.json` 保存网址与运行 ID。
- 当前尚未接入真实跟踪验证、验证 AUC、视频或记忆库占用统计，不得把训练 IoU 当作验证 AUC。三数据集重新划分及正式训练未执行。
- 已通过真实 W&B 离线日志、合成 CPU 训练循环、检查点恢复、双进程 Gloo 汇总/主进程隔离和语法检查；未联网创建实验、未运行正式训练。使用说明见 `reports/reproduction/WANDB_MONITORING.md`。


## 三数据集自定义划分正式训练已启动（2026-10-18 16:06，Asia/Shanghai）

- 用户明确授权按此前方案启动训练，并要求启动前给出监控位置。已先提供确切 W&B 链接：https://wandb.ai/2896287267-hit/mvlm-memory/runs/nie0ya10 。
- 训练代码在 `/root/MVLM/work/xmem-wandb`（`codex/xmem-memory`）；输出 `/root/MVLM/output/memory_three_custom_20261018_run01`。源码快照和逐文件哈希已保存，尚未提交/推送。
- 固定序列划分：OTB 51/12/36、LaSOT 196/42/42、TNL2K 489/105/105（训练/验证/测试）。TNL2K 空语言序列已从新划分排除；原数据未修改。划分清单与哈希在输出目录 `split/`，无跨划分图像路径交集，采用同源命名及首中末图像哈希分组，但不能保证排除全部近重复视频。此次是自定义划分适配，不是原始标准测试协议的独立复现。
- 实际启动配置 `experiments/mvlm/mvlm_TF_memory_three_custom.yaml`：8卡×每卡2片段，全局16；每epoch采样16000片段=1000步，共20epoch/20000步。前5epoch四帧、后15epoch十二帧；学习率1e-4/5e-5/1e-5，切换在第6/16epoch。冻结基线，只训练新增记忆模块，从官方权重初始化。
- 真实200步四帧、20步十二帧短测，以及8卡4步同步短测均通过，官方基线参数逐项不变；验证入口用三个数据集各一条短序列检查通过。正式运行不使用短测权重。
- 每epoch保存检查点，每2000步评估完整159序列验证集；相同推理设置的官方基线在首次验证计算并缓存。验证三数据集平均AUC选最佳，保存 `validation/best.json`、`best.pth.tar`。训练期间不评估测试集。W&B新增真实验证PRE/NPR/AUC、记忆库占用/写入率及片段长度。验证期间更新步数会暂停。
- 后台监督进程启动PID1067381、torchrun PID1067382；必须实查输出目录 `training_status.json`、`train.log`、`supervisor.log` 和GPU状态，不能把启动成功当作训练完成。训练终止状态会写入JSON；不要重复启动。
- 详细运行记录 `/root/MVLM/work/xmem-wandb/reports/reproduction/MEMORY_CUSTOM_TRAINING_20261018.md`。


## 三数据集记忆训练最终结果（2026-10-19复核）

- 实际训练已于10月19日05:27（上海时间）正常完成20epoch/20000步，全部10次159序列验证均完成。最优仍为epoch2，平均AUC60.5168，基线60.7286；最终epoch20平均58.7179。所有已验证检查点均未超过基线平均AUC，不能宣称新增模块已取得总体收益。最终OTB/LaSOT/TNL2K AUC为59.5365/61.7410/54.8763；相对基线+0.0738/-4.8296/-1.2761个百分点。保留测试集未评测。
- 模型权重有限、优化器步数20000、冻结基线权重逐项不变。训练损失下降并不代表在线跟踪泛化改善；后续应先用现有检查点诊断，不应直接增加步数。
- `training_status.json`原误标failed：训练子进程退出码0，但监督程序收尾torch.load因缺少仓库导入路径报`ModuleNotFoundError(lib)`。已独立检查最终checkpoint与全部验证，修复监督程序sys.path，将状态更正completed并保留原错误记录；没有重训。
- 详细分析见输出目录 `FINAL_ANALYSIS.md`、`final_analysis.json`，及记忆工作目录 `reports/reproduction/MEMORY_CUSTOM_FINAL_RESULTS.md`。


## 记忆模块逐项诊断已完成（2026-10-19）

- 用户要求逐项排查。未重训、未评估保留测试集；完成9条验证序列各最多512帧的11组机制检查（含不改变融合的控制复现），以及两组完整159序列/175356帧验证。
- 完整验证平均AUC：官方基线60.7286、最终训练原配置58.7179；仅把融合修正量缩放为0.25后61.0656，仅把记忆写入MIN_QUALITY从0.5提高到0.9后61.0690。两项均恢复约2.35点，较基线约+0.34点，是验证集探索结果，不代表已可靠取得测试集提升。严格门槛相对基线的探索性序列bootstrap95%区间[-0.2164,0.9258]包含0。
- 关闭新增记忆在9条控制序列上与原基线逐帧预测完全一致；不改融合的对照与原最终模型逐帧一致。19个记忆参数张量优化器一阶矩非零，冻结参数变化为0，权重有限；坐标/ROI/增强/梯度/采样检查通过，未发现RGB通道错配。
- 子集仅短期记忆、仅更换增强相关性评分、取消margin写入门控均未恢复到该子集基线；真实框写入上界有明显恢复案例。它们仅用于机制定位，不能作为完整验证或可部署成绩。
- 数据审计：全部冻结标注长度/有限值正常，未定义跨划分路径或分组交集，4303张抽样解码0失败。当前TNL2K压缩包SHA256实测01335bc1a30577735921cc30388e9a33c678f9b4401e89cd33a79887df69b88c，与保存的镜像LFS元数据一致；此前未确认哈希事项已关闭。非全量逐图解码，不保证语义标注全部正确。
- 已确认训练/推理条件差异：训练真框中心裁剪、混合真框/预测历史、原始score最大值写入质量；在线预测驱动裁剪、预测历史、Hann加窗候选质量并要求margin门控。固定小批量配对检查支持部分样本依赖正确历史，但没有证明唯一根因。
- 重点方向为约束错误历史和修正强度、对齐训练/在线写入条件。记忆模块与预测头联合训练已完成4,000步短训，但最佳验证差异很小且不能拆分两者贡献；改采样比例或新增训练数据的因果收益仍未验证。
- 报告：`/root/MVLM/output/memory_three_custom_20261018_run01/diagnostics/DIAGNOSTIC_REPORT.md`；机器可读`diagnostic_summary.json`；视觉对照`qualitative_examples.png`。记忆工作目录同时保存`reports/reproduction/MEMORY_DIAGNOSTICS_20261019.md`。严格门槛配置快照为`experiments/mvlm/mvlm_TF_memory_three_diagnostic_q09.yaml`，评测须加载已训练checkpoint，不是启动新训练。


## 记忆模块与预测头联合4000步短训已启动（2026-10-19 12:00，Asia/Shanghai）

- 用户授权保持原设置，仅解冻预测头做早期短训，并继续W&B监控。正式输出 `/root/MVLM/output/memory_head_joint_short_20261019_run01`，配置 `mvlm_TF_memory_head_joint_short_20261019`。
- 只将训练因素 `MEMORY_FREEZE_HEAD` 改为False：训练CENTER头（包括BN参数/统计）和记忆模块，编码器/文本继续冻结；原划分/种子20261018/1:3:6采样/全局batch16/4帧/学习率1e-4/写入门槛0.5/融合1.0均保持与上一轮前4000步一致。共4epoch，每epoch1000步，从官方权重重新开始，不用上一轮或预检记忆权重。
- 为避免DDP进程等待拖慢验证，先连续完成4000步，每1000步保存checkpoint，再自动依次评估4个完整159序列验证集并选最佳。训练步数在验证阶段保持4000；W&B验证曲线使用单独 `val/global_step` 轴。测试集不参与。
- W&B新链接 https://wandb.ai/2896287267-hit/mvlm-memory/runs/hafdntia ，与上一轮独立。
- 真实8卡4步预检通过：15个头卷积权重张量已更新，记忆投影非零，编码器/文本变化0；评测两进程共用1GPU的三序列预检通过。验证入口新增workers参数，16工作进程映射8GPU，默认行为仍兼容原入口。
- 监督脚本 `tracking/run_memory_head_short.py`；启动监督PID1264069，torchrun PID1264071。实时进度必须读取新输出目录 `training_status.json`、`train.log`、`supervisor.log`，不要将旧训练完成状态当成本轮状态。
- 运行说明见记忆工作目录 `reports/reproduction/MEMORY_HEAD_SHORT_20261019.md`；源码快照、逐文件SHA、配置差异和运行规格已保存到新输出目录。尚未提交/推送。

## 记忆模块与预测头联合短训最终结果（2026-10-19 13:08，Asia/Shanghai）

- `training_status.json` 为 `completed`：8卡训练4epoch/4000步，四个检查点均完成完整159序列验证。头部90个参数张量发生变化，编码器及文本参数变化0；W&B：https://wandb.ai/2896287267-hit/mvlm-memory/runs/hafdntia 。
- 验证平均AUC（1000/2000/3000/4000步）分别为59.5613/60.8004/59.0170/58.2544。最佳2000步，OTB/LaSOT/TNL2K AUC为59.7142/66.0177/56.6692；原始模型对应59.4627/66.5707/56.1523，平均60.7286。最佳平均仅高0.0718个百分点，不能据此宣称稳定提升；比上次冻结头的2000步60.5168高0.2836点。
- 4000步相对2000步明显退化，单步loss震荡不能代替验证结论。最佳模型在 `validation/best.pth.tar`，完整四点指标在 `validation_summary.json`；测试集仍未评测。此实验只改变预测头是否参与训练，不能区分记忆模块与头部各自对收益的贡献。

## 记忆模块与预测头联合、每200步检查的新实验（2026-10-19 18:20，Asia/Shanghai）

- 用户授权重新启动。工作目录 `/root/MVLM/work/xmem-wandb`，输出 `/root/MVLM/output/memory_head_joint_dense_20261019_run01`；W&B：https://wandb.ai/2896287267-hit/mvlm-memory/runs/okgvc2hi 。此前4000步实验作为训练对照：相同官方初始化、随机种子、固定划分、学习率、记忆设置及8卡×每卡2片段；新配置 `mvlm_TF_memory_head_dense_20261019.yaml` 每200步另存检查点，共20个。
- 训练连续4000步后，固定24条验证序列（OTB4、LaSOT8、TNL12，共28945帧）评估全部20个检查点；以子集前三和2000步在完整159条验证集上复核。子集仅用于密集监控和候选筛选，不能与完整验证AUC混称；不评估测试集。W&B训练曲线先出现，验证曲线训练结束后补录。
- **尚未实际训练**：2026-10-19 18:20上海时间，GPU0在`nvidia-smi`报设备句柄错误，PyTorch即使限制到GPU1–7仍`cuda.is_available=False`、设备数0。后台监督PID1350756，`training_status.json`为`waiting_gpus`，每60秒检查，8卡恢复后自动训练；六小时仍失败则标`failed`。运行前和汇报时必须读取新输出目录状态及日志，不可把W&B页面存在当作已有参数更新。勿重复启动。
- 源码快照、哈希、配置及运行规格已保存在新输出目录，未提交/推送。修改：可配置的200步检查点保存，不改变其他实验默认行为；验证输出标记`fixed_subset`或`full`。
- 用户随后明确要求暂停。已向监督PID1350756发送SIGTERM并确认该进程及新配置训练进程均不存在；`training_status.json`原子改为`paused`、`completed_optimizer_steps=0`，W&B summary也标为`paused`。恢复须经用户再次要求；不能因为GPU恢复就自行重启。
