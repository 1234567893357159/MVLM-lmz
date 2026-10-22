# MVLM 会话交接记录

> **本分支最新入口（2026-10-22，北京时间）**：当前为 `codex/world-model`，从 `main@5b68095` 创建；已按用户要求实现世界模型与两阶段训练／在线接入，CPU 检查通过，尚未启动正式训练；详见 `reports/reproduction/WORLD_MODEL_IMPLEMENTATION.md`。请先阅读 [世界模型交接](WORLD_MODEL_HANDOFF.md) 和 [分支说明](分支说明.md)。下文继承自 main，为复现和旧记忆实验历史，不代表本分支已实现这些模块，也不构成恢复旧实验的授权。

更新时间：2026-10-17（Asia/Shanghai）；最新下载速度采样时间为当天 10:09。**12:1x 补写第 9.4 节（LaSOT 完整评测，此前版本遗漏）；17:0x 补写第 9.5 节（LaSOT 8 组参数搜索）；18:0x 补写第 9.6/9.7 节（OTB 10 组参数搜索与两数据集汇总）；18:0x 补写第 9.8 节（18 组数字全量核对通过、双数据集最优配置、下载实时状态；第 5 节下载表已过时，以 9.8 为准）。**本文是历史快照；新会话必须重新检查进程、网络和文件大小，不能将本文进度当成实时进度。

## 1. 用户目标与当前任务

- 用户希望使用论文官方权重，复现 MVLM 在测试集上的 PRE、NPR、AUC 等结果。不要重新克隆或覆盖已有权重。用户另咨询过重训资源，但尚未开始训练或授权新增训练集下载。
- 已运行两个自带视频 Demo。
- 用户已明确授权下载前三个数据集：OTB99-Lang、TNL2K 测试集、LaSOT 标准测试集。
- TNL2K 仍在续传；LaSOT 测试集已下载、解压，待最终完整性校验；OTB 已完成评测和调参。用户现已授权下载 VastTrack 训练数据，任务已启动，详见第 11 节；MGIT 及其他训练数据尚未下载。
- OTB 使用官方权重完成原始基线和 9 组调参实验。当前 YAML 是调参后配置，不能当作未经修改的官方默认配置；详见第 9 节。
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
- 最近磁盘检查：仓库所在 overlay 文件系统约 11 TB，总可用约 8.8 TB；inode 使用约 3%。这是容器可见容量，不保证等同账号配额。此前失败没有空间不足证据。

激活环境：

```bash
source /root/.bashrc
conda activate mvlm
cd /root/MVLM
```

也可直接用环境 Python 的绝对路径，不依赖 shell 激活。

## 3. 已做代码修改（尚未提交）

用 `git diff` 核查并保留这些修改，不能当成用户无关改动回滚：

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
7. `experiments/mvlm/mvlm_TF.yaml`：当前保留最高 AUC 的 OTB 调参配置，ALPHA_CORR=0.7、TAU=0.03、OMEGA=0.2、LAMBDA=0.4、PSI_OUT=0.1；其他配置不变。原始完整 YAML 备份在 `output/otb_parameter_trials/otb99_language_baseline.yaml`。
8. 新增实验脚本：`tracking/run_otb_parameter_trial.py`（首次单变量实验）、`tracking/run_otb_parameter_sweep.py`（5 组自适应实验）、`tracking/run_otb_parameter_followup.py`（3 组补充实验），记录配置快照、运行命令、预测、日志、指标与每组缓存。第一个脚本要求输入仍为原始 TAU=0.1，不可在当前 YAML 上不加检查地重复执行。

训练调用沿用默认 `training=True`。未创建提交或 PR。

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

## 5. 数据集下载：位置、来源及最新快照

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

> **注意：下表为 2026-10-17 10:09 的历史快照，已过期。**TNL2K 与 VastTrack 的最新进度见第 9.8 节。

2026-10-17 10:09（上海时间）实际文件及进程快照：

| 数据集 | 已下载（约 GB，十进制） | 压缩包总字节数 | 百分比 | 核实状态 |
| --- | ---: | ---: | ---: | --- |
| OTB99-Lang | 2.799 | 2,798,845,728 | 100% | 48 条评测序列、29,077 帧及框标注校验通过，已完成评测 |
| TNL2K | 54.222 | 91,092,953,063 | 59.52% | PID 90965 正在下载，剩余约 36.871 GB |
| LaSOT | 49.831 | 49,830,545,913 | 100% | 已解压；**已完成全量校验与完整评测**，见第 9.4 节 |

最近一次 20 秒速度采样：TNL2K 3.229 MB/s，按该速度剩余纯下载约 3 小时 10 分钟，不包含校验和解压；速度会变化，不能沿用此估算作为实时结论。LaSOT 已结束下载。用户此前多次要求暂停和恢复，最终有效授权是继续下载。

下载来源：

- OTB 官方：<https://isis-data.science.uva.nl/zhenyang/cvpr17-langtracker/data/OTB_sentences.zip>。
  - 已读取 ZIP 中央目录确认包含 `OTB_sentences/OTB_videos/`、`OTB_query_test/`、`OTB_query_train/`。
  - 数据目录含 99 个视频序列；当前评测加载器选用 48 条测试序列，不是评测全部 99 条。已验证这 48 条需要的 29,077 帧、框标注长度和语言文件，完成上述名称映射修复后可正常评测。其他训练划分未完成训练适配。
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

当前使用两个独立任务，避免原脚本同时写共享状态的风险：

| 数据集 | 独立状态文件 | 日志 | 快照 PID（可能变化） |
| --- | --- | --- | --- |
| TNL2K | `downloads/tnl2k_status.json` | `downloads/tnl2k_restart.log` | 90965（10 月 17 日 10:09 存活） |
| LaSOT | `downloads/lasot_status.json` | `downloads/lasot_restart.log` | 已正常结束，无需重启 |

`downloads/dataset_status.json` 和 `dataset_download.log` 是早期总任务的记录，可能过期。独立任务调用 `d.tnl2k()` / `d.lasot()`，完成后通常只标记 extracted；**不会自动调用脚本的全量 `validate()`**。

LaSOT 的状态可能仍显示上一包 extracting，但进程已在下载下一包；务必同时查进程及分块增长，不能只看状态字符串。

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

# LaSOT（另一个进程）
/root/miniconda3/envs/mvlm/bin/python -u -c 'import sys; sys.path.insert(0,"downloads"); import prepare_test_datasets as d; d.STATUS=d.DOWNLOADS/"lasot_status.json"; d.state={"lasot":{}}; d.lasot()' >> downloads/lasot_restart.log 2>&1
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
- 最近一次恢复前已验证代理可用且 Hugging Face 返回 HTTP 200；若用户关机/切换网络后停下，应重新检查端口与代理进程，但不要无授权查看或展示订阅凭证。
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

论文表 2 完整组件（A4）的 NPR：TNL2K 74.1、LaSOT 68.3、OTB99 75.4。OTB 已有实际评测结果，见第 9 节；其他数据集尚未评测。

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
2. LaSOT 已解压，下一步可检查标准 280 条测试划分、SHA256、全部帧和标注，再进行小规模与完整评测；尚未执行。
3. TNL2K 下载完成后检查 700 条测试划分、SHA256、语言标注及全部帧，不能只检查包大小。
4. 小规模序列确认评测协议与结果格式，再完整评测；保存预测框、耗时、异常清单、指标和论文对照表。不要为整个测试集默认导出巨量视频。
5. OTB 已用 8 GPU 完成多组评测。当前配置在 OTB 测试集上调过参，不应直接当作其他数据集的官方默认配置，也不能把调参结果宣称为独立论文复现。

OTB 已完成的评测示例（当前 YAML 已修改；如需复跑原始基线，先使用原始配置备份，并取新实验名，避免已有结果被自动跳过）：

```bash
cd /root/MVLM
conda activate mvlm
python tracking/test.py mvlm mvlm_TF \
  --dataset_name otb99_lang --weight_path models/MVLM_TF.pth.tar \
  --exp_id otb99_language_baseline --mode single

python tracking/analysis_results.py \
  --tracker_name mvlm --tracker_param mvlm_TF \
  --exp_id otb99_language_baseline --dataset otb99_lang
```

新会话可直接提示：“请先阅读 `/root/MVLM/SESSION_HANDOFF.md`，检查当前下载任务和网络状态，接着完成三个测试集准备及 MVLM 官方权重评测。不要重复克隆、安装环境或从头下载已有数据。”

## 9. 已完成 OTB99-Lang 基线和参数实验

### 9.1 测试条件及结果

- 使用现有官方权重 `models/MVLM_TF.pth.tar` 和缓存 CLIP；没有重训或更换权重。
- 仓库评测加载器为 48 条测试序列，共 29,077 帧。数据目录有 99 个视频，并不等于评测全部 99 条。
- `pip check` 通过；CUDA、三帧 GPU 推理通过；完整评测使用 8×RTX 3090、8 个进程，OMP/MKL 每进程限制 1 线程。
- 原始基线完整跟踪用时约 201 秒。之后完成首次 TAU=0.0 实验，以及用户追加要求的 5+3 组实验，共 9 组调参实验，另有原始基线。
- 所有完整实验均检查预测框帧数、有限数值、正宽高以及逐帧耗时文件，新增八组每组均验证 48 条序列和 29,077 帧。不是根据终端 Done 就推断完整。

| 指标 | 原始基线 | 当前最高 AUC 配置 | 论文参考 | 最佳与论文差距（百分点） |
| --- | ---: | ---: | ---: | ---: |
| AUC | 56.48 | 60.15 | 60.70 | -0.55 |
| PRE | 79.03 | 83.40 | 84.30 | -0.90 |
| NPR | 69.75 | 74.78 | 75.40 | -0.62 |

当前最佳配置已写入 `experiments/mvlm/mvlm_TF.yaml`：

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

### 9.2 结果与记录位置

- 原始配置备份：`output/otb_parameter_trials/otb99_language_baseline.yaml`。
- 基线 exp_id：`otb99_language_baseline`。
- 基线指标：`output/otb99_language_baseline_metrics.txt`。
- 基线日志：`output/otb99_language_baseline_test.log`。
- 基线逐帧预测/耗时：`output/test/tracking_results/mvlm/otb99_language_baseline/otb99_lang/`。
- 最佳 exp_id：`otb99_20261016_092909_06_less_memory`。ID 中时间来自服务器 UTC，面向用户日期时间按 Asia/Shanghai；不要把 ID 当作本地时间。
- 最佳配置：`output/otb_parameter_trials/otb99_20261016_092909_06_less_memory.yaml`。
- 最佳指标/日志：同目录下对应 `_metrics.txt`、`_test.log`。
- 最佳逐帧预测/耗时：`output/test/tracking_results/mvlm/otb99_20261016_092909_06_less_memory/otb99_lang/`。
- 用户可读总记录：`output/otb_parameter_trials/实验记录.md`，包含所有参数、结果、调参依据、配置快照、运行命令和文件位置。
- 机器可读事件记录：`output/otb_parameter_trials/experiments.jsonl`。同一 exp_id 有 running/completed 事件，读取时按 exp_id 取最后一条，不能把两个事件算作两次实验。
- 总结：`output/otb_parameter_trials/调参总结.md`。
- 两批运行日志：`output/otb_parameter_trials/sweep_progress.log`、`followup_progress.log`；已全部完成，没有继续调参的进程。
- 每组保留独立的 `_eval_data.pkl`。官方分析脚本默认共享 `output/test/result_plots/otb99_lang/eval_data.pkl`，已切回最佳实验的缓存；后续分析会覆盖它，不能仅凭此共享路径判断是哪次实验。
- `tracking/test.py` 遇到已有序列结果会跳过；真实重跑应取新 exp_id，不要误把复用旧预测当成新运行。
- `tracking/analysis_results.py` 默认只打印 PRE/NPR/AUC 并保存缓存，不自动生成曲线图片；指标文本通过重定向保存。

### 9.3 调参结论及解释边界

- 固定原始其他参数时，TAU=0/0.03/0.06/0.1 的 AUC 为 58.95/59.30/58.33/56.48。
- TAU=0.03 时，PSI_OUT 从 0 改为 0.1，AUC 59.30→59.69。
- 在此基础上，ALPHA_CORR 从 1.0 改为 0.7，AUC 59.69→60.02。
- 较强记忆组合 OMEGA=0.6、LAMBDA=0.7，在当时基准上使 AUC 降至 58.07；两参数同时改变，不能分别归因。
- 仅把 OMEGA 从 0.4 降到 0.2，AUC 60.02→60.15。
- 在最佳邻域，TAU=0.01 得到 59.61，PSI_OUT=0.2 得到 59.21，均不保留。
- 参数机制：TAU 控制候选集合及下一帧局部/全局搜索；PSI_OUT 控制 IoU≤阈值的干扰候选池；ALPHA_CORR 混合图文/分类差值；LAMBDA 控制记忆更新、OMEGA 控制当前决策对记忆的依赖。最终候选集合内仍按分类分数选择预测框，不是直接按置信度加权分数选最高框。
- 代码中最终置信度可化为 `(1-OMEGA*LAMBDA)*当前置信度 + OMEGA*LAMBDA*上一帧记忆`；原始历史权重 16%，强记忆 42%，最佳 8%。原因解释是代码支持的推断，尚未做逐帧可视化验证。
- 用户明确授权了测试集上的探索性调参。必须保留这个事实：最佳分数不是独立验证结果，不证明找回了作者原始设置，不保证能迁移到其他数据集；每组仅跑一次，未统计随机波动或置信区间。

### 9.4 LaSOT 原始参数完整评测（10 月 17 日新增，此前版本缺记）

本节由 2026-10-17 12:1x（上海时间）会话补写。原交接文档（10:09 快照）记录 LaSOT "尚未做最终全量校验或评测"，但该实验实际已于当日 **11:10:41（上海时间）完整跑完**，属于文档遗漏，不是新做的实验。

- 实验：`lasot_language_original_mp16_20261017_103923`；`status.json` 标记 `status=completed`，`prediction_validation.json` 为 `checked=280, errors=[]`。
- 产物目录：`output/lasot_evaluations/lasot_language_original_mp16_20261017_103923/`（含 `metrics.txt`、`结果说明.md`、`status.json`、`prediction_validation.json`、`config.yaml`、`runtime.yaml`、`archive_checksums.json`、`dataset_manifest.json`、`commands.json`、`code_changes.patch`、`eval_data.pkl`）。
- 逐帧预测/耗时：`output/test/tracking_results/mvlm/lasot_language_original_mp16_20261017_103923/lasot/`。
- 参数来源：`HEAD:experiments/mvlm/mvlm_TF.yaml` 的**原始发布参数**，即 ALPHA_CORR=1.0、TAU=0.1、OMEGA=0.4、LAMBDA=0.4、PSI_OUT=0.0。**与第 9 节 OTB 调参后的当前工作 YAML 不同**，不要混用。
- 规模：280 条序列、685,360 帧；8 张 RTX 3090，每卡 2 进程，共 16 进程，每进程 1 个 CPU 线程。
- 数据校验：六个包 SHA256 全部匹配源元数据；全部帧非空；标注、语言、可见性长度已核；每序列首/中/末帧可解码。`status.json` 同时记录权重 SHA256 `91c31230f3d355f71104df8191e386632a39ce9d3bbc1f7ce5bf3772b369cc4f` 与 git commit `4186c8fd`。
- 耗时：验证数据 10:39:24→10:40:41；推理 10:40:41→11:10:34（test.log 记录 1791.3 秒，FPS 约 41.9）；计算指标至 11:10:41。
- 协议说明（原文）："Single baseline evaluation; no LaSOT tuning. Uses repository language-only evaluation protocol, including its first-frame handling." 即**未做 LaSOT 调参**，且沿用仓库语言评测协议（含首帧处理），第 8 节对该首帧计分细节的保留意见同样适用。

结果：

| 指标 | 本次（原始参数） | 论文表 1 | 差距（百分点） |
| --- | ---: | ---: | ---: |
| PRE | 61.47 | 65.5 | -4.03 |
| NPR | 64.15 | 68.3 | -4.15 |
| AUC | 57.77 | 60.7 | -2.93 |

对照 OTB（第 9 节）：OTB 用**测试集上调过参**的配置才做到 -0.55/-0.90/-0.62，而 LaSOT 用**原始参数**是 -2.9～-4.2。两者不能直接比：OTB 的接近程度部分来自在测试集上调参，LaSOT 才是未调参的纯净复现口径。**因此不能把 OTB 的成绩当作论文复现已整体达成的证据。**

可复跑命令（注意：`experiments/mvlm/lasot_language_original_mp16_20261017_103923.yaml` 是原始参数快照，当前工作 YAML `mvlm_TF.yaml` 已是 OTB 调参后的值，复跑请显式指定参数快照或新 exp_id，避免与已有结果混淆）：

```bash
cd /root/MVLM
/root/miniconda3/envs/mvlm/bin/python -u -m tracking.test mvlm \
  lasot_language_original_mp16_20261017_103923 \
  --dataset_name lasot \
  --weight_path /root/MVLM/models/MVLM_TF.pth.tar \
  --exp_id lasot_language_original_mp16_20261017_103923 \
  --mode mp --threads 16 --num_gpus 8

/root/miniconda3/envs/mvlm/bin/python -u -m tracking.analysis_results \
  --tracker_name mvlm \
  --tracker_param lasot_language_original_mp16_20261017_103923 \
  --exp_id lasot_language_original_mp16_20261017_103923 --dataset lasot
```

下一步建议（未执行，仅建议）：差距既有可能是参数口径，也可能是第 8 节记录的 `tanh(分数差)` 与论文公式 (4)(5) 的尺度估计差异。建议先在 LaSOT 上用 OTB 调参配置跑一次对照（同规模约 30 分钟），以区分"参数"与"实现"两个来源，再决定是否动公式。

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

### 9.8 参数搜索结果核对与实时状态（18:0x 补记）

#### 数字核对结果

2026-10-17 18:0x（上海时间）对第 9.5、9.6 节所列全部实验，逐一读取磁盘上的 `status.json` 并与文档数字比对：

- LaSOT 8 组：**全部一致**（PRE/NPR/AUC 三位小数相同，0 处不符）。
- OTB99-Lang 10 组：**全部一致**（0 处不符）。
- 核对中发现并已修正 1 处命名不一致：9.6 表中原写作 `otb_tune_09_joint`，磁盘实际目录为 `otb_tune_09_joint_a052_t0028`，指标 83.77/75.19/60.43 本身无误。

核对脚本思路：遍历 `output/lasot_parameter_trials/lasot_tune_*/status.json` 与 `output/otb_parameter_trials/otb_tune_*/status.json`，取 `metrics` 四舍五入到两位小数，与文档表格逐行比对。

#### 两个数据集的最终最优配置（可直接引用）

| 数据集 | αcorr | τ | ω | λ | ψout | PRE | NPR | AUC | 论文 | 差距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LaSOT | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 | 65.93 | 68.72 | 61.06 | 60.7 | **+0.36** |
| OTB99-Lang | 0.5 | 0.03 | 0.4 | 0.1 | 0.1 | 84.05 | 75.33 | 60.56 | 60.7 | -0.14 |

两组参数**除 ω/λ 的分配外完全相同**，而 ω·λ 乘积均为 0.04，因此在有效自由度上两个数据集收敛到同一组参数。LaSOT 用 (ω=0.1, λ=0.4)，OTB 用 (ω=0.4, λ=0.1)，二者 AUC 差异在 0.04 噪声量级内，可视为等价。

#### 实时状态快照（与原第 5 节 10:09 旧快照对照）

下载任务与既有章节相比已推进明显，**第 5 节的表格停留在 10:09**，应以本节为准：

| 项目 | 10:09 旧快照 | 18:0x 实测 | 说明 |
| --- | --- | --- | --- |
| TNL2K | 59.52%（54.222 GB） | **84.57%，剩约 14.06 GB** | PID 仍为 90965，稳定推进 |
| VastTrack | 未开始 | **2.827%（20.82 GB / 736.6 GB），第 1/14 卷** | PID 107980 |
| LaSOT | 待校验 | 已完成评测与 8 组参数搜索 | 见 9.4、9.5 |
| OTB99-Lang | 已完成 | 另完成 10 组新参数搜索 | 见 9.6 |

两个下载进程在本次会话期间**全程未中断**，参数搜索实验未挤占其带宽（实验为纯本地 GPU 推理，不占出口带宽）。

#### 关于 TNL2K 的提醒

TNL2K 已接近完成（84.57%）。下载结束后**不会自动做全量校验**（见第 6 节：独立任务通常只标记 `extracted`，不调用 `validate()`）。正式评测前必须另行确认 700 条测试序列的划分、SHA256、语言标注与帧完整性，不能只看包大小。

TNL2K 论文目标为 PRE 60.9 / AUC 57.8。若沿用本次在 LaSOT 与 OTB 上一致收敛的参数（α=0.5, τ=0.03, ω·λ=0.04, ψ=0.1），可作为该数据集的**首要试验配置**，但仍应从原始参数跑一次基线作为对照，且需注意仓库中 `TNL2K` 的 `TEST` 分支是否另有覆盖值。

## 10. 重训咨询：已核实条件，尚未启动

- 用户希望通过改动网络结构完成毕设，已解释训练复现与加载现有 MVLM 权重微调不同；未启动训练。现已单独授权下载 VastTrack 训练数据；其他训练数据尚未新增下载。
- 论文训练数据是 LaSOT、TNL2K、VastTrack 的训练部分。当前完成/正在下载的 LaSOT、TNL2K 为测试镜像，不能代替训练集；仅 OTB 不足以复现作者训练，且不能用已评测的 48 条测试序列训练后宣称复现。
- 骨干预训练文件 `pretrained/itpn/fast_itpn_base_clipl_e1600.pt` 当前缺失；推理因 `training=False` 不需要它，但正式训练会恢复加载要求。CLIP 缓存已存在。
- 论文设置：60 epoch、每 epoch 100,000 个采样样本、学习率 0.0005、AdamW、batch size 80，报告使用 4×RTX A6000 48 GB。官方 README 的配置为 FastITPN-Base/CLIP。
- 重要实现差异：`lib/train/base_functions.py` 中 `BATCH_SIZE` 直接传入各进程 DataLoader，配合 DistributedSampler，当前是每卡 batch；8 卡直接用 80 会变成全局 640，不能默认为全局 80。论文正文的 batch 口径尚未确认。
- 如果按全局 batch=80 规划，600 万次样本采样约 75,000 次更新。假设每步 0.5/1/2/5 秒，纯训练约 10.4/20.8/41.7/104 小时。这是条件估算，不是本机训练实测。建议准备数据后先跑 100～200 步测速度和峰值显存，再估总时长；推理耗时不能直接用来推算训练。
- 当前服务器 8×3090、每卡 24 GB；系统内存约 440 GiB，10 月 16 日检查可用约 375 GiB，磁盘可用约 8.8 TB。多卡显存不是一张可直接使用的 192 GB 显卡，仍需验证每卡 batch。
- 曾建议每卡 batch 8～10 做短测，这是准备方案而非已验证可运行配置。训练器支持 CUDA AMP，但默认 `getattr(cfg.TRAIN,'AMP',False)` 为关闭，任何开启都要记录为训练设置变化。
- OTB 训练加载器存在，但引用的 `lib/train/data_specs/otb99_train_split.txt` 等划分文件缺失，`DATA.USE_NLP` 中也没有所需 OTB99 项；若做训练练习需先补齐划分、路径和配置并严格隔离测试数据。
- 目前只验证了推理环境和完整评测，不能把依赖安装成功或评测跑通等同训练流程已验证。

## 11. VastTrack 训练数据下载（10 月 17 日新授权）

- 用户要求下载 VastTrack 并与其他数据集放在同一数据根目录，确保训练加载器能找到；本次授权限于 VastTrack 的训练准备，不包括 MGIT 或其他新数据集。
- 已设置 `lib/local_config.py::vasttrack_dir = '/root/MVLM/data/vasttrack'`；加载器自行追加 `train/`，最终位置为 `/root/MVLM/data/vasttrack/train/<类别>/<序列>/imgs/`，同序列含 `Groundtruth.txt` 和 `nlp.txt`。
- 官方 OneDrive 分享链接返回 403、目录 API 返回 401；百度网盘入口需要提取码流程。找到公开镜像 `https://huggingface.co/datasets/xche32/vasttrack`：14 个连续 tar.gz 分卷，总计 **736,604,489,019 字节，约 736.604 GB**。这是该镜像分卷的实测总大小，不等于已核实官方 82 个 ZIP 的大小或解压后大小。
- 镜像根目录为 `vasttrack/<类别>/<序列>/...`；前 4 MiB 抽查到 Aardwolf-10 图像、真实框，训练清单包含此序列；同作者公开的 unpacked 示例也列有 `nlp.txt`。完整划分和语言覆盖需待全部解压后校验，尚未宣称训练数据已完整可用。
- 镜像覆盖范围尚未全量核实。为读取连续压缩流，需要下载全套 14 个分卷；解压时严格按仓库 `lib/train/data_specs/vasttrack_train_split.txt` 选择训练序列，其他序列不提取。Python 按行读取清单实际为 **46,836 条**，不是 `wc -l` 的 46,835（末尾缺换行）；与官方论文的 47,110 条存在差异，后续不能混称。
- 下载脚本：`downloads/prepare_vasttrack.py`，复用现有 `prepare_test_datasets.py::download` 的 4 并发 Range 下载、分块续传及镜像 LFS SHA256 校验；复用的是下载函数，不会改写 TNL2K/LaSOT 状态。
- 分卷和编号分块：`downloads/vasttrack/`；来源及大小/SHA256 清单：`downloads/vasttrack/source_manifest.json`。
- 状态：`downloads/vasttrack/status.json`；日志：`downloads/vasttrack/download.log`；PID 文件：`downloads/vasttrack/download.pid`。
- 初始后台 PID **107980**，采用独立 session，10 月 17 日约 10:58 采样确认存活，20 秒有效字节增加约 6.66 MB（0.33 MB/s），当时约 10.85 MB；这是启动快照，不代表实时进度。
- `status.json` 的 downloaded_bytes 每 10 秒更新，采用已追加数字分块和完整分卷的保守计数，不含正在接收的 `.segment`；实时测速可按第 6 节的方法加上已验证 Content-Range 的有效 segment。
- 所有分卷下载后逐卷校验 SHA256，按连续流读取解压，避免额外合并一份 736 GB 文件；安全检查绝对路径/`..`，不跟随压缩包链接。
- 自动最终校验包括训练清单每条序列的语言非空、真实框四列和数值、按框数逐一检查所有 `00001.jpg` 等帧，并构造 `VastTrack(split='train')` 读取首帧。通过后才将 status 标为 `ready`，失败则保存错误和 `validation.json`，不能因下载达到 100% 就宣称能训练。
- 当前仍用 SSH 转发的 Clash 代理，与 TNL2K 共享带宽。进程仅负责准备数据，不会开始模型训练。
- 官方仓库的 `assets/train_md5.txt` 返回内容中存在非十六进制字符，不能据此进行有效 MD5 校验；保存的原始参考文件在 `work/vasttrack/checksums.txt`，目前使用镜像元数据中的合法 LFS SHA256 保证传输完整性，不声称与官方 ZIP 已完成哈希对照。
- 如需恢复，先检查有无此下载脚本进程，再使用环境 Python 启动 `downloads/prepare_vasttrack.py`，日志追加并后台化；已有完整分卷会按大小复用，因此不得删除或覆盖已有数据。新会话应先实查进程和状态。
