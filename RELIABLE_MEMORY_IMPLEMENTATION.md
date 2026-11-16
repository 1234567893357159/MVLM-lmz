# 轻量可靠记忆库：实施方案

**2026-11-16 更新：本文件保留最初的受限残差设计作为历史与对照。当前主方案 M6 已改为直接通道拼接，M7 为可靠写入＋干扰保留＋受限残差对照；当前配置、代码和训练预算以[拼接训练方案](RELIABLE_MEMORY_TRAINING_PLAN.md)为准。**

日期：2026-11-16（北京时间）

分支：`codex/reliable-memory`；代码基点：`a115a99`

状态：首版已实现并完成工程短测。本文保留完整实施目标；实际入口、已验证范围和待补内容见[首版跑通记录](reports/reproduction/RELIABLE_MEMORY_SMOKE_20261116.md)。类型草案等设计说明以当前源码为准。

本方案细化[研究设计](RELIABLE_MEMORY_PLAN.md)，以“可靠写入＋干扰感知保留＋受限融合”为第一版目标。保持现有 FastITPN、CLIP、CENTER 与双路记忆读取结构，以便验证改进来自哪里。主线为 TF 语言跟踪，首帧自主预测，不引入首帧真实框视觉模板。

## 1. 已核查的代码和改造原因

| 当前入口 | 实际行为 | 实施决定 |
|---|---|---|
| `lib/models/mvlm/memory.py` | `MemoryEntry` 存 token、帧号、质量和框；短期 FIFO，长期按质量与冗余筛选 | 保留旧类；新增可选择的管理器与有界状态 |
| `lib/models/mvlm/memory_fusion.py` | 读取后经过门控、输出投影，以残差加入搜索特征 | 在输出投影之后、残差相加之前限幅；保持参数名和维度 |
| `lib/models/mvlm/mvlm.py:forward_memory_sequence` | 在预先裁好的图像上运行；混合 GT/预测写入；用原始热图最大值判断质量 | 新闭环训练不走这一接口；保留它复现历史实验 |
| `lib/test/tracker/mvlm.py:track` | Hann 后解码候选、MVLM margin 筛选、选择最终框；用最终分数和 margin 写入 | 抽出共享候选决策和状态更新，不重复维护另一套训练公式 |
| `lib/train/data/sampler.py:get_frame_ids_memory` | 尽量选全部可见帧，回退时也只选可见帧 | 新采样器保留片段内不可见帧，覆盖遮挡和恢复 |
| `lib/train/data/processing.py` | 围绕扰动后的 GT 裁剪 | 新数据管线返回完整 RGB 帧，在逐帧核心中按预测裁剪 |
| `tracking/evaluate_memory_custom.py` | 第一帧直接保存 GT | 新评测入口显式处理第 0 帧，保存真实预测 |
| `lib/train/actors/mvlm.py:tracking_loss` | GT 裁剪框直接 clamp；GIoU 异常被兜底为零 | 新闭环损失显式区分有效、不可见和出裁剪区域；异常应报错并保留诊断 |

源分支没有 `tests/` 目录，现有工程检查集中在 `tracking/check_memory_module.py` 等脚本。新增检查沿用轻量脚本形式；此文不要求先引入新的测试框架。

## 2. 第一版的明确选择

1. 三项改动分别开关；新增策略全部关闭时走旧管理/融合规则，但仍可使用统一的新闭环训练和严格首帧评测。
2. 确认缓冲仅保存描述子和少量元信息。连续 K 帧确认后，第一版只提交当前帧条目，不回填此前缓冲帧。这样不破坏现有按 frame_id 严格递增的写入约束。
3. 干扰感知仅改变合格条目的长期保留，不修改当前框选择，也不额外过滤写入；确保单因素实验 M4 可解释。
4. 限幅上限作用于最终残差；第一版不新增可训练融合门控，继续使用原有门控。
5. 推理状态与可训练网络参数分开；候选选择、框映射、记忆写入均停止梯度，定位头和读取器保留当前帧梯度。
6. 先单设备通过，再做真实双卡同步检查；资源数量由后续短测确定。

“零记忆复现基线”仅指相同网络权重、相同输入和相同先前跟踪状态下，新增记忆不产生特征修正。已经微调 CENTER 的权重不会因关闭记忆自动恢复成官方权重。递归轨迹一致性应从同一初始状态开始比较。

## 3. 文件拆分和职责

| 文件 | 拟实现内容 |
|---|---|
| 新 `lib/models/mvlm/tracking_core.py` | `TrackingState`、裁剪变换、候选解码、原 MVLM 筛选、下一搜索状态；训练测试共用 |
| 新 `lib/models/mvlm/memory_policy.py` | 可靠性信号、确认状态机、竞争候选选择、长期保留评分 |
| 新 `lib/models/mvlm/reliable_memory.py` | 新运行时管理器，委托/复用旧 ROI 与 pack 格式，不继承可训练参数 |
| 改 `lib/models/mvlm/memory_fusion.py` | 限幅函数、可选读取诊断，默认兼容旧调用与权重 |
| 改 `lib/models/mvlm/mvlm.py` | 管理器工厂、可靠模式单帧 forward 返回文本/特征/热图；可训练计算保持在正常 forward 内 |
| 改 `lib/test/tracker/mvlm.py` | 新模式委托共享核心，新增无框初始化入口；旧模式照常分派 |
| 新 `lib/train/data/reliable_memory_dataset.py` | 按视频清单采样完整 RGB 片段、标注和时间戳，collate 保留变长帧列表 |
| 新 `lib/train/actors/reliable_memory.py` | 逐帧闭环 forward、GT 到损失的映射、预热和有效帧归一化 |
| 新 `tracking/train_reliable_memory.py` | 实验配置、冻结分组、优化器、检查点与恢复；不自动启动后续实验 |
| 新 `tracking/evaluate_reliable_memory.py` | 严格首帧评测、只读预测导出、固定数据清单与统计 |
| 新 `tracking/check_reliable_memory.py` | 策略、因果顺序、限幅、坐标、训练/测试一致性检查 |
| 改 `lib/config/mvlm/config.py`，新增独立 YAML | 注册新键、实验组设置与配置校验；新能力默认关闭 |

数据读取、训练器、检查点兼容的改动分别提交，避免把旧实验入口一起替换。所有新增模块只依赖 PyTorch、NumPy 和现有图像/ROI 工具，不添加向量数据库服务。

## 4. 数据结构与接口契约

### 4.1 坐标和时间

- `frame_id`：视频中的原始零基帧号，不是 batch 序号；训练 stride 大于 1 时按真实帧号计算年龄和位移时间。
- `box_crop_xywh`：网络裁剪内归一化坐标；由候选 `cxcywh` 显式转换，ROI 提取前校验并与裁剪有效域求交。
- `box_image_xywh`：原始图像像素坐标，裁剪和恢复都使用它。
- `box_normalized_xywh`：原图宽高归一化坐标，仅用于保存和跨尺寸描述，不与像素框隐式混用。
- `CropTransform`：明确原图到网络输入的 x/y 尺度、裁剪原点、填充和输入尺寸；提供可逆的点/框变换。

全图缩放和局部裁剪必须分别记录变换，非正方形图像使用独立 x/y 尺度。不得只沿用某个面积缩放比反映射全图框。迁移时给旧映射和统一映射分别标版本，不能静默改变历史指标。

### 4.2 类型草案

```python
@dataclass
class Observation:                 # 不含 GT
    frame_id: int
    box_image_xywh: Tensor         # [4]
    box_normalized_xywh: Tensor    # [4]
    roi_tokens: Tensor             # [4, 512]，detach 后保存
    descriptor: Tensor             # [512]，归一化原始 ROI 均值
    score: float                  # Hann 后最终候选分数
    cls_margin: Optional[float]
    language_margin: Optional[float]
    vlm_pass: bool
    roi_valid: bool
    used_global_search: bool

@dataclass
class WriteDecision:
    action: str                   # reject / pending / commit
    reasons: tuple
    streak: int
    transition: str                # bootstrap / normal / recovery

# 以下接口运行于 no_grad；状态明确传入，不保存在全局变量中。
signals = compute_reliability(observation, previous_observation, policy_cfg)
decision = policy.observe(signals, observation)   # 更新确认缓冲
competitors = collect_competitors(candidates, selected_index, raw_features)
memory.update(observation, decision, competitors)
packed = memory.pack(features, frame_ids, tokens_per_entry=4)
```

`pack` 继续向现有读取器提供 token、有效 mask、quality、age、long_term；候选缓冲永不出现在 pack 中。新管理器另外保存 `committed_at`、辨识差距滑动均值、有效观察次数、最近读取信息。第一版只提交当前帧，因此 `committed_at == frame_id`；读取必须满足 `frame_id < 当前帧号`。

所有状态支持 `reset()` 和带版本号的 `state_dict()/load_state_dict()`；普通模型 checkpoint 不默认携带视频内容。离线继续某视频时才显式保存对应状态，并校验视频 ID、语言、权重哈希和下一帧号。

## 5. 共享的逐帧执行流程

```python
state = reset(video_id, language)        # 空库，global_search=True
text_emb = encode_language_once(language)
for frame_id, rgb_image in ordered_frames:
    crop, transform = prepare_crop(rgb_image, state)  # 首帧全图，无 GT
    output = model(crop, text_emb, state.memory)       # 正常 nn.Module.forward
    selected = decode_and_select(output, state, transform, cfg)
    observation = make_observation(output.raw_features, selected)
    decision = write_policy.observe(observation, state)  # detach/no_grad
    state.memory.update(observation, decision, selected.competitors)
    state.update_search(selected)                      # 原 MVLM 局部/全局规则
    yield output, selected.box_image_xywh, diagnostics
```

- 单帧内部先 pack 读取、后写入。返回的当前帧预测不因写入重新计算。
- 语言改变时清空内容库、确认缓冲、保留统计、原 MVLM 的 kmem 和局部搜索状态；重新全图定位。
- 写入拒绝本身不强制修改原 MVLM 的局部/全局切换；策略只影响内容记忆，避免引入额外混杂。
- 无效定位返回显式失败标志，停止写入并将下一帧设为全图；评测完整保留帧数，不用 GT 替换失败输出。
- 训练和推理仅在模型 grad 模式、数据增强和损失计算方面不同；同一固定输出输入到决策函数，应得到相同框和写入动作。
- GT 不进入 `TrackingState`、`Observation` 或 `step` 参数。评测先生成预测，再通过独立 scorer 关联 GT。

第一阶段先从旧 tracker 抽出候选选择为纯函数，固定预测输出验证 argmax、margin、Hann、并列候选顺序和 kmem 更新一致；再迁移裁剪映射。新评测可向旧 TF 编码器传全图占位模板以满足签名，但必须断言 `TEMPLATE_FREE=True`，不得从 GT 构造占位。

## 6. 可靠写入状态机

### 6.1 初始参数和明确语义

下列值仅为可运行起点；上线训练前须经开发视频诊断，冻结后所有配对组共用。每个阈值单独记录，不能解释为校准后的身份概率。

| 参数 | 初始值 | 含义 |
|---|---:|---|
| `MIN_SCORE` | 0.9 | 最终候选定位分数下限；旧策略对照仍为 0.5 |
| `MIN_CLS_MARGIN` / `MIN_LANGUAGE_MARGIN` | 0 / 0 | 有有效竞争候选时，原候选差距不能为负 |
| `CONFIRM_FRAMES` | 2 | 正常连续可靠观测数量 |
| `RECOVERY_CONFIRM_FRAMES` | 3 | 全图恢复后的连续确认数量 |
| `MIN_DESCRIPTOR_SIM` | 0.7 | 相邻待确认观测外观相似度起点 |
| `MAX_CENTER_SPEED` | 2.0 | 像素位移除以前一框面积平方根及帧间隔 |
| `MAX_LOG_SCALE_RATE` | log(4) | 最大单轴尺度比的对数，按帧间隔归一化 |

硬拒绝：ROI 非法、数值非有限、分数低于下限、未通过既有 MVLM 候选筛选，或有有效竞争者且候选差距不达标。无竞争候选时 margin 为缺失，只略过该项，不把缺失编码成大正值。

运动大幅变化和外观不一致触发重新确认，不永久拒绝目标。进入全图恢复后使用更长 streak，不沿用跨遮挡的位移约束；以当前连续观察的一致性建立新轨迹。旧库的相似度作为诊断，不作为恢复的必备条件，避免目标外观改变后永远无法恢复。

### 6.2 转移规则

- `EMPTY/PENDING`：可靠观测形成 streak；与上一 pending 不一致时以当前观测重新开始；未达 K 不提交。
- `TRACKING`：确认后提交当前帧。连续一致的后续可靠帧可立即提交，不必每两帧才写一次；一次硬拒绝将 streak 清零。
- `RECOVERING`：首次全图定位或运动异常后重新建立连续一致链，达到恢复 K 后提交当前帧并退出恢复状态。bootstrap 首次全图也使用恢复 K。 处于恢复状态时，连续全图搜索不应在每帧重复清空 streak；只有新的不一致/硬拒绝才重置，防止无法积累确认。
- 拒绝/重置仅清空 pending，不清空已确认历史。目标切换或新视频才清空全部内容库。

第一版 pending 至多保存 K-1 个描述子/标量和上一预测框，不保存完整特征图。与原设计中“可延迟提交历史帧”的设想相比，这里选定只提交确认时当前帧；若未来引入延迟提交，必须另做时间顺序和重复 ID 检查。

## 7. 干扰感知长期保留

候选仍来自同一次 CENTER forward，不增加检测器或第二次全图编码。先排除与所选框 IoU>0.1 的候选，将其余候选按定位分数和语言分数的独立排名交错取样，以稳定顺序去重；候选间 IoU>0.5 去重，最多保留 3 个。对有效竞争候选提取原始 ROI 描述子，称为“疑似干扰”，不当作真实负标注。

对已确认的观测以及当前待插入的长期候选 e：

```text
d_e = cos(e, 当前目标描述子) - max_j cos(e, 竞争候选描述子_j)
disc_ema_e ← 0.9 * disc_ema_e + 0.1 * d_e
```

第一次有效观察直接设为 d_e；没有竞争候选不更新计数或均值。至少 3 次有效观察后才使用辨识统计做保留决策。新候选统计不足时，视作未知而非低质量。

第一版保留规则：

1. 所有接收条目先进入容量 8 的短期 FIFO。
2. 长期库内描述子余弦相似度>0.9 时视作重复；优先更高质量，相差≤0.02 且双方辨识统计成熟时优先更高 `disc_ema`，仍相同时保留较新者。
3. 非重复候选加入长期集合；超过 16 条时，先计算旧版 `utility = quality - max_similarity_to_other_entries`。
4. 找出 utility 距最低值≤0.02 的集合。若至少两项辨识统计成熟，只在成熟项中淘汰最低 disc_ema；否则淘汰最低 utility，最终并列按最老 frame_id。
5. 条目替换和 short/long 去重继续按 frame_id；pack 返回的总独立条目最多 24、token 最多 96。

这里辨识统计只打破原质量/冗余效用相近的淘汰决策，避免第一版为高度不确定的伪干扰付出过强代价。保护首个可靠条目暂不加入第一版，防止无框初始化错误被长期锁死。

M4 单因素关闭可靠写入时，仅对原写入规则通过的观测计算统计，保留规则本身不加第二道过滤。M6 同时启用可靠写入和此保留机制。读取频率与年龄先用于日志，不加进本版效用函数。

## 8. 受限融合的具体实现

在现有 `delta = output_proj(gate * retrieved)` 后进行：

```python
raw_delta = delta.float()
base = features.float()
base_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
delta_norm = torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True)
scale = torch.minimum(torch.ones_like(delta_norm),
                      rho * base_norm / delta_norm.clamp_min(1e-6))
bounded = raw_delta * scale
bounded = bounded * has_memory.to(bounded.dtype)
result = features + bounded.to(features.dtype)
```

- rho≥0；rho=0 显式返回 features，避免 FP32→低精度往返改变结果。
- 保留 output_proj 的零初始化；零输出在 rho>0、base_norm>0 时 scale 为 1，确认第一步投影梯度非零；两步后再检查读取层梯度。
- 一个 batch 可同时有空库和非空库，不能把整个 batch 统一当作有记忆。
- 任一非有限特征报错并输出诊断，不靠乘零掩盖 NaN；全部 mask 为空时避免 softmax 全负无穷。
- 原门控参数和旧 state_dict 名称不变；限幅不增加可训练参数。
- 关闭限幅时执行原计算路径，记录诊断也不能修改数值。读取权重统计可通过 `return_diagnostics=False` 默认关闭。

初始 rho=0.25；开发阶段最多比较预先登记的 0.1/0.25 两个非零值及 0 旁路，禁止在标准测试集扩展网格。范数限制与旧“残差乘 0.25”分别命名和记录。

## 9. 新闭环训练管线

### 9.1 片段与裁剪

新采样器只从审计后的训练视频清单采样，输出 12 帧有序 RGB 原图、原始 frame_id、语言，以及单独的 GT 标注。选片段起点可要求有效目标，片段内部保留不可见和无有效标注帧；不因它们的存在重采样掉整段。

优先从 LaSOT/VastTrack/TNL2K 的官方训练部分按 1:1:1 采样，实际清单数量以审计后 manifest 为准。旧 159 条实验的训练列表涉及标准测试来源，不直接用于新主结果训练。清单按视频与可核查同源组隔离，保存哈希并记录官方初始化的预训练来源限制。

前 4 帧预热仅 forward 和更新预测状态，不算定位梯度；后 8 帧按有效监督计算损失。首个片段帧也必须全图定位。每个片段重置状态；同一片段采用一致亮度等增强，不做改变语言左右语义的随机翻转。预测裁剪与标注映射分别计算，不让 GT 决定本帧搜索区域。

### 9.2 有效损失与失败样本

复用权重 `2*GIoU + 5*L1 + Focal`，首轮关闭 CM。

- 标注有效且目标完整处于搜索区域时计算回归和目标热图。
- 有有效标注且目标完全位于裁剪外，回归不计算，Focal 使用全背景目标；保持状态继续真实预测，不能用 GT 将裁剪拉回。
- 部分截断目标先只计算有效位置的热图，暂停该帧框回归；单独统计数量，后续若拟训练截断回归需另定义目标。
- 明确全遮挡/出画且协议确认无可见目标时可用背景热图；仅标注缺失或语义不明确时屏蔽监督，不把未知标签当作背景。
- 所有标注均不用于禁止/允许记忆写入；可见性只决定损失，不能成为推理不可获得的写入门控。
- 后 8 帧监督量按定义归一化，记录监督帧数、出裁剪比例、空监督批次。某组更早失跟可能导致有效回归帧变少，应报告该差别。

### 9.3 梯度、分布式和恢复

正式起点为同一官方 MVLM，训练记忆读取器和 CENTER；视觉编码器、CLIP 本体及语言投影冻结，CENTER BN 运行统计冻结、仿射可训练。共同模块 AdamW 学习率暂定 1e-4、weight decay 1e-4、梯度范数上限 1.0；4000 更新预算，前 200 更新线性预热至峰值，此后 cosine 到零。全局 batch 原建议 16 已撤回，修订为显式 `--global-batch 80` 个片段，四卡每卡累计 20 个片段、每次最多 640 个监督帧。这些是尚待正式规模验证的配对微调配置，不能称为论文 batch 80 的严格等价复现；论文及官方代码的具体对照见[训练方案](RELIABLE_MEMORY_TRAINING_PLAN.md#0-论文与官方实现核对)。

新增配置单独声明冻结范围，避免复用旧 `memory_enabled` 分支时无记忆 M1 没有加载官方权重或冻结范围不同。加载审计必须列出语言投影、骨干、CLIP、头和读取器各参数组。

运行时历史全部 detach，不做跨帧 BPTT。当前帧读取器参数、限幅和 CENTER 的梯度正常传播；可先对 8 个监督帧累积计算图，测显存后再决定是否逐帧 backward。两种路径必须匹配损失归一化，不同时重复求梯度。

DDP 下所有可训练计算通过 DDP 包裹对象的正常 forward 调用；不直接经 `model.module` 绕开同步。空库会导致读取层未参与计算，先保留未使用参数检测并验证真实两卡步骤。GT 有效数量不同的 rank 使用一致的全局归一化和 collective 顺序；全体无监督时共同跳过 optimizer/scheduler 更新，记录原因，不能个别 rank 单独跳过。

保存模型、优化器、scheduler、真实更新计数、每个 rank 的随机状态、清单/配置哈希。第一版只在完整片段 batch 边界保存，不承诺中途恢复某个训练片段状态或 DataLoader 预取队列的位级重放。短测还应确认冻结参数哈希不变、BN 统计不变、共同可训练参数实际更新。

## 10. 配置和实验组

新键必须先在 `lib/config/mvlm/config.py` 注册，默认关闭；所有路径从显式参数解析，不把其他 worktree 的代码路径当作依赖。

```yaml
MODEL:
  ENCODER:
    TEMPLATE_FREE: true
  MEMORY:
    ENABLED: true
    SHORT_CAPACITY: 8
    LONG_CAPACITY: 16
    RELIABLE_WRITE: false
    DISTRACTOR_RETENTION: false
    BOUNDED_FUSION: false
    POLICY:
      MIN_SCORE: 0.9
      MIN_CLS_MARGIN: 0.0
      MIN_LANGUAGE_MARGIN: 0.0
      CONFIRM_FRAMES: 2
      RECOVERY_CONFIRM_FRAMES: 3
      MIN_DESCRIPTOR_SIM: 0.7
      MAX_CENTER_SPEED: 2.0
      MAX_LOG_SCALE_RATE: 1.38629436112
    RETENTION:
      MAX_COMPETITORS: 3
      MAX_TARGET_IOU: 0.1
      NMS_IOU: 0.5
      MIN_OBSERVATIONS: 3
      EMA_DECAY: 0.9
      UTILITY_TOLERANCE: 0.02
    FUSION:
      MAX_RESIDUAL_RATIO: 0.25
      EPS: 1.0e-6
TRAIN:
  RELIABLE_CLOSED_LOOP: false
  CLIP_FRAMES: 12
  WARMUP_FRAMES: 4
  MAX_UPDATES: 4000
TEST:
  FIRST_FRAME_PROTOCOL: strict_language
```

`MIN_QUALITY=0.5` 和旧新颖性阈值保留原键。可靠写入启用时，先按新策略确认，再执行兼容的 ROI/质量校验；配置要保证底层阈值不比新阈值更严而造成隐式拦截。初始实验组：

| 组 | 记忆 | 可靠写入 | 干扰保留 | 受限融合 | 训练范围 |
|---|---|---|---|---|---|
| M0 | 关 | — | — | — | 不训练，官方起点 |
| M1 | 关 | — | — | — | 仅 CENTER |
| M2 | 开 | 关 | 关 | 关 | 原读取器＋CENTER |
| M3 | 开 | 开 | 关 | 关 | 同 M2 |
| M4 | 开 | 关 | 开 | 关 | 同 M2 |
| M5 | 开 | 关 | 关 | 开 | 同 M2 |
| M6 | 开 | 开 | 开 | 开 | 同 M2 |

M1–M6 使用共同的新闭环、数据、种子、BN、定位损失和步数。M2 是旧记忆规则在新训练中的重训对照，不将旧 20000 步权重当作配对起点。旧权重仅用于明确标记的推理干预诊断。

12 帧尤其加上确认延迟后，长期库常无法填满；M4 的短训不能充分检验淘汰机制。正式宣称长期保留有效前，必须补长视频闭环评估，以及 M2/M4/M6 相同预算的至少 32 帧适配短训；实际填满比例和淘汰事件不足时延长片段而不是虚报已验证长期机制。

## 11. 实施批次及每批完成条件

| 批次 | 工作与依赖 | 可交付结果 | 完成条件 |
|---|---|---|---|
| P0 | 配置、数据审计、初始化和共同协议登记 | manifest、配置草案、加载审计 | 实际权重/帧可读；训练验证测试交集与局限有记录 |
| P1 | 先独立实现残差限幅 | 限幅函数和检查 | 空库/ρ=0 恒等，幅度有界、两步梯度有效、旧权重可加载 |
| P2 | 抽取候选决策、裁剪映射和逐帧状态 | 共享 tracking_core | 旧输入决策回放一致；非方图坐标正确；首帧实际预测 |
| P3 | 写入状态机与新管理器 | 可靠写入和因果状态检查 | bootstrap、连续写入、歧义、快速运动与恢复不死锁 |
| P4 | 竞争候选与长期保留 | 保留策略、事件日志 | 无竞争者/满库/未知统计有确定行为；容量不增加 |
| P5 | 新训练数据、actor、单卡与双卡入口 | 闭环训练和恢复检查 | 训练测试决策一致、无 GT 泄漏、冻结范围和 DDP 更新正确 |
| P6 | 固定验证与逐帧离线诊断 | 严格评测入口、诊断汇总 | 逐帧预测齐全、拒绝原因与污染统计可追溯 |
| P7 | 先工程短测，再配对筛查及正式预算 | M0–M6 全量记录 | 预先固定选模规则；完整终点结果和失败记录齐全 |

每批可独立审阅。实施任务的首个可运行里程碑是 P1–P3：能加载旧记忆权重、严格首帧运行并输出写入与限幅日志；在 P4/P5 完成前，只能称为推理诊断版本。

## 12. 验证清单和结果记录

轻量检查应覆盖以下实际失败边界，避免仅验证“函数能调用”：

1. 相同初始状态和权重，三个新开关关闭后复现旧决策；空库和零残差不改变网络输出。
2. 不同时刻和不同视频状态隔离；同一帧不能先写后读；拒绝后不会把 pending 当作已确认条目。
3. 两个相似候选交替胜出、快速运动、遮挡后重现时，能进入/退出待确认状态；不能要求 GT 或旧模板才能恢复。
4. 无竞争候选、统计缺失和单条长期记忆不会出现 max 空集合、NaN 或任意淘汰。
5. 限幅前后范数、零向量和混合空库 batch；FP32 主流程及拟启用精度各自核验。
6. 预测首帧并连续跟踪两帧；评测修改 GT 后预测和写入轨迹完全相同，只有离线指标改变。
7. 相同裁剪输出分别经训练/推理核心回放，框、候选 ID、分数、写入动作和 memory IDs 一致。
8. 框出裁剪区域、无效标注、空监督批次的 loss 和 DDP 行为明确；随机恢复检查不掩盖失败。
9. 同权重和状态重载后下一帧预测一致；模型参数中不混入视频 ROI 内容。

最终记录整体 AUC/PRE/NPR、首帧 IoU、逐视频配对差值；污染指标只对有有效 GT 的入库事件统计，并同时报告 GT 未知比例。主污染诊断先固定 IoU<0.3，另附 IoU<0.1/<0.5 的敏感性结果，不按结果挑阈值。错误条目未被淘汰的情况按截尾存活时长报告，不当作已经清除。

遮挡重现的观察窗口、可见性定义、连续 IoU 恢复条件在分析前登记。记录写入率、确认延迟、竞争候选数量、长短期占用、辨识统计成熟比例、淘汰次数、读取年龄、残差比及单视频延迟；必须覆盖成功与失败案例。

首轮按更新 1000/2000/3000/4000 保存并完整验证固定视频子集，按三来源等权 AUC 选择，平分选较早步数；正式完整验证清单与规模在 P0 固定。标准测试仅在方案和检查点冻结后执行；没有未参与开发的确认数据时如实标为探索性。更多阈值组合及种子属于另行登记的实验。

## 13. 运行入口

以下 CLI 已提供。数据清单、设备、权重、输出目录必须显式指定；W&B 为可选监控，默认关闭，本地日志始终保留。数据清单采用 `official-train-video-split-v1` 格式。

```bash
# P1–P4：纯逻辑、坐标与数值检查
python tracking/check_reliable_memory.py --device cpu

# P5：数步训练与保存/恢复检查
python tracking/train_reliable_memory.py \
  --config mvlm_TF_reliable_memory --group M6 \
  --manifest /absolute/path/to/manifest.json \
  --checkpoint /absolute/path/to/MVLM_TF.pth.tar \
  --max-updates 4 --global-batch 2 --device cuda:0 \
  --out /absolute/path/to/new-smoke-directory

# P6：预先固定的视频清单；不从标准测试分数筛选视频
python tracking/evaluate_reliable_memory.py \
  --config mvlm_TF_reliable_memory --group M6 \
  --checkpoint /absolute/path/to/trained-memory-checkpoint.pth.tar \
  --manifest /absolute/path/to/frozen-validation.json \
  --protocol strict_language --device cuda:0 \
  --out /absolute/path/to/new-evaluation-directory
```

任何续跑先核查进程、已有结果 manifest 和哈希；存在目录不自动覆盖。正式开始前保存预检结论和实际配置，失败同样记入实验记录。

### 可选 W&B 监控

上述训练命令追加 `--wandb --wandb-project mvlm-reliable-memory --wandb-name M6-seed20261116` 即可联网记录（账号使用现有 `wandb login` 配置）。追加 `--wandb-mode offline` 可先在本机验证，离线验证不会上传。日志间隔由 `--wandb-log-interval 20` 控制，首步和停止步始终记录。也可直接修改配置中的 `WANDB` 段；命令行优先于同名 `WANDB_*` 环境变量，再优先于 YAML。仅设置环境变量不会自动开启监控，仍需 `--wandb` 或 `WANDB.ENABLED: true`。

监控包含全局平均损失、全局监督/写入/拒绝帧数、写入与拒绝率、学习率和梯度范数；容量、候选分数、拒绝原因、融合残差比/限幅比例、耗时和显存单独标记为 rank0 指标。多卡只有 rank0 创建 W&B run。未增加验证循环，因而不会凭空生成验证集精度曲线。

训练进程先写 `wandb_monitor/session-*/events.jsonl`，独立进程连接 SDK。SDK 缺失、认证或网络错误不终止训练；查看各 session 的 `worker.log`、`status.json` 确认是否连接成功。退出时等待最多约 6 秒，未完成的监控保留本地事件。缺少依赖可在训练环境安装 `wandb>=0.22,<0.23`（本次验证版本 0.22.3）。不上传模型权重、视频或源码。

`--resume` 时应保留相同的 W&B 参数与输出目录；联网模式沿用 `wandb_monitor/run.json` 的 ID，通过实际更新步数续接。离线恢复沿用 ID，但 SDK 会为每次启动生成独立离线目录，**不等于云端已经合并**。如需上传，可在训练结束后对选定离线目录手动使用 `wandb sync`。源码审计包含监控模块；旧版无监控训练快照不能跳过源码一致性检查强行续跑。

首版已覆盖记忆策略、共享逐帧流程、训练/评测和检查入口。正式配对实验前，继续补齐完整数据审计、保留全部候选检查点、周期完整验证与扩展诊断；不能把本次短测当作 P7 精度研究已经完成。
