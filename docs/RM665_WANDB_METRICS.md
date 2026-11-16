# RM65＋SG100：W&B 指标与单位

本说明对应本轮新增的日志层。数据、模型输入输出、归一化与控制接口继续使用弧度；仅上报的物理误差换算为 degree。其他机器人任务的指标不转换。
旧的无单位物理 L1 键不再用于新日志，以免将 rad 与 degree 混入同一字段。已有 W&B 历史记录不会被重写。

## Loss 与梯度

以下数值不是角度，不进行 degree 换算：

- `loss`：665 日志窗口内的总 loss 平均值。
- `ce_loss`：训练 AR 分支返回的 CE，可能包含动作及其他有效监督 token，并沿用模型中的损失权重。
- `fm_loss`：训练 FM 分支返回的损失，沿用模型中的损失权重。
- `grad_norm`：现有训练代码记录的梯度裁剪前范数。
- `eval/action/teacher_forcing/ar_action_ce_loss`：当前验证 batch 的模型 CE；字段名称不代表另算了纯动作 token 的 CE。
- `eval/action/teacher_forcing/fm_loss`：当前验证 batch 的 FM loss。

## 策略预测的物理误差

以下指标保留原评估函数的目标与掩码语义：与反归一化的训练标签比较；`unclipped` 表示预测值尚未做物理限幅，`bounded` 表示按部署边界处理后的预测。训练标签经历过归一化处理，不应将其与未改动的原始记录标签混为一谈。
分组名来自 merger 的真实关节映射，不再把 left_control/right_control 等模型槽位当成机器人关节名称。

完整 W&B 字段：

- `eval/action/rollout/ar_action_mae_deg_unclipped`
- `eval/action/rollout/ar_action_mae_deg_bounded`
- `eval/action/rollout/fm_action_mae_deg_unclipped`
- `eval/action/rollout/fm_action_mae_deg_bounded`

RM65 六关节：
- `eval/action/rollout/per_group/rm65_arm_6d/ar_action_mae_deg_unclipped`
- `eval/action/rollout/per_group/rm65_arm_6d/fm_action_mae_deg_unclipped`

SG100 前六关节：
- `eval/action/rollout/per_group/sg100_thumb_index_6d/ar_action_mae_deg_unclipped`
- `eval/action/rollout/per_group/sg100_thumb_index_6d/fm_action_mae_deg_unclipped`

SG100 后五关节：
- `eval/action/rollout/per_group/sg100_middle_ring_5d/ar_action_mae_deg_unclipped`
- `eval/action/rollout/per_group/sg100_middle_ring_5d/fm_action_mae_deg_unclipped`

按 embodiment 汇总的物理 MAE：

- `per_emb/rm65_sg100/eval/ar_action_mae_deg_unclipped`
- `per_emb/rm65_sg100/eval/fm_action_mae_deg_unclipped`

MAE 是有效维度上的平均绝对误差；归一化空间的 `eval/action/rollout/ar_action_l1_normalized`、准确率与限幅比例均为无量纲值，保持原数值。

## Codec encode→decode 重建

在每次周期验证的同一个 held-out batch 上运行真实 codec 后端的 encode/decode，不运行 AR 预测来替代 codec 重建。输入取自实际 `samples[].action.value`；使用与训练相同的 RVQ residual 数。codec 临时进入 eval/no-grad 模式，结束后恢复原模块模式，不更新权重或 EMA。

所有以下重建误差均在逆变换、还原相对机械臂动作之后，以物理 degree 计算，且在物理限幅之前计算。padding 关节和 padding 时间步被排除。

- `vs_codec_input`：将 decode 结果与 codec 输入分别还原到物理空间后比较，隔离 codec 量化重建误差。
- `vs_recorded_action`：与批次中未改动的原始绝对关节标签 `gt_action` 比较，包含预处理/归一化表示损失以及 codec 量化损失。
- `mae_deg`：有效标量的平均绝对角度误差。
- `max_abs_error_deg`：当前验证 batch（多卡时为全部 rank）中的最大绝对角度误差。
- `nonfinite_fraction`：非有限误差所占比例；非零时 MAE/max 报为 inf，避免悄悄排除异常值后给出好看的均值。
- `valid_scalar_count`：排除 padding 后参与统计的标量数。MAE 按全部 rank 的误差和、标量数加权汇总。若没有有效标量，MAE/max 为 NaN，而不是伪造零误差。

完整 W&B 字段：

- `eval/codec_encode_decode_unclipped/rvq_num_residuals`：本次真实使用的 RVQ residual 数，当前配置应为 2。

RM65 六关节：
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/mae_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/max_abs_error_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/nonfinite_fraction_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/valid_scalar_count_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/mae_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/max_abs_error_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/nonfinite_fraction_vs_codec_input`
- `eval/codec_encode_decode_unclipped/rm65_arm_6d/valid_scalar_count_vs_codec_input`

SG100 前六关节：
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/mae_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/max_abs_error_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/nonfinite_fraction_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/valid_scalar_count_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/mae_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/max_abs_error_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/nonfinite_fraction_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_thumb_index_6d/valid_scalar_count_vs_codec_input`

SG100 后五关节：
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/mae_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/max_abs_error_deg_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/nonfinite_fraction_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/valid_scalar_count_vs_recorded_action`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/mae_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/max_abs_error_deg_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/nonfinite_fraction_vs_codec_input`
- `eval/codec_encode_decode_unclipped/sg100_middle_ring_5d/valid_scalar_count_vs_codec_input`

Codec 是共享的冻结预训练模块。这些曲线随验证 batch 改变而波动，不表示 codec 本身经过了策略训练；不要要求它们随着训练 step 单调下降。

## 记录条件与频率

- 启用 `logger.type=wandb`，由外部环境提供 W&B 认证；本轮不加入自动登录代码。
- `G05_VAL_DATASET_DIR` 指向独立 40-episode 验证集。
- `logger.log_steps=10` 时训练 loss/梯度每 10 个 optimizer step 记录。
- `eval_steps=100` 时物理预测误差及 codec 重建误差每 100 个 optimizer step 记录。当前代码每次取一个 validation batch，不是完整遍历 40 条 episode；验证步数应与 log_steps 对齐。
- 本轮总步数为 10000 且仅保留最终策略 checkpoint 时，可设 `checkpointing_steps=10001`，不改变上述周期评估频率。

## 本轮验证范围

结果：18 项 CPU 检查全部通过，Python 3.10 语法检查通过。

CPU 检查覆盖真实 Hydra 构建的 6+6+5 processor、度数换算、模型槽位到真实关节组的映射、padding 排除、相对机械臂逆变换、同训练 RVQ 数、codec 模式恢复、非有限误差显式上报、按样本数加权的多卡统计，以及周期 evaluator 到日志字典的接线。codec 使用可控替身检验计算和集成；本轮未连接云端，未加载真实 codec checkpoint 或做 H20 GPU 测试。
临时测试文件仅在项目目录外运行，完成后删除；项目中不新增测试文件。
