# convert_fix

- 新增根目录脚本 `convert_model_to_paddle.py`，用于把本项目 Torch 本地权重目录转换成 Paddle `model_state.pdparams`。
- 转换脚本不依赖 `poseidon_paddle` 当前实现，避免被未修复的 Paddle 侧模型/Trainer 阻塞。
- 保留参数名不变，只对 `torch.nn.Linear.weight` 做转置；原因是 Paddle `Linear` 权重布局与 Torch 相反。
- 输出目录会复制 `config.json` 并写入 `conversion_summary.json`，方便后续核对转换结果。
- 当前脚本是项目定制版，仅面向 `poseidon/scOT/model.py` 中的 `ScOT` 架构，不作为通用 Hugging Face 转换器。
- 手工重写 `poseidon_paddle/scOT/model.py`，移除 `paddleformers` 和 Hugging Face 运行时依赖，补齐 `DropPath`、窗口划分/还原、SwinV2 attention/intermediate/output，并新增本地 `ScOTConfig.from_pretrained/save_pretrained` 与 `ScOT.from_pretrained/save_pretrained`。
- `poseidon_paddle/scOT/model.py` 保持 Torch 权重参数名层级不变，使根目录转换脚本产出的 `model_state.pdparams` 可直接加载。
- 手工重写 `poseidon_paddle/scOT/trainer.py` 为单卡 Paddle 训练器，仅保留项目实际依赖的 `TrainingArguments`、`Trainer.train/predict/save_model/set_ar_steps` 和参数分组逻辑。
- `poseidon_paddle/scOT/train.py` 保留原 CLI 形态，但移除 `accelerate`/`transformers` 训练依赖，改为单卡训练流程；`push_to_hf_hub` 当前改为显式 `NotImplementedError`。
- `poseidon_paddle/scOT/problems/base.py` 将原来的 `broadcast_object_list` 替换为单卡直通，若初始化了 Paddle 分布式则仅做 barrier。
- `poseidon_paddle/scOT/problems/base.py` 现在按进程懒打开 HDF5 reader，避免把父进程里创建的文件句柄带入 Paddle DataLoader worker 导致崩溃；各 HDF5 数据集的 `self.reader` 初始化改为只记录文件路径。
- `poseidon_paddle/scOT/inference.py` 改为依赖本地 `EvalPrediction/Trainer`，修正了自动转换留下的 `numpy`/`Tensor` 混用问题。
- `poseidon_paddle/minimal_forward.py` 改为加载 `models/camlab-ethz_Poseidon-T_paddle`，并使用 Paddle 设备与张量 API。
- `poseidon_paddle/minimal_train.sh` 改为直接使用 `python` 启动，不再依赖 `accelerate launch`。
- `poseidon_paddle/scOT/train.py` 现在显式选择单卡运行设备，并恢复与 Torch 入口一致的 `metric_for_best_model="loss"` 和 early stopping 回调接线。
- `poseidon_paddle/scOT/trainer.py` 补齐了 step 级日志、eval 运行时指标、W&B 日志上报、best model 选择、训练状态保存与 resume；训练期终端输出按 Torch 语义对齐，不复刻 Hugging Face `tqdm` 控制字符。
- Paddle `AdamW` 的内部参数名不如 Torch 稳定，resume 时可能无法可靠恢复 optimizer state；当前实现保存/恢复 optimizer state 和所有 scheduler state dicts，恢复失败时跳过并打印警告。
- `poseidon_paddle/scOT/inference.py` 现在与训练入口共用相同的设备选择逻辑，避免推理和训练落在不同设备路径上。
- `poseidon_paddle/scOT/utils.py` 新增统一的 Paddle 设备解析 helper，优先复用当前设备，其次按 Paddle 已注册的标准设备/自定义设备类型选择，便于后续迁移到 custom device。
- Paddle `DataLoader` 没有与 Torch `pin_memory` 完全等价的参数，本轮保留配置字段但不做 1:1 底层复刻；这不改变当前单卡训练的数值语义。
- `poseidon_paddle/scOT/trainer.py` 不再把 `places=self.device` 传给 Paddle `DataLoader`；数据加载保持在 CPU 侧，batch 在主进程统一搬到目标设备。
- 为避免非 CPU 设备上的 Paddle worker 随机崩溃，`poseidon_paddle/scOT/trainer.py` 现在会在 `gpu/custom device` 路径强制使用单进程数据加载。
  Embedding / Time-Embedding 学习率 (poseidon_paddle/scOT/trainer.py)
  根因: Paddle 的 AdamW 在参数组指定 float learning_rate 时，该组不受全局 scheduler 影响。原代码传入的是 scale = embedding_lr / base_lr（比例值），导致：
  1. 实际学习率值错误（比例而非真实值）
  2. 不跟随 warmup / cosine decay 调度

  修复方案:
  - create_optimizer: 为 embedding 和 time_embedding 参数组各创建独立的 LambdaDecay scheduler（以各自实际 LR 为 base_lr），确保每个组独立跟随
    warmup/decay 调度
  - create_scheduler: 新增 base_lr 参数支持不同基线学习率
  - \_create_group_scheduler: 新方法，为参数组创建独立 scheduler
  - 训练循环: 遍历 self.\_all_lr_schedulers 统一 step 所有 scheduler
  - checkpoint: \_save_training_state / \_load_training_state 保存/恢复所有 scheduler state dicts
  Paddle normalize backward NaN 修复 (poseidon_paddle/scOT/model.py)
  根因: Paddle 的 `paddle.nn.functional.normalize` backward 对 L2 范数为零的 token 会产生极大梯度值（除以接近零的范数），在 float32 下溢出为 NaN/Inf。Airfoil 等数据集有大量 padding token（~94%），patch embedding + LayerNorm 后这些 token 的范数恰好为零。

  修复方案: 在 Swinv2SelfAttention.forward() 中对 query_layer 和 key_layer 各加 `_NORM_STABILITY_EPS = 1e-6` 后再做 normalize，避免零范数触发梯度溢出。对非零 token 的影响可忽略不计（1e-6 相对于通常 1~10 的范数值）。
