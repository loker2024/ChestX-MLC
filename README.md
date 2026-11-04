# SW-02：Swin V2-Tiny-384 + ASL + EMA

仅Loss由SW01官方验证20轮BCE改为ASL，gamma_pos=1、gamma_neg=4、clip=0.05、eps=1e-8、detach_focal_weight=true（沿用C1实现）。
不加载SW01最佳权重续训，从同一原始TorchVision ImageNet文件开始。
普通/EMA独立保存最佳，以官方valid Study Macro-AUC选择。patience3/min_delta0.0001，两者都满足才停止，最多20轮。
训练ASL与验证图片BCE分别命名，不直接比较Loss数值。G0和0.5辅助F1不变。

## 对照与数据边界

本地已完成Swin为v2/12轮（EMA最佳0.808079），不是本次的SW01官方20轮对照。审查见`docs/SW02_SW01_AUDIT.md`。
服务器必须提供SW01真实目录（train_config.json、split_audit.json、history.csv、run_summary.json、best_normal.pth、best_ema.pth）。
程序在GPU启动前检查：官方train18513 Study/30344图，官方valid859 Study/1422图，患者/Study/图片不交叉；同源指纹；配置仅损失及实验名称不同；普通/EMA最佳权重SHA及历史指标一致。
缺SW01记录只能做预检，不允许正式训练；不使用v2的0.808079比较官方valid。配置不同先回传差异，不擅自调整。
验证集不加入训练；不读取test。

## 固定配置

`configs/experiments/SW02_SwinV2_ASL_EMA.json`：384保持比例黑边填充、ImageNet归一化；与SW01相同温和增强、Study采样及逐Study均衡归约；物理最多4 Study/8图，累积约32图、尾组按实际Study数归一；worker8。
AdamW，骨干2e-5/头2e-4，WD0.03；warmup2+Cosine到1e-6；20轮；EMA0.999，仅成功Step后更新；clip1；seed42。
原生BF16默认（已修复旧FP16溢出），Loss及归约始终FP32。CPU玩具模型测试BF16/FP16不能证明GPU Swin FP16稳定；不自动切精度。
T4不支持原生BF16训练，本包会停止，需先反馈设备。若必须改精度，SW01与SW02均须同精度重新建立可比对照，不能把改精度混成仅改ASL。
不承诺新AUC或跨设备完全复现；原始权重按用户确认许可采用，不冒称独立核验主办方书面答复。

## Linux服务器命令

解压到新目录，不覆盖SW01。原始数据/模型不在ZIP内，服务器复用已有文件。

```bash
cd /root/SW02_SwinV2_ASL_EMA_v1
export PY=/root/my-baseline/.venv/bin/python
export REF=/root/实际SW01官方验证20轮结果目录

"$PY" scripts/swinv2_server_paths.py --official \
  --data /data/cxr_first \
  --weight /data/pretrained/swin_v2_t-b137f0e2.pth \
  --output configs/internal_ab/paths.swinv2.official.json

mkdir -p outputs/logs
LOG="outputs/logs/launcher_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/run_sw02.sh > "$LOG" 2>&1 < /dev/null &
echo "PID: $! 日志: $LOG"
```

脚本依次运行CPU测试、真实数据/原始权重/SW01对照预检、GPU单更新冒烟、正式训练；任一步失败即停止。手动启动后才会使用GPU。
依赖沿用SW01 CUDA环境，参考requirements-baseline.txt；开发环境Python3.13.11/torch2.11.0+cu128/torchvision0.26.0+cu128。新环境先安装清单并`python -m pip check`，不要覆盖SW01环境。

## 本地PowerShell预检与训练

项目根目录执行（数据路径替换为本机真实位置）：

```powershell
cd "C:\计算机能力挑战赛"
$PY = "$PWD\.venv_ra\Scripts\python.exe"
$REF = "C:\实际SW01官方验证20轮结果目录"
$TAG = Get-Date -Format "yyyyMMdd_HHmmss"

& $PY scripts/swinv2_server_paths.py --official --data "$PWD\cxr_first\cxr_first" --weight "$PWD\checkpoints\pretrained\swin_v2_t-b137f0e2.pth" --output configs/internal_ab/paths.sw02.local.json

& $PY -B -X utf8 -u -m src.run_swinv2 --config configs/experiments/SW02_SwinV2_ASL_EMA.json --paths configs/internal_ab/paths.sw02.local.json --reference-run "$REF" --mode check --output "outputs/SW02_check_$TAG"
if ($LASTEXITCODE -ne 0) { throw "预检失败" }

# 以下由本人手动执行：先GPU冒烟，成功后训练。
& $PY -B -X utf8 -u -m src.run_swinv2 --config configs/experiments/SW02_SwinV2_ASL_EMA.json --paths configs/internal_ab/paths.sw02.local.json --reference-run "$REF" --mode smoke --execute --output "outputs/SW02_smoke_$TAG"
if ($LASTEXITCODE -ne 0) { throw "冒烟失败，不训练" }
& $PY -B -X utf8 -u -m src.run_swinv2 --config configs/experiments/SW02_SwinV2_ASL_EMA.json --paths configs/internal_ab/paths.sw02.local.json --reference-run "$REF" --mode train --execute --output "outputs/SW02_run_$TAG"
```

没有REF时可省略`--reference-run`执行check查看数据，但ready_for_formal_training=false，不是训练许可。

## 结果与官方Valid导出

输出独立`outputs/X_SWINV2_SW02_ASL_EMA_时间_PID/`，保存配置、SW01_reference_audit.json、split_audit.json、history.csv（train_study_asl/validation_image_bce）、每轮10类AUC/普通EMA指标和逐图/Study缓存、各自最佳Checkpoint/SHA、selected_checkpoint.json、run_summary.json与清单。
每轮已经完成完整官方valid评价，不必另跑推理获得成绩。回传history、summary、validation_metrics、selected_checkpoint、配置、SW01_reference_audit与最佳轮预测。
如需四列8590行官方Valid预测，训练结束后本人手动执行：

```powershell
& $PY -B -X utf8 -u -m src.run_swinv2 --mode valid --execute --paths configs/internal_ab/paths.sw02.local.json --run "outputs/SW02_run_替换实际时间" --valid-images "$PWD\cxr_first\cxr_first\valid" --output "outputs/SW02_valid_export_$TAG"
```

服务器同样使用`$PY -m src.run_swinv2 --mode valid --execute --paths configs/internal_ab/paths.swinv2.official.json --run 实际输出目录 --valid-images /data/cxr_first/valid --output 新导出目录`。
记录所选普通/EMA，禁止混淆。当前不支持中断无缝恢复。最终T4计时及500MB整包限制尚未在服务器实测。

原始预训练SHA：`b137f0e2d744bbe1e82ad10ddb8dc938676cf04a4332eae66e45c2048ac09711`。
公开地址：https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth
