# ENS-02团队交接包

请解压到独立目录，不要直接覆盖朋友项目的src/configs。

- configs/：R1、R2推理参数、固定50:50与权重SHA。
- src/：复用原项目的DenseNet加载、预处理、评价、G0聚合；ens02_infer.py为独立入口。
- tests/：合成数据测试，无患者数据。
- docs/：模型来源、实验分数、许可及限制。
- models/：仅私下发送的两份最佳Checkpoint，放在权重ZIP中；禁止上传公开仓库。

## 安装与运行
优先复用朋友兼容的PyTorch环境；requirements.txt是本机已测版本，不是官方T4认证镜像。不需要timm，不下载ImageNet权重。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
python -m src.ens02_infer --csv /实际数据路径/valid.csv --images /实际数据路径/valid --output outputs/official_valid_run1 --device cuda
```

Windows PowerShell可使用：
```powershell
& "C:\计算机能力挑战赛\.venv_ra\Scripts\python.exe" -B -X utf8 -u -m src.ens02_infer --csv "C:\实际路径\valid.csv" --images "C:\实际路径\valid" --output "outputs/official_valid_run1" --device cuda
```

先cd到本包根目录。把权重ZIP解压到同一根目录，确保存在models/R1_best_model.pth和models/R2_best_model.pth。只需修改命令中的官方valid CSV及图片目录，无需修改归一化或384配置。入口限定官方valid1422图/859 Study；没有正式test自动推理或上传功能。

输出R1/R2逐图和Study缓存、ENS02四列8590行预测、真实标签及metrics.json。四列Predict_class是类别编号，不是真实阳性标签；请按ID关联。始终先各模型Sigmoid、Study图片平均，再两模型等权平均。

本包仅准备交接，不自动执行完整推理。解压副本的单元测试与两份Checkpoint严格加载结果见DELIVERY_CHECK.json。跨设备数值完全一致未保证；正式T4计时、cost.py及最终材料尚需完成。不要根据test分数搜索集成权重。
