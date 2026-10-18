# Physiological tool selection — P7 scientific implementation

本目录是依据冻结方案 `PM_P6_20261010_v1` 新建的科学代码版本，不引用旧模型或旧实验结果。P7只实现适配器、模型组件和算法合同测试；真实数据的quick流水线、full性能评价与论文更新分别属于P8、P9及后续步骤，不能把本目录测试输出作为论文性能结果。

## 文件用途

| 文件 | 作用 |
|---|---|
| `config/` | P6协议、实际人员四池分配、全部监测记录的身份角色、环境版本的逐字节副本 |
| `pmtools/protocol.py` | 校验冻结哈希，禁止外部评价者进入训练／规划／conformal拟合 |
| `pmtools/adapters.py`、`schema.py` | 九个来源层的统一窗口接口、独立参考与在线信号分离、连续时间轴 |
| `pmtools/signals.py` | 完成窗口内的重采样、滤波、HR与特征、质量代理、双通道CNN输入 |
| `pmtools/models.py` | 等人员权重的逻辑回归；CPU CNN；训练人员三折选择轮数；EDA折内缩放 |
| `pmtools/safety.py` | 质量阈值、split conformal、严格预算、延迟收益规划、延后调用二级工具 |
| `pmtools/monitoring.py` | BOCPD、10/30秒调度、校准报警上限、真实决策时刻、事件／暴露评估 |
| `pmtools/statistics.py` | 人级损失、双腕先配对、10,000次配对聚类bootstrap、Holm、未定义值处理 |
| `pmtools/profiling.py`、`persistence.py` | 仅policy人员的CPU计时；带协议身份的新版模型保存与重载 |
| `tests/` | 可提交的算法单元／合成集成测试；与编辑论文的辅助脚本不同 |
| `scripts/validate_dataset.py` | 仅读取并核查数据接口；不会训练模型或生成性能指标 |

## 环境与运行

Python 3.12，CPU。依赖版本与 `config/frozen_environment.json` 一致，核心依赖在`pyproject.toml`。现有项目`.venv`可直接使用。PyTorch采用已安装的CPU构建；另一台机器应先按该构建配置官方CPU安装源，再装其余依赖，不能认为普通pip命令自动保证CPU轮子。

在本目录打开PyCharm，解释器选择`D:\大学学习\论文\论文项目\.venv\Scripts\python.exe`，运行模块`pytest`，工作目录选本目录，参数：

```text
-q -p no:cacheprovider --basetemp "D:\大学学习\论文\编辑有关的代码\PM_P7_20261010\pytest_user_new"
```

`pytest --basetemp`会清理指定测试临时目录，因此只能指定新的、专用的测试临时子目录；不得设置为项目根目录、数据目录或存放重要文件的位置。

数据接口验证示例（本目录终端）：

```powershell
python scripts/validate_dataset.py --data-root "D:\大学学习\论文\论文项目\URL" --source WESAD --role train --max-people 1 --output "D:\大学学习\论文\编辑有关的代码\PM_P7_20261010\user_schema_check_new.json"
```

原始信号不复制到投稿包，仍位于`论文项目/URL`。衍生资格索引在投稿包的`03_Data/qualified_indices`，代码用它们锁定纳排，不能按预测效果重新挑窗。移动投稿包时须保持`02_Code`和`03_Data`的相对结构；原始数据根用`--data-root`明确传入。只读解压后的官方文件，不执行来源库脚本／压缩包。仅加载已核验的本地PKL和本项目生成的模型；pickle/joblib可执行代码，不可加载不可信文件。

## 重要口径及实现细节

- AF/AFL联合vs官方N，不称纯AF/健康对照；VitalDB为围术期指端PPG，不称腕部外部验证。
- DaLiA参考HR的第k个原8秒段为`[2k,2k+8)`，以中心`2k+4`落入10秒窗的有限参考值中位数对齐。全部15人标签数须与可用信号时长相符。参考仅为工程目标，不能由PPG预测来调整时差。
- PPG-Sport29人双腕先在同一人同一时间求平均；GalaxyPPG23人单独报告。两库跨研究身份不重叠仅是来源假设。
- WESAD2为压力，1/3/4为非压力，混杂与其他状态不进入离线分类集。Stress-Predict30人为主要外部任务状态验证；WearableDevice两个协议18/17人只作分来源补充。a/b段继承同一人，但不能拼接采集间隙。
- SHDB全部合格记录按人隔离，`005/020`冲突副本不能被读取为合格记录。未知／混杂参考不删除信号，也不触发在线重置。
- PPG/ECG原始来源单位保留；不能为未知单位凭空补g或mV。固定SHDB ECG1经WFDB增益/基线换算为mV。EDA为文档声明的微西门子，未知单位直接报错。处理后特征及窗口MAD/CNN尺度是算法尺度，不是新物理单位。
- 使用完成的10秒窗后才做双向滤波，不能宣称零延迟在线滤波。决策时间=窗末+实际CPU耗时；未知节律不进入在线控制器。
- CNN EDA分支输入原始微西门子张量，`fit_cnn(...,modality='EDA')`在每个训练人员CV折内拟合尺度，最后在全部训练人员重拟合；不能先用全部人员预标准化后做CV。BVP/ECG按窗MAD缩放。
- 规划器拟合只用policy人员。实际调用只接收主工具概率／质量及剩余整数预算，不查看评价参考、二级概率或二级质量。次工具选择后才调用callback；低主质量不否决高质量次工具救援，次工具失败则弃答。
- 调用策略／预算／监测时长分别校准集合。传给`decide`的primary/fusion路径须绑定**同一个策略执行分布的Conformal对象**，其`key`必须是`(strategy, budget整数百分单位, 10)`；程序已拒绝不同策略、预算或不同路径单独阈值的混用。P8将验证完整流水线的实际分策略拟合与输出。
- 参考ECG/胸部HR纠正只是离线oracle，纠正后重新算误差；物理采集延迟记未测，绝不填虚构零毫秒。
- BOCPD采用高斯NIG预测及长度上限256，剪去超出上限的增长质量后对保留后验显式归一化；不是无限历史的精确后验。前6个有效tick禁报变点，变点后快窗总6个tick，随后重新收集30秒。
- 报警阈值先在policy集满足固定／自适应策略的非目标阳性决策比例各≤5%，无法达到90%目标事件检出时保留不可行标记。conformal弃答不能从阴性暴露分母扣时间；5%决策比例不等于FAR/h保证。
- 同一记录上的合格事件与报警一对一，发作末端半开。零报警、全弃答、单类AUROC、无阴性暴露等不伪装成有效成功。
- 所有聚类推断以人作为独立单位，跨来源保留来源层。conformal跨域／重复窗口覆盖只是实证检验，不宣称临床安全保证。

第三方原始数据许可不由本项目重新授权；来源清单见`03_Data/DATA_SOURCES.md`。本项目代码发布许可与实际公开仓库由作者在投稿发布阶段确认，本轮不代建GitHub仓库、不自动提交期刊。
