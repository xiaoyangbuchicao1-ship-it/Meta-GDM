# Meta-GDM 主模型训练代码说明
本代码包用于提供论文中 Meta-GDM 主模型的核心训练源码，便于审稿人核验模型实现与训练流程。 
本项目只保留主模型训练所需代码，核心结构包括： 上下文编码器 + 变分潜变量 + 社会注意力机制 + SAC Actor-Critic 训练框架

## 1. 文件说明
| 文件 | 说明 |
| --- | --- |
| `env.py` | 群体观点演化环境，包含共识水平、非线性调整成本、公平性惩罚和边界约束 |
| `model.py` | 上下文编码器和社会注意力模块 |
| `agent.py` | Meta-GDM 智能体，包括 Actor、Critic 和 SAC 更新逻辑 |
| `replay_buffer.py` | 经验回放池 |
| `context_utils.py` | 历史交互上下文构造工具 |
| `train_meta.py` | 主模型训练与评估入口 |
| `plot_training_curves.py` | 根据训练历史 CSV 绘制训练曲线 |
| `requirements.txt` | Python 依赖 |

## 2. 环境安装

```bash
pip install -r requirements.txt
```
## 3. 主模型训练
默认训练命令：
```bash
python train_meta.py --device auto
```

## 5. 输出文件

运行后会生成 `results/` 目录，主要包括：

| 输出文件 | 说明 |
| --- | --- |
| `results/checkpoints/*.pth` | 训练得到的模型权重 |
| `results/checkpoints/*_training_history.csv` | 训练过程记录 |
| `results/figures/training_curves/*.png` | 训练曲线图，包括共识水平、Episode Return、Actor Loss 和 Critic Loss |
| `results/meta_gdm_main_eval_raw.csv` | 逐 episode 评估结果 |
| `results/meta_gdm_main_eval_summary.csv` | 聚合后的评估结果 |
。
