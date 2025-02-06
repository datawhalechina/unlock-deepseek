# Unlock-DeepSeek

面向广泛 AI 研究爱好者群体的 DeepSeek 系列工作解读、扩展和复现，致力于传播 DeepSeek 在 AGI 实践之路上的创新性成果，并提供从 0 代码实现，打造 LLM 前沿技术教学项目

### 项目受众

- 有大语言模型相关概念基础，具有大学数理能力的初学者
- 希望进一步了解深度推理的学习者
- 希望将推理模型运用到实际工作中的从业人员

### 项目亮点

我们将 DeepSeek-R1 及其系列工作拆分为三个重要部分：

- MoE
- Reasoning Models
- 关键要素 Keys（Data、Infra、...）

与大众的关注性价比优势不同，我们关注 DeepSeek 在实践 AGI 之路的创新性工作，致力于将 DeepSeek 现有公开工作细分拆解，向更广泛的 AI 研究爱好者讲述清楚其中的创新方法细节，同时我们会对比介绍同期其他类似工作（如 Kimi-K1.5），呈现 AGI 之路的不同可能性

我们也将结合其他社区的工作，探索 DeepSeek-R1 的复现方案，提供中文复现教程

## 目录

1. MoE: DeepSeek 所坚持的架构

   1. DeepSeek-R1 蒸馏模型（Qwen）的部署（self-llm/DeepSeek-R1-Distill-Qwen）
   2. MoE 历史脉络回顾
   3. MoE 从 0 实现（tiny-universe/Tiny MoE）
   4. [多个小节] DeepSeek 模型中的 MoE 设计解读（带实现）
2. Reasoning Models: DeepSeek-R1 的关键技术

   1. 推理模型介绍
      1. LLM and Reasoning
      2. 推理效果可视化
      3. OpenAI-o1与Inference Scaling Law
      4. Qwen-QwQ and Qwen-QVQ
      5. DeepSeek-R1 and DeepSeek-R1-Zero
      6. Kimi-K1.5
   2. 推理模型关键算法原理（`2.1 推理模型介绍`里涉及的技术尽量都覆盖到）
      1. CoT，ToT，GoT
      2. 蒙特卡洛树搜索 MCTS
      3. 强化学习概念速览
      4. DPO、PPO、GRPO
      5. ...
3. [实验性] Keys: DeepSeek 为什么便宜又好用

   由于缺失大量资料，这部分只能尽力而为

   - Data
   - Infra
   - Trick
   - Distill
   - ...

## 贡献者名单

| 姓名   | 职责          | 简介       |
| :----- | :------------ | :--------- |
| 骆秀韬 | 项目负责人    | 似然实验室 |
| 姜舒凡 | 项目负责人    |            |
| 陈嘉诺 | 负责Infra部分 |  广州大学   |
| 林景豪 | GRPO 算法解读 |            |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

`<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" />``</a><br />`本作品采用`<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">`知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议`</a>`进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
