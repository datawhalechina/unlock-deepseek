# Unlock-DeepSeek

<p align="center"> <img src="https://avatars.githubusercontent.com/u/46047812?s=200&v=4" style="width: 40%;" id="title-icon">  </p>
<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
<!-- <a href="" target="_blank" style="margin-left: 6px">🤗</a> <a href="https://modelscope.cn/models/linjh1118/WidsoMenter-8B/summary" target="_blank" style="margin-left: 6px">HuggingFace</a>  • | 
<a href="" target="_blank" style="margin-left: 10px">🤖</a> <a href="https://modelscope.cn/models/linjh1118/WidsoMenter-8B/summary" target="_blank" style="margin-left: 6px">ModelScope</a>  • |
<a href="" target="_blank" style="margin-left: 10px">📃</a> <a href="./resources/WisdoMentor_tech_report.pdf" target="_blank" style="margin-left: 6px">[Wisdom-8B @ arxiv]</a>

</p>

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
🍭 <a href="http://wisdomentor.jludreamworks.com" target="_blank"  style="margin-left: 6px">WisdoMentor在线体验</a> • |
<a href="" target="_blank" style="margin-left: 10px">💬</a> <a href="./resources/wechat.md" target="_blank"  style="margin-left: 6px">WeChat</a> 
</p> -->

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
<a href="https://github.com/datawhalechina/unlock-deepseek/blob/main/README@en.md" target="_blank"  style="margin-left: 6px">English Readme</a>  • |
<a href="https://github.com/datawhalechina/unlock-deepseek/blob/main/README.md" target="_blank"  style="margin-left: 6px">中文 Readme</a> 
</p>

面向广泛 AI 研究爱好者群体的 DeepSeek 系列工作解读、扩展和复现，致力于传播 DeepSeek 在 AGI 实践之路上的创新性成果，并提供从 0 代码实现，打造 LLM 前沿技术教学项目。

### 项目受众

- 有大语言模型相关概念基础，具有大学数理能力的初学者
- 希望进一步了解深度推理的学习者
- 希望将推理模型运用到实际工作中的从业人员

### 项目亮点

我们将 DeepSeek-R1 及其系列工作拆分为三个重要部分：

- MoE
- Reasoning
- Infra

与大众的关注性价比优势不同，我们关注 DeepSeek 在实践 AGI 之路的创新性工作，致力于将 DeepSeek 现有公开工作细分拆解，向更广泛的 AI 研究爱好者讲述清楚其中的创新方法细节，同时我们会对比介绍同期其他类似工作（如 Kimi-K1.5），呈现 AGI 之路的不同可能性

我们也将结合其他社区的工作，探索 DeepSeek-R1 的复现方案，提供中文复现教程

## 目录

1. MoE: DeepSeek 所坚持的架构

   1. MoE 简介
   2. MoE 结构的代码实现
   3. DeepSeek MoE

2. Reasoning: DeepSeek-R1 的核心能力

   1. 推理模型介绍
      1. LLM and Reasoning
      2. 推理效果可视化
      3. OpenAI-o1与Inference Scaling Law
      4. Qwen-QwQ and Qwen-QVQ
      5. DeepSeek-R1 and DeepSeek-R1-Zero
      6. Kimi-K1.5

   2. 推理模型关键算法原理
      1. CoT，ToT，GoT
      2. 蒙特卡洛树搜索
      3. 强化学习概念速览
      4. DPO、PPO、GRPO
      
3. Infra: DeepSeek 训练高效且便宜的关键
   - FlashMLA
   - DeepEP
   - DeepGEMM
   - DualPipe & EPLB
   - 3FS


## 贡献者名单

| 姓名   | 职责          | 简介       |
| :----- | :------------ | :--------- |
| [骆秀韬](https://github.com/anine09) | 项目负责人    | 似然实验室 |
| [姜舒凡](https://github.com/Tsumugii24) | 项目负责人    | 华东理工大学 |
| [陈嘉诺](https://github.com/Tangent-90C) | 负责Infra部分 |  广州大学   |
| [林景豪](https://github.com/linjh1118) | GRPO 算法解读 |     智谱       |
| [邓恺俊](https://github.com/kedreamix) | Kimi-K1.5论文解读 | 深圳大学 |
| [刘洋]() | MCTS 算法解读 |            |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈
- 如果你想参与贡献本项目，欢迎提Pull request

## 提交规范
feat: 用于新功能（例如，feat: 添加新的 AI 模型）
fix: 用于错误修复（例如，fix: 解决内存泄漏问题）
docs: 用于文档更新（例如，docs: 更新贡献指南）
style: 用于代码风格变更（例如，style: 重构代码格式）
refactor: 用于代码重构（例如，refactor: 优化数据处理）
test: 用于添加或更新测试（例如，test: 为新功能添加单元测试）
chore: 用于维护任务（例如，chore: 更新依赖项）


## 致谢
我们衷心感谢以下开源资源和帮助，使我们能够构建这个项目：[DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [trl](https://github.com/huggingface/trl), [mini-deepseek-r1](https://www.philschmid.de/mini-deepseek-r1)（我们的初始代码库），[TinyZero](https://github.com/Jiayi-Pan/TinyZero)，[flash-attn](https://github.com/Dao-AILab/flash-attention)，[modelscope](https://github.com/modelscope/modelscope)，[vllm](https://github.com/vllm-project/vllm)。



## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
