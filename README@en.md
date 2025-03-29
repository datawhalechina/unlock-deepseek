# Unlock-DeepSeek

<p align="center"> <img src="https://avatars.githubusercontent.com/u/46047812?s=200&v=4" style="width: 40%;" id="title-icon">  </p>
<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
<!-- <a href="" target="_blank" style="margin-left: 6px">🤗</a> <a href="https://modelscope.cn/models/linjh1118/WidsoMenter-8B/summary" target="_blank" style="margin-left: 6px">HuggingFace</a>  • | 
<a href="" target="_blank" style="margin-left: 10px">🤖</a> <a href="https://modelscope.cn/models/linjh1118/WidsoMenter-8B/summary" target="_blank" style="margin-left: 6px">ModelScope</a>  • |
<a href="" target="_blank" style="margin-left: 10px">📃</a> <a href="./resources/WisdoMentor_tech_report.pdf" target="_blank" style="margin-left: 6px">[Wisdom-8B @ arxiv]</a>

</p>

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
🍭 <a href="http://wisdomentor.jludreamworks.com" target="_blank"  style="margin-left: 6px">WisdoMentor Online Experience</a> • |
<a href="" target="_blank" style="margin-left: 10px">💬</a> <a href="./resources/wechat.md" target="_blank"  style="margin-left: 6px">WeChat</a> 
</p> -->

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
<a href="https://github.com/datawhalechina/unlock-deepseek/blob/main/README.md" target="_blank"  style="margin-left: 6px">English Readme</a>  • |
<a href="https://github.com/datawhalechina/unlock-deepseek/blob/main/README@en.md" target="_blank"  style="margin-left: 6px">Chinese Readme</a> 
</p>

Unlock-DeepSeek is a series of works dedicated to interpreting, expanding, and reproducing DeepSeek's innovative achievements on the path to AGI for a wide audience of AI enthusiasts. It aims at disseminating these innovations and providing hands-on projects from scratch to master the cutting-edge technologies in large language models.

### Audience

- Beginners with foundational knowledge in large language models and university-level mathematical skills.
- Learners interested in understanding deep reasoning further.
- Professionals looking to apply reasoning models in their work.

### Highlights

We dissect the DeepSeek-R1 and its related works into three significant parts:

- MoE
- Reasoning Models
- Key Elements (Data, Infra, ...)

Instead of focusing merely on cost-effectiveness, we emphasize the innovative practices of DeepSeek towards achieving AGI, breaking down its publicly available work into digestible segments for a broader audience. We also compare and introduce similar works like Kimi-K1.5 to showcase different possibilities on the path to AGI.

Additionally, we will explore reproduction schemes for DeepSeek-R1 by integrating contributions from other communities, providing Chinese-language reproduction tutorials.

## Table of Contents

1. MoE: The Architecture Upheld by DeepSeek
   1. Deployment of DeepSeek-R1 Distilled Model (Qwen) (self-llm/DeepSeek-R1-Distill-Qwen)
   2. A Retrospective on the Evolution of MoE
   3. Implementing MoE from Scratch (tiny-universe/Tiny MoE)
   4. [Multiple Subsections] Decoding MoE Design in DeepSeek Models (with Implementation)
2. Reasoning Models: The Critical Technology of DeepSeek-R1
   1. Introduction to Reasoning Models
      1. LLM and Reasoning
      2. Visualization of Reasoning Effects
      3. OpenAI-o1 and Inference Scaling Law
      4. Qwen-QwQ and Qwen-QVQ
      5. DeepSeek-R1 and DeepSeek-R1-Zero
      6. Kimi-K1.5
   2. Key Algorithmic Principles of Reasoning Models (covering as much technology as possible introduced in `2.1 Introduction to Reasoning Models`)
      1. CoT, ToT, GoT
      2. Monte Carlo Tree Search MCTS
      3. Quick Overview of Reinforcement Learning Concepts
      4. DPO, PPO, GRPO
      5. ...
3. [Experimental] Keys: Why DeepSeek Is Cost-Effective and Efficient

Due to the lack of extensive documentation, this section can only be approached with our best efforts.

- Data
- Infra
- Tricks
- Distillation
- ...

## Contributors

| Name         | Role                | Bio       |
| :----------- | :------------------ | :-------- |
| [XiuTao Luo](https://github.com/anine09) | Project Leader    | SiLiang Lab |
| [ShuFan Jiang](https://github.com/Tsumugii24) | Project Leader    | East China University of Science and Technology |
| [JiaNuo Chen](https://github.com/Tangent-90C) | Responsible for Infra Part | Guangzhou University |
| [JingHao Lin](https://github.com/linjh1118) | Interpreting GRPO Algorithm | Zhipu.AI |
| [Kaijun Deng](https://github.com/kedreamix) | Kimi-K1.5 Paper Explanation | Shenzhen University |

## Contributing

- If you find any issues, feel free to open an issue or contact the [caretaker team](https://github.com/datawhalechina/DOPMC/blob/main/OP.md) if there's no response.
- If you wish to contribute to this project, please submit a pull request or reach out to the [caretaker team](https://github.com/datawhalechina/DOPMC/blob/main/OP.md) if it goes unanswered.
- If you're interested in starting a new project with Datawhale, follow the guidelines provided in the [Datawhale Open Source Project Guide](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md).

## Acknowledgments
Our heartfelt thanks go out to the following open-source resources and assistance that made this project possible: [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [trl](https://github.com/huggingface/trl), [mini-deepseek-r1](https://www.philschmid.de/mini-deepseek-r1) (our initial codebase), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), [flash-attn](https://github.com/Dao-AILab/flash-attention), [modelscope](https://github.com/modelscope/modelscope), [vllm](https://github.com/vllm-project/vllm).


## Follow Us

<div align=center>
<p>Scan the QR code below to follow our official account: Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

`<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" />``</a><br />`This work is licensed under a `<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">`Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License`</a>`.

*Note: Default use of CC 4.0 license, but other licenses can be selected based on the specifics of your project.*