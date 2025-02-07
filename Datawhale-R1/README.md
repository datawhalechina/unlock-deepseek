# Datawhale-R1 复现文件

对应文章：[DeepSeek R1 Zero中文复现教程来了！
](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247700308&idx=1&sn=aa6324d30cc6d054c1dbb238b013b9b5&chksm=e98d1841220df3bd906ebe92682d4ff32dfa2fbd382fc714c8c457631de484068775c1c26846&mpshare=1&scene=2&srcid=0206JNv8uw29ECf9inhhzaxg&sharer_shareinfo=a3c5178266c37875a63b36d4a96bde91&sharer_shareinfo_first=5cd0c564850ed06c98ad41d8c06b256f#rd)

> [!CAUTION]
> 更正说明
>
> 1. 本文并不是严谨复现的 DeepSeek-R1-Zero，如果需要尽可能贴近原文，请使用 [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) 而不是 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。但是请注意，这可能需要更长的训练步长才能达到理想效果。
> 2. **请删除思考长度奖励函数 `thought_len_reward_func`，并将 `GRPOTrainer` 的 `reward_funcs` 参数的值改为 `[format_reward_func, equation_reward_func]`**，本仓库提供的代码已经修改。思考长度奖励函数由 [骆师傅](https://github.com/anine09) 对 DeepSeek-R1-Zero 训练方法的错误理解而引入，经过与其他同学的讨论，它会影响模型性能，详见下一条分析， **请立刻更新你的训练代码！**
> 3. 由于思考长度奖励函数带来的影响，请谨慎评估本文的 **训练结果解读** 部分，思考长度奖励函数可能造成模型过分追求长输出，而导致的 Token 重复问题。更长的思考长度与更深度、细致的思考没有必然的因果关系，由文章报告的结果也能看出，模型后期放弃追求思考长度奖励，而回归一个稳定的输出长度。其他大部分同学的复现报告观察到“输出长度随着问题困难度增加而自然增长”、“输出长度有先降低后增加的趋势”。思考长度奖励函数是由于在训练初期观察到输出长度不断降低，从而引入这个奖励试图对抗长度降低趋势，但是这是一个错误设计，关于文章中任何提到思考长度奖励函数的部分都应该被删除，包括：介绍、代码、举例、示意图、训练曲线。
> 4. 我们在文章中推荐大家使用的 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 项目没有这个错误。
> 5. 关于 Aha Moment 的判断大家当笑话看看就好，仅为 [骆师傅](https://github.com/anine09) 个人观点，仍需更多严谨研究验证。
