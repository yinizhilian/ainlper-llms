随着人工智能技术的飞速发展，各种大模型层出不穷，为解决复杂问题提供了强大的支持。这些大型模型在各个领域取得了显著的成功，如计算机视觉、自然语言处理、语音识别等。为此，本文旨在整理目前所有的大语言模型相关内容，例如：大模型论文（LLaMA、Alpaca、Vicuna、GLM、Bard、PaLM1、PaLM2 等）、大模型Demo、大模型测试数据集、大模型评估方法等。

此项目长期不定时更新，欢迎watch和fork！不过给个star⭐就更好了❤️。

知乎地址：[**ShuYini**](https://www.zhihu.com/people/wangjini521/activities)

微信公众号: [**AINLPer**（每日更新，欢迎关注）](https://mp.weixin.qq.com/s?__biz=MzUzOTgwNDMzOQ==&mid=2247487079&idx=1&sn=4aa0c38c7701148f28f67bc66a291b00&chksm=fac399bbcdb410ad4517460b96a071c08c3854d67d1beafa4caa424e9c12791dc1955be1f56e&token=802874842&lang=zh_CN#rd)

### Google系

| **<span style="display:inline-block;width:75px">模型名称</span>** | **<span style="display:inline-block;width:70px">发布时间</span>** | **模型概述**                                                 | **<span style="display:inline-block;width:90px">原文    </span>** | **<span style="display:inline-block;width:60px">代码</span>** | **Demo**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PaLM**                                                     | 2022年                                                       | PaLM: Scaling Language Modeling with Pathways<br /><br />PaLM由Google于2022年4月份发布，它是一种使用 Pathways 系统训练的 5400 亿参数、仅密集解码器的 Transformer 模型，它使我们能够跨多个 TPU v4 Pod 高效地训练单个模型。 在数百个语言理解和生成任务上评估了 PaLM，发现它在大多数任务中都实现了最先进的小样本性能，在许多情况下都有很大的优势。 |       [[Paper](https://arxiv.org/pdf/2204.02311.pdf)]        | [[Code](https://github.com/lucidrains/PaLM-pytorch)]         | /                                                            |
| **mT5**                                                      | 2020年                                                       | mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer<br /><br />mT5是一种多语言文本到文本转换模型，是Google Brain团队于2020年提出的。**mT5是T5（Text-to-Text Transfer Transformer）模型的扩展版本**，旨在支持多种语言的自然语言处理任务。 |  [[Paper](https://aclanthology.org/2021.naacl-main.41.pdf)]  | [[Code](https://github.com/google-research/multilingual-t5)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/mt5)] |
| **T5**                                                       | 2019年                                                       | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer<br /><br />T5是谷歌于2019年推出的一种基于深度学习的自然语言处理模型。它是一种预训练的语言模型，可以用于各种NLP任务，如机器翻译、文本摘要、问答系统等。T5模型的核心是基于Transformer架构的神经网络，**由多个编码器和解码器组成，每个编码器和解码器都包含多个层**。目前有：T5-small、T5-base、T5-large、T5-3b、T5-11B等5个子版本。 |       [[Paper](https://arxiv.org/pdf/1910.10683.pdf)]        | [[Code](https://github.com/google-research/text-to-text-transfer-transformer)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/t5Demo)] |


### 鹏程实验室

| 模型名称    | <span style="display:inline-block;width:70px">发布时间</span> | 模型概述                                                     | <span style="display:inline-block;width:75px">原文</span>  | <span style="display:inline-block;width:60px">代码</span>    | Demo                                                         |
| ----------- | ------------------------------------------------------------ | :----------------------------------------------------------- | :--------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PanGu-α** | 2021年                                                       | PANGU-α: LARGE-SCALE AUTOREGRESSIVE PRETRAINED CHINESE LANGUAGE MODELS WITH AUTO-PARALLEL COMPUTATION<br /><br />PanGu-α模型于2021年4月份发布，是业界**首个2000亿参数以中文为核心的预训练生成语言模型**，目前开源了两个版本：鹏程·PanGu-α和鹏程PanGu-α增强版，并支持NPU和GPU两个版本，支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备较强的少样本学习的能力。目前**支持2.6B、13B模型下载**。之前经常听华为宣传盘古大模型，实际上该模型是由鹏城实验室、华为MindSpore、华为诺亚方舟实验室和北京大学等相关单位联合开发的。 | [[Paper](https://aclanthology.org/2021.naacl-main.41.pdf)] | [[Code](https://github.com/google-research/multilingual-t5)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/mt5)] |
|             |                                                              |                                                              |                                                            |                                                              |                                                              |



### 清华大学

| 模型名称     | <span style="display:inline-block;width:70px">发布时间</span> | 模型概述                                                     | <span style="display:inline-block;width:75px">原文</span> | <span style="display:inline-block;width:60px">代码</span> | Demo                                                        |
| ------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | :-------------------------------------------------------: | --------------------------------------------------------- | ----------------------------------------------------------- |
| **GLM-130B** | 2023年                                                       | GLM-130B: An Open Bilingual Pre-trained Model<br /><br />GLM-130B，一种具有 1300 亿个参数的双语（英文和中文）预训练语言模型。在相关基准测试中始终显着优于最大的中文语言模型 ERNIE TITAN 3.0 260B。允许在 4×RTX 3090 (24G) 或 8×RTX 2080 Ti (11G) GPU 上进行有效推理。 |        [[Paper](https://arxiv.org/abs/2210.02414)]        | [[Code](https://github.com/THUDM/GLM-130B)]               | [[Demo](https://huggingface.co/spaces/THUDM/GLM-130B)]      |
| **GLM-6B**   | 2023年                                                       | GLM: General Language Model Pretraining with Autoregressive Blank Infilling<br /><br />ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低**只需 6GB 显存**）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。 |        [[Paper](https://arxiv.org/abs/2103.10360)]        | [[Code](https://github.com/THUDM/ChatGLM-6B)]             | [[部署](https://mp.weixin.qq.com/s/2t1B5ApHrWAE5Qegy8a8BA)] |

### Meta AI

| 模型名称  | 发布时间 | 模型概述                                                     |                      原文                       | 代码                                                | Demo |
| --------- | -------- | :----------------------------------------------------------- | :---------------------------------------------: | --------------------------------------------------- | ---- |
| **LIMA**  | 2023年   | LIMA: Less Is More for Alignment<br /><br />在没有任何RLHF的情况下，使用1000个精心筛选的提示和响应**「对LLaMA-65B进行微调得到了LIMA模型」**，实验表明该模型展现出了非常强大的性能，最后作者指出**几乎所有大型语言模型的知识都是在预训练期间学习的，仅需要有限的指导调整数据来教模型产生高质量的输出**。 | [[Paper](https://arxiv.org/pdf/2305.11206.pdf)] | /                                                   | /    |
| **LLaMA** | 2023年   | LLaMA: Open and Efficient Foundation Language Models<br /><br />LLaMA是Meta发布的一组基础语言模型，参数范围从 7B 到 65B，使用公开可用的数据集来训练最先进的模型。 LLaMA-13B 在大多数基准测试中都优于 GPT-3 (175B)，而 LLaMA-65B 可与最佳模型 Chinchilla-70B 和 PaLM-540B 竞争。 |   [[Paper](https://arxiv.org/abs/2302.13971)]   | [[Code](https://github.com/facebookresearch/llama)] | /    |



### OpenAI

| 模型名称        | 发布时间 | 模型概述                                                     |                             原文                             | 代码                                                         | Demo                                       |
| --------------- | -------- | :----------------------------------------------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------ |
| **GPT-4**       | 2023年   | GPT-4可以生成、编辑并与用户一起完成创意和技术写作任务，例如创作歌曲、编写剧本或学习用户的写作风格。GPT-4 可以接受图像作为输入并生成说明、分类和分析。GPT-4 能够处理超过 25,000 个单词的文本，允许使用长格式内容创建、扩展对话以及文档搜索和分析等用例。 |          [[Blog](https://openai.com/product/gpt-4)]          | /                                                            | [[ChatGPT-Plus](https://chat.openai.com/)] |
| **ChatGPT**     | 2022年   | ChatGPT 使用与 InstructGPT 相同的方法，使用人类反馈强化学习 (RLHF) 训练该模型，但数据收集设置略有不同。 |          [[Blog](https://openai.com/blog/chatgpt)]           | /                                                            | [[Demo](https://chat.openai.com/)]         |
| **InstructGPT** | 2021年   | 收集了所需模型行为的标记器演示数据集，用它来使用监督学习微调 GPT-3； 然后，收集模型输出排名的数据集，使用人类反馈强化学习 (RLHF) 进一步微调该监督模型，生成的模型称为 InstructGPT。 在对我们的提示分布的人工评估中，1.3B 参数 InstructGPT 模型的输出优于 175B GPT-3 的输出，尽管参数少 100 倍。 | [[Paper](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf)] | [[Code](https://github.com/openai/following-instructions-human-feedback)] | /                                          |
| **GPT-3**       | 2020年   | Language Models are Few-Shot Learners<br /><br />扩大语言模型极大地提高了与任务无关的、少样本的性能，有时甚至可以与之前最先进的微调方法相媲美。GPT-3，这是一种具有 1750 亿个参数的自回归语言模型，在少样本中测试其性能，结果显示其性能相比之前的任何非稀疏语言模型效果要好10倍。 |       [[Paper](https://arxiv.org/pdf/2005.14165.pdf)]        | [[Code](https://github.com/openai/gpt-3)]                    | /                                          |
| **GPT-2**       | 2019年   | Language Models are Unsupervised Multitask Learners<br /><br />GPT-2 是一个 1.5B 参数的 Transformer，它在零样本设置中的 8 个测试语言建模数据集中的 7 个上取得了最先进的结果，但仍然不适合 WebText。 | [[Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)] | [[Code](https://github.com/openai/gpt-2)]                    | /                                          |
| **GPT-1**       | 2018年   | Improving Language Understanding by Generative Pre-Training<br /><br />尽管大量未标记的文本语料库很丰富，但用于学习这些特定任务的标记数据却很少，这使得经过判别训练的模型难以充分执行。 我们证明，通过在不同的未标记文本语料库上对语言模型进行生成式预训练，然后对每个特定任务进行判别式微调，可以实现这些任务的巨大收益。 | [[Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)] | [[Code](https://github.com/openai/finetune-transformer-lm)]  | /                                          |



### 待整理

Pengi: An Audio Language Model for Audio Tasks（https://arxiv.org/abs/2305.11834）



