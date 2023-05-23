随着人工智能技术的飞速发展，各种大模型层出不穷，为解决复杂问题提供了强大的支持。这些大型模型在各个领域取得了显著的成功，如计算机视觉、自然语言处理、语音识别等。为此，本文旨在整理目前所有的大语言模型相关内容，例如：大模型论文（LLaMA、Alpaca、Vicuna、GLM、Bard、PaLM1、PaLM2 等）、大模型Demo、大模型测试数据集、大模型评估方法等。

此项目长期不定时更新，欢迎watch和fork！不过给个star⭐就更好了❤️。

知乎地址：[**ShuYini**](https://www.zhihu.com/people/wangjini521/activities)

微信公众号: [**AINLPer**（每日更新，欢迎关注）](https://mp.weixin.qq.com/s?__biz=MzUzOTgwNDMzOQ==&mid=2247487079&idx=1&sn=4aa0c38c7701148f28f67bc66a291b00&chksm=fac399bbcdb410ad4517460b96a071c08c3854d67d1beafa4caa424e9c12791dc1955be1f56e&token=802874842&lang=zh_CN#rd)

### Google系列 

| **<span style="display:inline-block;width:75px">模型名称</span>** | **<span style="display:inline-block;width:70px">发布时间</span>** | **模型概述**                                                 | **<span style="display:inline-block;width:90px">    论文原文    </span>** | **<span style="display:inline-block;width:60px">代码</span>** | **Demo**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PaLM**                                                     | 2022年                                                       | PaLM: Scaling Language Modeling with Pathways<br /><br />PaLM由Google于2022年4月份发布，它是一种使用 Pathways 系统训练的 5400 亿参数、仅密集解码器的 Transformer 模型，它使我们能够跨多个 TPU v4 Pod 高效地训练单个模型。 在数百个语言理解和生成任务上评估了 PaLM，发现它在大多数任务中都实现了最先进的小样本性能，在许多情况下都有很大的优势。 |       [[Paper](https://arxiv.org/pdf/2204.02311.pdf)]        | [[Code](https://github.com/lucidrains/PaLM-pytorch)]         | /                                                            |
| **mT5**                                                      | 2020年                                                       | mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer<br /><br />mT5是一种多语言文本到文本转换模型，是Google Brain团队于2020年提出的。**mT5是T5（Text-to-Text Transfer Transformer）模型的扩展版本**，旨在支持多种语言的自然语言处理任务。 |  [[Paper](https://aclanthology.org/2021.naacl-main.41.pdf)]  | [[Code](https://github.com/google-research/multilingual-t5)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/mt5)] |
| **T5**                                                       | 2019年                                                       | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer<br /><br />T5是谷歌于2019年推出的一种基于深度学习的自然语言处理模型。它是一种预训练的语言模型，可以用于各种NLP任务，如机器翻译、文本摘要、问答系统等。T5模型的核心是基于Transformer架构的神经网络，**由多个编码器和解码器组成，每个编码器和解码器都包含多个层**。目前有：T5-small、T5-base、T5-large、T5-3b、T5-11B等5个子版本。 |       [[Paper](https://arxiv.org/pdf/1910.10683.pdf)]        | [[Code](https://github.com/google-research/text-to-text-transfer-transformer)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/t5Demo)] |


### 鹏程实验室

| 模型名称    | <span style="display:inline-block;width:70px">发布时间</span> | 模型概述                                                     | <span style="display:inline-block;width:75px">论文原文</span> | <span style="display:inline-block;width:60px">代码</span>    | Demo                                                         |
| ----------- | ------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PanGu-α** | 2021年                                                       | PANGU-α: LARGE-SCALE AUTOREGRESSIVE PRETRAINED CHINESE LANGUAGE MODELS WITH AUTO-PARALLEL COMPUTATION<br /><br />PanGu-α模型于2021年4月份发布，是业界**首个2000亿参数以中文为核心的预训练生成语言模型**，目前开源了两个版本：鹏程·PanGu-α和鹏程PanGu-α增强版，并支持NPU和GPU两个版本，支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备较强的少样本学习的能力。目前**支持2.6B、13B模型下载**。之前经常听华为宣传盘古大模型，实际上该模型是由鹏城实验室、华为MindSpore、华为诺亚方舟实验室和北京大学等相关单位联合开发的。 |  [[Paper](https://aclanthology.org/2021.naacl-main.41.pdf)]  | [[Code](https://github.com/google-research/multilingual-t5)] | [[Demo](https://huggingface.co/docs/transformers/model_doc/mt5)] |
|             |                                                              |                                                              |                                                              |                                                              |                                                              |



### 清华大学

| 模型名称     | <span style="display:inline-block;width:70px">发布时间</span> | 模型概述                                                     | <span style="display:inline-block;width:75px">论文原文</span> | <span style="display:inline-block;width:60px">代码</span> | Demo                                                        |
| ------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------: | --------------------------------------------------------- | ----------------------------------------------------------- |
| **GLM-130B** | 2023年                                                       | GLM-130B: An Open Bilingual Pre-trained Model<br /><br />GLM-130B，一种具有 1300 亿个参数的双语（英文和中文）预训练语言模型。在相关基准测试中始终显着优于最大的中文语言模型 ERNIE TITAN 3.0 260B。允许在 4×RTX 3090 (24G) 或 8×RTX 2080 Ti (11G) GPU 上进行有效推理。 |         [[Paper](https://arxiv.org/abs/2210.02414)]          | [[Code](https://github.com/THUDM/GLM-130B)]               | [[Demo](https://huggingface.co/spaces/THUDM/GLM-130B)]      |
| **GLM-6B**   | 2023年                                                       | GLM: General Language Model Pretraining with Autoregressive Blank Infilling<br /><br />ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低**只需 6GB 显存**）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。 |         [[Paper](https://arxiv.org/abs/2103.10360)]          | [[Code](https://github.com/THUDM/ChatGLM-6B)]             | [[部署](https://mp.weixin.qq.com/s/2t1B5ApHrWAE5Qegy8a8BA)] |



### 待整理

Pengi: An Audio Language Model for Audio Tasks（https://arxiv.org/abs/2305.11834）



