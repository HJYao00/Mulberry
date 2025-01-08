<div align="center">

<h1> Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search </h1>

<h5 align="center"> 

<a href='https://arxiv.org/abs/2412.18319'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<!--<a href='https://huggingface.co/collections/HuanjinYao/denseconnector-66500e173fc8c9f05dc98dea'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/700000183)
<a href='https://huggingface.co/spaces/HuanjinYao/DenseConnector-v1.5-8B'><img src='https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg'></a>-->


[Huanjin Yao](https://scholar.google.com/citations?user=pDtsCBQAAAAJ&hl=zh-CN)<sup>2,3*</sup>,
[Jiaxing Huang](https://jxhuang0508.github.io/)<sup>1*✉️</sup>,
[Wenhao Wu](https://whwu95.github.io/)<sup>3</sup>,
[Jingyi Zhang]()<sup>1</sup>,
[Yibo Wang]()<sup>2</sup>,
[Shunyu Liu]()<sup>1</sup>,
[Yingjie Wang]()<sup>1</sup>,

[Yuxin Song]()<sup>3</sup>,
[Haocheng Feng]()<sup>3</sup>,
[Li Shen]()<sup>4</sup>,
[Dacheng Tao]()<sup>1</sup>


<sup>1</sup>[Nanyang Technological University](https://www.ntu.edu.sg/), <sup>2</sup>[Tsinghua University](https://www.tsinghua.edu.cn/en/), <sup>3</sup>[Baidu](https://vis.baidu.com/#/), <sup>4</sup>[SYSU](https://www.sysu.edu.cn/sysuen/)

<sup>*</sup>Equal Contribution,       <sup>✉️</sup>Corresponding Author

</h5>
</div>


## 📰 News
- [x] **`Dec 24, 2024.`** We release our paper in [arxiv](https://arxiv.org/abs/2412.18319).


## 📊 Results

We conduct extensive experiments with four powerful baseline models, including [LLaVA-Next-8b](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf), [LLaMA-3.2-Vision-11B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct). The **Main Results** comparing the Mulberry models with other state-of-the-art models across several popular benchmarks are shown in the figure below.

<div align=center>
<img width="650" alt="image" src="figure/main_results.png">
</div>

## 👀 Quantitative Results
Mulberry creates rich, explicit and well-defined reasoning steps with comprehensive understanding, ultimately arriving at the correct answer.
<div align=center>
<img width="700" alt="image" src="figure/qualitative_results_reasoning.png">
</div>

## 🖇️ Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{yao2024mulberry,
  title={Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search},
  author={Yao, Huanjin and Huang, Jiaxing and Wu, Wenhao and Zhang, Jingyi and Wang, Yibo and Liu, Shunyu and Wang, Yingjie and Song, Yuxin and Feng, Haocheng and Shen, Li and Tao, Dacheng},
  journal={arXiv preprint arXiv:2412.18319},
  year={2024}
}
```

## 👍 Acknowledgment
Our work is primarily based on the following codebases. We are sincerely grateful for their work.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use llama-factory to fine-tune Mulberry Models.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use VLMEvalKit for evaluation.

