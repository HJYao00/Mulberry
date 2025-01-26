<div align="center">

<h1> <img src="figure/mulberry.png" style="vertical-align: -10px;" :height="50px" width="50px"> Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.


<h5 align="center"> 

<a href='https://arxiv.org/abs/2412.18319'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/HuanjinYao/Mulberry_llava_8b'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
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


## News
- [x] **`Jan 14, 2025.`** We release the [instructions](https://github.com/HJYao00/Mulberry?tab=readme-ov-file#evaluation) and [code](https://github.com/HJYao00/Mulberry/tree/main/evaluation) for evaluating Mulberry-LLaVA-8B on different benchmarks through the VLMEvalKit tool.
- [x] **`Jan 08, 2025.`** We release the **CoMCTS code** for searching step-by-step reasoning and reflection data, along with the [**Mulberry-LLaVA-8B**](https://huggingface.co/HuanjinYao/Mulberry_llava_8b) model and its **reasoning inference code**.
- [x] **`Dec 24, 2024.`** We release our paper in [arxiv](https://arxiv.org/abs/2412.18319).

## Reasoning Inference
We provide the inference code for running Mulberry models, which can output detailed step-by-step reasoning.

```bash
python infer.py \
--model 'Mulberry_llava_8b' \
--model_path 'HuanjinYao/Mulberry_llava_8b' \
--question 'Question: <Your_Question>' \
--img_path '<Your_Img_Path>' 
```


<details>
<summary>You can also run the following command if you only require the final answer.</summary>

```bash
python infer.py \
--model 'Mulberry_llava_8b' \
--model_path 'HuanjinYao/Mulberry_llava_8b' \
--question 'Question: <Your_Question>' \
--img_path '<Your_Img_Path>' \
--only_output_final_answer
```

</details>

## Data Constrution with CoMCTS
We release **CoMCTS Code** for generating reasoning and reflection data, which leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection.

Please refer [here](https://github.com/HJYao00/Mulberry/tree/main/comcts) for more details.

## Training
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune the Mulberry models. We provide the training instructions and configs here.

First, install LLaMA-Factory according to the [official_instruction](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation).

Then, refer [here](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) and update the following customized dataset into `dataset_info.json` in LLaMA-Factory.
```bash
"mulberry": {
    "file_name": "./mulberry_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
```

Finally, you can use the following command to train the models.
```bash
llamafactory-cli train examples/train_full/mulberry_llava_8b_full_sft.yaml
```

## Evaluation
We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate the Mulberry models on different benchmarks. We provide the evaluation instructions and key code here.

First, you need to install VLMEvalKit according to the [official instructions](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md).

Next, replace the `llava.py` file in `VLMEvalKit-main/vlmeval/vlm/llava/` with the `llava.py` file we provide [here](https://github.com/HJYao00/Mulberry/tree/main/evaluation).

Finally, you can use the following command to perform the evaluation.
```bash
python run.py --data MathVista_MINI --model llava_next_llama3 --verbose
```


## Main Results

We conduct extensive experiments with four powerful baseline models, including [LLaVA-Next-8b](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf), [LLaMA-3.2-Vision-11B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct). The **Main Results** comparing the Mulberry models with other state-of-the-art models across several popular benchmarks are shown in the figure below.

<div align=center>
<img width="650" alt="image" src="figure/main_results.png">
</div>



## Quantitative Results
Mulberry creates rich, explicit and well-defined reasoning steps with comprehensive understanding, ultimately arriving at the correct answer.
<div align=center>
<img width="700" alt="image" src="figure/qualitative_results_reasoning.png">
</div>

## Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{yao2024mulberry,
  title={Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search},
  author={Yao, Huanjin and Huang, Jiaxing and Wu, Wenhao and Zhang, Jingyi and Wang, Yibo and Liu, Shunyu and Wang, Yingjie and Song, Yuxin and Feng, Haocheng and Shen, Li and Tao, Dacheng},
  journal={arXiv preprint arXiv:2412.18319},
  year={2024}
}
```


## Acknowledgment
Our work is primarily based on the following codebases. We are sincerely grateful for their work.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use llama-factory to fine-tune Mulberry Models.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use VLMEvalKit for evaluation.

## Limitations
Mulberry is a preliminary exploration work in o1-like MLLM, leveraging Collective Monte Carlo Tree Search to enable effective and efficient reasoning-path searching and learning. 
CoMCTS leverages collective knowledge to significantly improve the search success rate and efficiency of reasoning path searches.
By training on the reasoning data generated through CoMCTS, Mulberry has gained step-by-step reasoning capabilities, leading to a substantial improvement in overall performance.
Nevertheless, certain limitations must be acknowledged.

Hallucinations in intermediate steps: Hallucinations are still prevalent in MLLMs, whether in closed or open-source models.
For instance, the models may generate obvious errors in intermediate reasoning steps yet still arrive at the correct final answer in CoMCTS.
Therefore, although we incorporated multiple models to better detect errors, some errors still persist in the intermediate steps because ensuring the correctness of all intermediate steps often requires human checks, which is extremely costly and unaffordable for us.

Error localization: 
During our experiments, we observed that models struggle to detect their own errors. To address this, CoMCTS employs multiple models to cross-check each other's errors.
However, our findings also revealed that smaller models often fail to generate effective detection responses, while larger models occasionally exhibit inaccurate error localization.
Thus, inaccurate localization may impact the efficiency of the search and we recommend using larger models for error localization or exploring better prompts to enable smaller models to localize errors more accurately.

