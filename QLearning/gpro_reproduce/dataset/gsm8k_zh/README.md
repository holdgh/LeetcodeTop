---
license: mit
task_categories:
- question-answering
language:
- en
- zh
tags:
- math
- math-qa
- chinese-math-qa
size_categories:
- n<1K

---


# Dataset

`GSM8K_zh` is a dataset for mathematical reasoning in Chinese, question-answer pairs are translated from GSM8K (https://github.com/openai/grade-school-math/tree/master) by `GPT-3.5-Turbo` with few-shot prompting.
The dataset consists of 7473 training samples and 1319 testing samples. The former is for **supervised fine-tuning**, while the latter is for **evaluation**.

for training samples, `question_zh` and `answer_zh` are question and answer keys, respectively;
for testing samples, only the translated questions are provided (`question_zh`).



# Citation

If you find the `GSM8K_zh` dataset useful for your projects/papers, please cite the following paper.

```bibtex
@article{yu2023metamath,
  title={MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models},
  author={Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
  journal={arXiv preprint arXiv:2309.12284},
  year={2023}
}
```