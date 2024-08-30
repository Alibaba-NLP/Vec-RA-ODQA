# Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts
> Source code for paper [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://aclanthology.org/2024.findings-acl.458/)

![a](figs/overall_2.jpg)

## Python Environment
```bash
conda create -n vec python=3.9
conda activate vec
pip install -r requirements.txt
```

## Training (E.g. on TriviaQA template dataset, 1gpu)
baseline
```bash
bash triviaqa-0vec.sh
```

(Encoder frozen) +5k & +10k
```bash
bash triviaqa-10vec.sh
bash triviaqa-20vec.sh
```

(Encoder training) +5k & +10k
```bash
bash triviaqa-10vec-enc-dec.sh
bash triviaqa-20vec-enc-dec.sh
```

## Evaluate (E.g. on TriviaQA dev. template dataset, 1gpu)
In top directory of this repo., you might need to modify the model path in `src/eval_triviaqa_em_online.py `.
```python
python src/eval_triviaqa_em_online.py 
```
## Citation
```bibtex
@inproceedings{chen-etal-2024-improving-retrieval,
    title = "Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts",
    author = "Chen, Zhuo  and
      Wang, Xinyu  and
      Jiang, Yong  and
      Xie, Pengjun  and
      Huang, Fei  and
      Tu, Kewei",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.458",
    pages = "7683--7694",
    abstract = "In the era of large language models, applying techniques such as Retrieval Augmented Generation can better address Open-Domain Question-Answering problems. Due to constraints including model sizes and computing resources, the length of context is often limited, and it becomes challenging to empower the model to cover overlong contexts while answering questions from open domains. This paper proposes a general and convenient method to cover longer contexts in Open-Domain Question-Answering tasks. {\%}It leverages a small encoder language model that effectively encodes contexts, and the encoding applies cross-attention with origin inputs.It leverages a small encoder and cross-attention mechanism and effectively encodes contexts. With our method, the original language models can cover several times longer contexts while keeping the computing requirements close to the baseline. Our experiments demonstrate that after fine-tuning, there is improved performance across two held-in datasets, four held-out datasets, and also in two In Context Learning settings. Our code will be released at https://github.com/Alibaba-NLP/Vec-RA-ODQA.",
}
```


## Note
1. Be careful of the version of `transformers` and `pytest`, and please follow the requirements listed in requirements.txt. 
2. Due to networking issues, some of the models/metric in the code are loaded locally. Please adjust it into your way for loading.
