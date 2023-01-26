# MARS


Code and datasets for the ICLR2023 paper "[Multimodal Analogical Reasoning over Knowledge Graphs](https://arxiv.org/pdf/2210.00312.pdf)"

## Quick links
* [MARS](#mars)
    * [Overview](#overview)
    * [Requirements](#requirements)
    * [Data Preparation](#data-collection-and-preprocessing)
    * [Evaluate on Benchmark Mehods](#evaluate-on-benchmark-mehods)
        * [Multimodal Knowledge Representation Methods](#multimodal-knowledge-representation-methods)
        * [Transformer-based Methods](#transformer-based-methods)
    * [Citation](#citation)


## Overview

In this work, we propose a new task of multimodal analogical reasoning over knowledge graph. A overview of the Multimodal Analogical Reasoning task can be seen as follows:

<div align=center>
<img src="resource/task.png" width="75%" height="75%" />
</div>

We provide a knowledge graph
to support and further divide the task into single and blended patterns. Note that the relation marked
by dashed arrows ($\dashrightarrow$) and the text around parentheses under images are only for annotation and
not provided in the input.

## Requirements

```setup
pip install -r requirements.txt
```

## Data Collection and Preprocessing

To support the multimodal analogical reasoning task, we collect a multimodal knowledge graph dataset MarKG and a Multimodal Analogical ReaSoning dataset MARS. A visual outline of the data collection as shown in following figure:

<div align=center>
<img src="resource/flowchart.png" width="75%" height="75%" />
</div>

We collect the datasets follow below steps:
1. Collect Analogy Entities and Relations
2. Link to Wikidata and Retrieve Neighbors
3. Acquire and Validate Images
4. Sample Analogical Reasoning Data

The statistics of the two datasets are shown in following figures:

<div align=center>
<img src="resource/MARS.png" width="75%" height="75%" />
</div>


<div align=center>
<img src="resource/MarKG.png" width="75%" height="75%" />
</div>

We put the text data under `MarT/dataset/`, and the image data can be downloaded through this [link](https://pan.baidu.com/s/1WZvpnTe8m0m-976xRrH90g) with extraction code (7hoc) and placed on `MarT/dataset/MARS/images`.

The expected structure of files is:

```
Multimodal Analogical Reasoning over Knowledge Graph
 |-- M-KGE	# multimodal knowledge representation methods
 |    |-- IKRL_TransAE   
 |    |-- RSME
 |-- MarT
 |    |-- data          # data process functions
 |    |-- dataset
 |    |    |-- MarKG    # knowledge graph data
 |    |    |-- MARS     # analogical reasoning data
 |    |-- lit_models    # pytorch_lightning models
 |    |-- models        # source code of models
 |    |-- scripts       # running scripts
 |    |-- tools         # tool function
 |    |-- main.py       # main function
 |-- resources   # image resources
 |-- requirements.txt
 |-- README.md

```

## Evaluate on Benchmark Mehods

We select some baseline methods to establish the initial benchmark results on MARS, including multimodal knowledge representation methods (IKRL, TransAE, RSME), pre-trained vision-language models (VisualBERT, ViLBERT, ViLT, FLAVA) and a multimodal knowledge graph completion method (MKGformer).

<div align=center>
<img src="resource/model.png" width="75%" height="75%" />
</div>

In addition, we follow the structure-mapping theory to regard the Abudction-Mapping-Induction as explicit pipline steps for multimodal knowledge representation methods. As for transformer-based methods, we further propose MarT, a novel framework that implicitly combines these three steps to accomplish the multimodal analogical reasoning task end-to-end, which can avoid error propagation during analogical reasoning. The overview of the baseline methods can be seen in above figure.

### Multimodal Knowledge Representation Methods
#### 1. [IKRL](https://github.com/thunlp/IKRL)

We reproduce the IKRL models via TransAE framework, to evaluate on IKRL, running following code:
```bash
cd M-KGE/IKRL_TransAE
python IKRL.py
```

You can choose pre-train/fine-tune and TransE/ANALOGY  by modifing `finetune` and `analogy` parameters in `IKRL.py`, respectively.


#### 2. [TransAE](https://github.com/ksolaiman/TransAE)

To evaluate on IKRL, running following code:
```bash
cd M-KGE/IKRL_TransAE
python TransAE.py
```

You can choose pre-train/fine-tune and TransE/ANALOGY  by modifing `finetune` and `analogy` parameters in `TransAE.py`, respectively.

#### 3. [RSME](https://github.com/wangmengsd/RSME)
We only provide part of the data for RSME. To evaluate on RSME, you need to generate the full data by following scripts:
```bash
cd M-KGE/RSME
python image_encoder.py  # -> analogy_vit_best_img_vec.pickle
python utils.py          # -> img_vec_id_analogy_vit.pickle
```

Firstly, pre-train the models over MarKG:
```bash
bash run.sh
```
Then modify the `--checkpoint` parameter and fine-tune the models on MARS:
```bash
bash run_finetune.sh
```

More training details about the above models can refer to their [offical repositories](https://github.com/wangmengsd/RSME).

### Transformer-based Methods

We leverage the MarT framework for transformer-based models. MarT contains two steps: pre-train and fine-tune. 

To train the models fast, we encode the image data in advance with this script (Note that the size of the encoded data is about 7GB):
```bash
cd MarT
python tools/encode_images_data.py
```

Taking MKGformer as an example, first pre-train the model via following script:
```bash
bash scripts/run_pretrain_mkgformer.sh
```

After pre-training, fine-tune the model via following script:
```bash
bash scripts/run_finetune_mkgformer.sh
```

## Citation

If you use or extend our work, please cite the paper as follows:

```bibtex
@article{DBLP:journals/corr/abs-2210-00312,
  author    = {Ningyu Zhang and
               Lei Li and
               Xiang Chen and
               Xiaozhuan Liang and
               Shumin Deng and
               Huajun Chen},
  title     = {Multimodal Analogical Reasoning over Knowledge Graphs},
  journal   = {CoRR},
  volume    = {abs/2210.00312},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.00312},
  doi       = {10.48550/arXiv.2210.00312},
  eprinttype = {arXiv},
  eprint    = {2210.00312},
  timestamp = {Tue, 27 Dec 2022 08:16:14 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-00312.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}
```
