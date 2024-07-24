# Pro-Prime

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ai4protein/Prime/"><img width="200px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/band.png"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
[![GitHub license](https://img.shields.io/github/license/ai4protein/Prime)](https://github.com/ai4protein/Prime/blob/main/LICENSE)

Updated on 2023.10.20



## Introduction

This repository provides the official implementation of Prime (Protein language model for Intelligent Masked pretraining and Environment (temperature) prediction).

Key feature:
- Zero-shot mutant effect prediction.

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/ai4protein/Prime) 

## Details

### What is Prime?
Prime, a novel protein language model, has been developed for predicting the Optimal Growth Temperature (OGT) and enabling zero-shot prediction of protein thermostability and activity. This novel approach leverages temperature-guided language modeling.
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/model.png"></a>
</div>


## Use of PRIME

**Main Requirements**  
biopython==1.81
torch==2.0.1

**Installation**
```bash
pip install -r requirements.txt
```

**Download checkpoints of PRIME**

[Our official files website](https://lianglab.sjtu.edu.cn/files/Prime-2023/)

For example
```shell
wget -O checkpoints/prime_base.pt https://lianglab.sjtu.edu.cn/files/Prime-2023/Base_model/prime_base.pt
```

**Predicting Mutant Effect**
```shell
python predict.py --model_path checkpoints/prime_base.pt \
--fasta ./tm_data/fasta/O25949-7.2.fasta \
--mutant tm_data/mutant/O25949-7.2.csv \
--save O25949-7.2.prime_base.csv
```

## ProteinGym Scoring can be downloaded in
https://drive.google.com/file/d/1AEpK3TmgFNszZXJQWwRPkHUugrdHrTgk/view?usp=sharing

## 🚀 Example Notebooks

- Zero-shot mutant effect prediction, see in this [notebook](/notebooks/zero-shot-mutant-effect-prediction.ipynb).
- OGT prediction, see in this [notebook](/notebooks/predict_ogt.ipynb).
- Tm prediction, see in this [notebook](/notebooks/predict_tm.ipynb).
- Topt prediction, see in this [notebook](/notebooks/predict_topt.ipynb).

## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers) and [esm](https://github.com/facebookresearch/esm).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023,
      title={Engineering Enhanced Stability and Activity in Proteins through a Novel Temperature-Guided Language Modeling.}, 
      author={Pan Tan and Mingchen Li and Liang Zhang and Zhiqiang Hu and Liang Hong},
      year={2023},
      eprint={2304.03780},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
