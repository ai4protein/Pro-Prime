# Pro-Prime

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ai4protein/Pro-Prime/"><img width="200px" height="auto" src="https://github.com/ai4protein/Pro-Prime/blob/main/band.png"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
[![GitHub license](https://img.shields.io/github/license/ai4protein/Pro-Prime)](https://github.com/ai4protein/Pro-Prime/blob/main/LICENSE)

Updated on 2024.07.24

## Introduction

This repository provides the official implementation of Prime (Protein language model for Intelligent Masked pretraining and Environment (temperature) prediction).

Key feature:
- Zero-shot mutant effect prediction.
- OGT Prediction

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/ai4protein/Pro-Prime) 

## Details

### What is Pro-Prime?
Pro-Prime, a novel protein language model, has been developed for predicting the Optimal Growth Temperature (OGT) and enabling zero-shot prediction of protein thermostability and activity. This novel approach leverages temperature-guided language modeling.
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ai4protein/Pro-Prime/blob/main/model.png"></a>
</div>


## Use of PRIME

**Main Requirements**  
biopython==1.81
torch (2.4)

**Installation**
```bash
pip install -r requirements.txt
```
## ProteinGym Scores can be downloaded in
https://drive.google.com/file/d/1AEpK3TmgFNszZXJQWwRPkHUugrdHrTgk/view?usp=sharing

## 🚀 Run Notebooks
<!-- - Zero-shot mutant effect prediction, see in this [notebook](/notebooks/zero-shot-mutant-effect-prediction.ipynb). -->
- Run ProtienGym Benchmark or Zero-shot mutant Effect Prediction, see in this [notebook](/notebooks/run_proteingym.ipynb).
- OGT prediction, see in this [notebook](/notebooks/predict_ogt.ipynb).
- Tm prediction, see in this [notebook](/notebooks/predict_TM.ipynb).
- Topt prediction, see in this [notebook](/notebooks/predict_TOPT.ipynb).

<!-- ## Supervised fine-tuning for mutant fitness learning
See sft/sft_mutant.sh -->

## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:tpan1039@gmail.com)

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
