This repo is based on https://github.com/SinaLab/ArabicNER implementation. The system was developed as part of the WojoodNER Shared Task, designed to tackle both flat and nested Arabic NER tasks using advanced machine learning techniques.

## Key Features

- **BERT-based Multi-Task Learning Model**: Utilizes multiple training objectives and a multi-task loss variance penalty to enhance model accuracy.
- **Arabic Pretrained Language Models (PLMs)**: Tests and integrates various Arabic PLMs for optimal sentence encoding.
- **Comprehensive Loss Function Integration**: Includes Cross-Entropy, Dice, Tversky, and Focal loss functions to refine training outcomes.
- **High Performance**: Achieved micro-F1 scores of 0.9113 and 0.9303 in flat and nested NER tasks, respectively.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- PyTorch 2 or higher
- Transformers library
- TensorBoard for monitoring training progress

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-repository-url.git
cd your-repository-directory
pip install -r requirements.txt
```

### Usage

To run the training script, use the following command with necessary parameters:

```bash
python train.py --output_path ./output --train_path data/train.txt --val_path data/val.txt --test_path data/test.txt --bert_model aubmindlab/bert-base-arabertv2 --batch_size 32 --gpus 0
```

You can adjust the parameters based on your computational resources and requirements.

## Results

Our system has been rigorously tested on the official test set, securing high rankings among participants:
- **Sub-Task 1 (Flat NER):** 6th position with a micro-F1 score of 0.9113.
- **Sub-Task 2 (Nested NER):** 2nd position with a micro-F1 score of 0.9303.

## Citation

Please cite our work if you use this system or the dataset:

```
@inproceedings{mahdaouy-etal-2023-um6p,
    title = "{UM}6{P} {\&} {UL} at {W}ojood{NER} shared task: Improving Multi-Task Learning for Flat and Nested {A}rabic Named Entity Recognition",
    author = "El Mahdaouy, Abdelkader  and
      Lamsiyah, Salima  and
      Alami, Hamza  and
      Schommer, Christoph  and
      Berrada, Ismail",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.87",
    doi = "10.18653/v1/2023.arabicnlp-1.87",
    pages = "777--782"
}
```

