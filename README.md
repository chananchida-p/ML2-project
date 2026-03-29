# Amazon Beauty Reviews: Topic Modelling and Brand Response Prioritisation

## Project description

This repository contains the code used for the analysis in the SSIM916 Problem Set 2.

The project investigates the following research question:

What topics in low-rated Amazon Beauty reviews signal product quality issues, and how can brands use these insights to prioritise responses?

The full methodology, results, and discussion are presented in the accompanying report. This repository focuses on providing the implementation needed to reproduce the analysis.

## Link to report

The report explains:

* the motivation for focusing on low-rated reviews
* the comparison between LDA and BERTopic
* the role of emotion classification in identifying complaint severity
* the development of the prioritisation framework

This repository supports the **Replication section** of the report by providing the full code pipeline.

## Dataset

The dataset is accessed directly from Hugging Face:

jhan21/amazon-beauty-reviews-dataset

No manual download is required. The script loads the dataset automatically.

## Method summary (brief)

The implementation follows the same pipeline described in the report:

* filtering to 1–2 star reviews
* TF-IDF and count vectorisation
* LDA with coherence-based K selection
* BERTopic modelling using embeddings, UMAP, and HDBSCAN
* emotion classification using a transformer model
* construction of a prioritisation matrix based on volume and emotional severity

Further explanation of each step is provided in the report.

## How to run the code

The full analysis is executed from a single script:

```bash
python PS2_final_clean.py
```

Before running, install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn gensim bertopic sentence-transformers transformers datasets umap-learn hdbscan
```

If cloning from GitHub:

```bash
git clone https://github.com/chananchida-p/ML2-project.git
cd ML2-project
python PS2_final_clean.py
```

## Important notes

* Only `PS2_final_clean.py` is required to run the project
* The script includes the complete pipeline from data loading to final outputs
* The first run may take several minutes due to model and dataset loading
* Emotion classification is performed on CPU by default

## Outputs

Running the script generates the outputs referenced in the report, including:

* figures used in the analysis (e.g. coherence, UMAP, emotion distribution)
* topic modelling outputs
* validation samples
* prioritisation matrix

These outputs correspond to the figures and tables discussed in the report.

## Reproducibility

A fixed random seed is used in the code to improve consistency across runs.
All results presented in the report are generated from this pipeline.

## Project structure

```text
ML2-project/
├── PS2_final_clean.py
├── README.md
└── generated outputs
```

## Author
Chananchida Pattanichanon
MSc Business Analytics
# ML2-project