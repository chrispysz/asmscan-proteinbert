
# ASMscan-ProteinBERT
![Python Version](https://img.shields.io/badge/python-3.9-306998?logo=python) ![GitHub License](https://img.shields.io/github/license/chrispysz/asmscan-proteinbert)
## Description

This repository contains data and script used to train and evaluate ProteinBERT for binary amyloid motif classification.
If you want to test this model without the need of setting up environment from scratch please use
**[asmscan-proteinbert-run](https://github.com/chrispysz/asmscan-proteinbert-run)** Docker image.

On the other hand, for the purpose of fine-tuning ProteinBERT on your own data the authors provide some excellent examples in their [repository](https://github.com/nadavbra/protein_bert).


## Environment Setup (Python 3.9)
In case you want to reproduce the results from our paper follow the steps below:
1. Clone the repository:
   ```bash
   git clone https://github.com/chrispysz/asmscan-proteinbert.git
   ```
2. Install ProteinBERT from the original [repository](https://github.com/nadavbra/protein_bert)

3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure that your versions match the ones specified in this file.
   

## References
Original model weights and architecture:

* Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
"ProteinBERT: A universal deep-learning model of protein sequence and function" 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020<br>
https://github.com/nadavbra/protein_bert

ASMscan-ProteinBERT model is part of the [ASMscan](https://github.com/wdyrka-pwr/ASMscan) project:

* Not yet published.

ASMscan project is an extension of the ASMs analysis conducted with the [PCFG-CM](https://git.e-science.pl/wdyrka/pcfg-cm) model:

* W. Dyrka, M. Gąsior-Głogowska, M. Szefczyk, N. Szulc, "Searching for universal model of amyloid signaling motifs using probabilistic context-free grammars", *BMC Bioinformatics*, 22, 222, 2021.

* W. Dyrka, M. Pyzik, F. Coste, H. Talibart, "Estimating probabilistic context-free grammars for proteins using contact map constraints", *PeerJ*, 7, e6559, 2019.