# 5g-traffic-text-intrusion-dataset
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A public 5G network traffic dataset, pre-processed and formatted as text sequences. It is specifically designed for training and evaluating Large Language Models (LLMs), Transformers, and other NLP models for intrusion detection research.

This dataset was created and used in the research published in the paper(s) listed below. It is derived from a feature-rich dataset and has been converted into a string format, making it ideal for fine-tuning language models on the specialized domain of network security. We kindly ask that you cite our work if you use this data.

## Table of Contents
- [Related Publications](#related-publications)
- [Dataset Description](#dataset-description)
- [Data Format](#data-format)
- [Getting Started](#getting-started)
- [File Structure](#file-structure)
- [How to Cite](#how-to-cite)
- [License](#license)
  
## Related Publications

The primary way to give credit for this dataset is by citing the associated peer-reviewed papers, particularly the primary paper for which this data was formatted:

1.  **Sheikhi, S., Kostakos, P., & Pirttikangas, S. (2024).** *Effective Anomaly Detection in 5G Networks via Transformer-Based Models and Contrastive Learning.* In 2024 8th Cyber Security in Networking Conference (CSNet) (pp. 38-43).
2.  **Sheikhi, S., & Kostakos, P. (2024).** *Advancing Security in 5G Core Networks through Unsupervised Federated Time Series Modeling.*
3.  **Sheikhi, S., & Kostakos, P. (2023).** *DDoS attack detection using unsupervised federated learning for 5G networks and beyond.* In 2023 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit) (pp. 442-447).

## Dataset Description

This dataset contains detailed packet-level features from a simulated 5G core network, formatted specifically for training and fine-tuning Large Language Models (LLMs) and other Natural Language Processing (NLP) models. The numerical features from the original traffic data have been converted into textual descriptions, making them suitable for tokenization and processing by Transformer architectures. The primary goal is to provide a ready-to-use resource for advancing research in 5G security using state-of-the-art language models.

* **Network Environment:** Simulated using Open5GS and UERANSIM.
* **Key Scenarios Covered:**
    * Normal user traffic
    * **Simulated Anomalies:**
        * Distributed Denial of Service (DDoS) attacks
        * Distributed Port-Scan Attacks

## Data Format

Unlike traditional tabular datasets, each row in the CSV files consists of two columns: `text` and `label`. The `text` column contains a single string concatenating all feature names and their corresponding values for a given traffic flow. This format is ideal for language model training.

**Example Record:**
```
text: "Network traffic features: ip.len: 84.0, ip.flags.df: 1.0, ip.flags.mf: 0.0, tcp.port: 80.0, tcp.window_size: 0.0, tcp.ack_raw: 1266782107.0, ip.fragment: 0.0, ..."
label: 0
```
The `label` is an integer where **0** represents normal traffic and **1** represents an anomaly/attack.

## Getting Started

This dataset is designed to be used with NLP libraries like Hugging Face Transformers for training and fine-tuning models. The following example shows how to load the data.

**Prerequisites:**
* Python 3.8+
* Pandas
* PyTorch or TensorFlow
* Hugging Face `transformers` library

**Example Usage:**
```python
import pandas as pd
from datasets import Dataset

# Load the training data
df_train1 = pd.read_csv('data/Train_subset_1.csv')
df_train2 = pd.read_csv('data/Train_subset_2.csv')
df_train = pd.concat([df_train1, df_train2], ignore_index=True)

# For easy use with the Hugging Face ecosystem, convert the pandas DataFrame
# to a Dataset object. This is a recommended step for training.
hf_dataset = Dataset.from_pandas(df_train)

print("Dataset loaded and converted successfully!")
print(hf_dataset)
print("\nExample record:")
print(hf_dataset[0])

# From here, you can proceed with tokenization and model training.
```

## File Structure

The data is provided in multiple CSV files located within the `data/` directory of this repository:
* `Train_subset_1.csv` & `Train_subset_2.csv`: These contain training data formatted for Transformers. They can be used separately or combined.
* `Test_Data.csv`: A unified test set for evaluating the final model.

The repository is structured as follows:
```
.
├── data/
│   ├── Test_Data.csv
│   ├── Train_subset_1.csv
│   └── Train_subset_2.csv
├── .gitignore
├── LICENSE
└── README.md
```

## How to Cite

Please cite the relevant papers listed in the [Related Publications](#related-publications) section. For your convenience, here is the BibTeX entry for the primary paper associated with this dataset:

```bibtex
@inproceedings{sheikhi2024effective,
  title={Effective Anomaly Detection in 5G Networks via Transformer-Based Models and Contrastive Learning},
  author={Sheikhi, Saeid and Kostakos, Panos and Pirttikangas, Susanna},
  booktitle={2024 8th Cyber Security in Networking Conference (CSNet)},
  pages={38--43},
  year={2024},
  organization={IEEE}
}
```

## License

This dataset is licensed under the **MIT License**. See the `LICENSE` file for more details.
