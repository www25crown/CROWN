# [WWW'25] CROWN: A Novel Approach to Comprehending Users' Preferences for Accurate Personalized News Recommendation
## The overview of CROWN
![The overview of CROWN](./assets/crown_overview.png)

<!--
## The model accuracy of CROWN according to different GNN models in MIND-small
|         | GraphSAGE | GCN    | LightGCN |   GAT   |
| ------- | --------- | ------ | -------- |-------- |
| AUC     | 0.6823    | 0.6821 | 0.6846   | 0.6857  |
| MRR     | 0.3354    | 0.3350 | 0.3380   | 0.3405  |
| nDCG@5  | 0.3697    | 0.3692 | 0.3745   | 0.3783  |
| nDCG@10 | 0.4293    | 0.4304 | 0.4364   | 0.4384  |

## Visualization of the attention weights on the disentangled intents(K=3)
![news1](./assets/sample_news_1.png)
![news2](./assets/sample_news_2.png)
![sports1](./assets/sample_sports_1.png)
![sports2](./assets/sample_sports_2.png)
![weather1](./assets/sample_weather_1.png)
![weather2](./assets/sample_weather_2.png)
-->

## Datasets
1. MIND: [**link**](https://msnews.github.io/)
2. Adressa: [**link**](https://reclab.idi.ntnu.no/dataset/)

## Dependencies
Our code runs on the Intel i7-9700k CPU with 64GB memory and NVIDIA RTX 2080 Ti GPU with 12GB, with the following packages installed:
```
python 3.8.10
torch 1.12.1
torchtext 0.12.0
torch-scatter
pandas
nltk
numpy
sklearn
```

## How to run
```
python3 main.py --news_encoder=CROWN --user_encoder=CROWN
```
