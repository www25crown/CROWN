## The model accuracy of CROWN according to different GNN models in MIND-small
|         | Baseline(AVG) | GCN     | gain  | LightGCN | gain  | GraphSAGE | gain  |
| :-------: | :-------------: | :-------: | :-----: | :--------: | :-----: | :---------: | :-----: |
| AUC     |    0.6720     | 0.6813  | 1.38% |  0.6819  | 1.47% |  0.6823   | 1.53% |
| MRR     |    0.3194     | 0.3348  | 4.82% |  0.3350  | 4.88% |  0.3354   | 5.01% |
| nDCG@5  |    0.3550     | 0.3691  | 3.97% |  0.3705  | 4.36% |  0.3697   | 4.14% |
| nDCG@10 |    0.4174     | 0.4309  | 3.23% |  0.4289  | 2.76% |  0.4293   | 2.85% |

## References
[4] Kipf, T. N., et al. "Semi-Supervised Classification with Graph Convolutional Networks." In Proceedings of the ICLR 2016.

[5] He, X., et al. "Lightgcn: Simplifying and powering graph convolution network for recommendation." In Proceedings of the SIGIR 2020.

[6] Hamilton, W., et al. "Inductive representation learning on large graphs." In Proceedings of the NeurIPS 2017.
