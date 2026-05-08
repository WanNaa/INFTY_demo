# Citation

If INFTY or its included algorithms are useful in your work, please cite the corresponding software and method papers.

## Software citation

For the toolkit, use a software citation similar to:

```bibtex
@software{infty_toolkit,
  title        = {INFTY: An Optimization Toolkit to Support Continual AI},
  author       = {INFTY contributors},
  year         = {2026},
  url          = {https://github.com/WanNaa/INFTY_demo},
  note         = {Research software, alpha release}
}
```

Replace the year, version, DOI, or official repository URL if a formal release is created.

## Algorithm papers

INFTY currently exposes three optimizer families under `infty.optim`. The papers below are grouped by family, and each family is ordered by year from newest to oldest.

### Geometry reshaping

| Year | Supported optimizer API | Paper |
| --- | --- | --- |
| 2025 | `C_Flat` with `strategy="plus"` | [*C-Flat++: Towards a More Efficient and Powerful Framework for Continual Learning.*](https://arxiv.org/abs/2508.18860) |
| 2024 | `C_Flat` with `strategy="basic"` | [*Make Continual Learning Stronger via C-Flat.*](https://arxiv.org/abs/2404.00986v2) |
| 2023 | `GAM` | [*Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization.*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Gradient_Norm_Aware_Minimization_Seeks_First-Order_Flatness_and_Improves_Generalization_CVPR_2023_paper.html) |
| 2022 | `LookSAM` | [*Towards Efficient and Scalable Sharpness-Aware Minimization.*](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Efficient_and_Scalable_Sharpness-Aware_Minimization_CVPR_2022_paper.html) |
| 2022 | `GSAM` | [*Surrogate Gap Minimization Improves Sharpness-Aware Training.*](https://openreview.net/forum?id=edONMAnhLu-) |
| 2021 | `SAM` | [*Sharpness-Aware Minimization for Efficiently Improving Generalization.*](https://openreview.net/forum?id=6Tm1mposlrM) |

### Zeroth-order updates

| Year | Supported optimizer API | Paper |
| --- | --- | --- |
| 2025 | `ZeroFlow` | [*Zeroflow: Overcoming Catastrophic Forgetting Is Easier Than You Think.*](https://arxiv.org/abs/2501.01045) |
| 2023 | `ZeroFlow` with `inftyopt` in `zo_sgd`, `zo_adam`, `zo_sgd_sign`, `zo_adam_sign`, `zo_sgd_conserve`, or `zo_adam_conserve` | [*Fine-Tuning Language Models with Just Forward Passes.*](https://arxiv.org/abs/2305.17333) |
| 2022 | `ZeroFlow` with `inftyopt="forward_grad"` | [*Gradients without Backpropagation.*](https://arxiv.org/abs/2202.08587) |

### Gradient filtering

| Year | Supported optimizer API | Paper |
| --- | --- | --- |
| 2024 | `UniGrad_FS` | [*UniGrad-FS: Unified Gradient Projection With Flatter Sharpness for Continual Learning.*](https://ieeexplore.ieee.org/abstract/document/10636267) |
| 2021 | `CAGrad` | [*Conflict-Averse Gradient Descent for Multi-Task Learning.*](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) |
| 2021 | `GradVac` | [*Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models.*](https://openreview.net/forum?id=F1vEjWK-lH_) |
| 2020 | `PCGrad` | [*Gradient Surgery for Multi-Task Learning.*](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) |
| 2019 | `OGD` | [*Orthogonal Gradient Descent for Continual Learning.*](https://arxiv.org/abs/1910.07104) |

For publications, cite the specific paper corresponding to the optimizer class or execution path used in your experiments.

## Reproducibility note

When citing INFTY in experiments, report:

- INFTY version or commit hash;
- optimizer name;
- optimizer `args` dictionary;
- base optimizer and hyperparameters;
- PyTorch version;
- benchmark and task order.
