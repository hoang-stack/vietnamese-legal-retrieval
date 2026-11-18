# Results

## Test questions (n = 788)

| System | MRR@10 (95% CI) | nDCG@10 | R@1 | R@5 | R@10 (95% CI) | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid (BM25 + fine-tuned e5, RRF) | 0.782 (0.759–0.804) | 0.824 | 0.676 | 0.915 | 0.951 (0.935–0.965) | 0.991 |
| e5-base fine-tuned | 0.739 (0.713–0.764) | 0.786 | 0.628 | 0.881 | 0.933 (0.914–0.950) | 0.990 |
| bge-m3 (zero-shot) | 0.730 (0.704–0.756) | 0.776 | 0.621 | 0.871 | 0.921 (0.901–0.939) | 0.986 |
| Hybrid + bge-reranker-v2-m3 | 0.719 (0.692–0.745) | 0.764 | 0.619 | 0.847 | 0.907 (0.887–0.926) | 0.991 |
| multilingual-e5-small (zero-shot) | 0.712 (0.685–0.737) | 0.761 | 0.591 | 0.871 | 0.914 (0.894–0.934) | 0.988 |
| BM25 | 0.681 (0.655–0.706) | 0.732 | 0.562 | 0.835 | 0.890 (0.867–0.911) | 0.980 |
| multilingual-e5-base (zero-shot) | 0.681 (0.654–0.707) | 0.736 | 0.555 | 0.834 | 0.907 (0.886–0.927) | 0.986 |
| BM25 (untuned) | 0.666 (0.640–0.692) | 0.719 | 0.542 | 0.829 | 0.883 (0.859–0.904) | 0.980 |

Paired bootstrap on MRR@10 against **Hybrid (BM25 + fine-tuned e5, RRF)**:

| vs. | Δ MRR@10 | 95% CI | p |
| --- | ---: | ---: | ---: |
| e5-base fine-tuned | +0.043 | +0.022 to +0.063 | < 0.001 |
| bge-m3 (zero-shot) | +0.053 | +0.028 to +0.076 | < 0.001 |
| Hybrid + bge-reranker-v2-m3 | +0.064 | +0.040 to +0.089 | < 0.001 |
| multilingual-e5-small (zero-shot) | +0.071 | +0.046 to +0.095 | < 0.001 |
| BM25 | +0.101 | +0.083 to +0.120 | < 0.001 |
| multilingual-e5-base (zero-shot) | +0.102 | +0.080 to +0.124 | < 0.001 |
| BM25 (untuned) | +0.116 | +0.098 to +0.136 | < 0.001 |

## Dev questions (used for every choice)

| Stage | System | MRR@10 | R@10 | R@100 |
| --- | --- | ---: | ---: | ---: |
| tokenization | BM25 (syllables, default k1/b) | 0.584 | 0.800 | 0.927 |
| tokenization | BM25 (syllables + bigrams, default k1/b) | 0.679 | 0.863 | 0.964 |
| final | multilingual-e5-small (zero-shot) | 0.692 | 0.869 | 0.976 |
| final | multilingual-e5-base (zero-shot) | 0.685 | 0.892 | 0.973 |
| final | bge-m3 (zero-shot) | 0.712 | 0.904 | 0.981 |
| finetune_epochs | e5-base fine-tuned, epoch 1 | 0.728 | 0.908 | 0.979 |
| finetune_epochs | e5-base fine-tuned, epoch 2 | 0.738 | 0.913 | 0.990 |
| finetune_epochs | e5-base fine-tuned, epoch 3 | 0.745 | 0.915 | 0.990 |
| final | e5-base fine-tuned | 0.745 | 0.915 | 0.990 |
| final | Hybrid (BM25 + fine-tuned e5, RRF) | 0.788 | 0.931 | 0.984 |
| final | Hybrid + bge-reranker-v2-m3 | 0.775 | 0.930 | 0.984 |

## GPU timings (Kaggle T4)

- bm25_index: 24 s
- multilingual-e5-small (zero-shot)_encode: 168 s
- multilingual-e5-base (zero-shot)_encode: 409 s
- bge-m3 (zero-shot)_encode: 1235 s
- e5-base fine-tuned, epoch 1_encode: 411 s
- e5-base fine-tuned, epoch 2_encode: 410 s
- e5-base fine-tuned, epoch 3_encode: 410 s
- rerank: 1293 s
