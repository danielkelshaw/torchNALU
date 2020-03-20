# torchNALU

**PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)**

<p align=center>
<img src="./media/nalu_nac_architecture.png" width=60%/>
</p>

## **Experiments:**

To reproduce the interpolation results for a static task, run:

```bash
python network.py
```

This generates a file ```results/nalu_results.csv``` which contains the relevant information.

The table below demonstrates an example of these results:

|         | Relu6     | None     | NAC        | NALU     |
| ------- | :-------: | :------: | :--------: | :------: |
| **add** | 4.65136   | 0.69667  | 0.30679    | 1.08069  |
| **sub** | 88.11118  | 17.63755 | 0.51088    | 12.23579 |
| **mul** | 48.88384  | 1.30515  | 89.22546   | 0.00434  |
| **div** | 890.01470 | 45.57086 | 7376.06001 | 48.44410 |

*This is a demonstration of the relative error compared to a random network - see paper for details.*

## **References:**

```
@misc{trask2018neural,
    title={Neural Arithmetic Logic Units},
    author={Andrew Trask and Felix Hill and Scott Reed and Jack Rae and Chris Dyer and Phil Blunsom},
    year={2018},
    eprint={1808.00508},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```