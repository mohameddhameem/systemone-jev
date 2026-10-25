# SystemOne-Jev

A router that decides whether a small language model's answer is good enough, or whether the question should go to a
large one. It is a 33M cross-encoder fine-tuned from [nano-jev](https://huggingface.co/sdmlai/nano-jev) v1.0 to answer
one typed question, *"Which model should answer this question?"*, and it returns P(large model).

Two routers are published on the Hugging Face Hub:

| Router | Reads | Decides | Hub |
|---|---|---|---|
| `qp` | question and the small model's top-2 passages | before the small model runs | [systemone-jev-qp](https://huggingface.co/mohameddhameem/systemone-jev-qp) |
| `qpa` | the same, plus the small model's answer | after the small model runs | [systemone-jev-qpa](https://huggingface.co/mohameddhameem/systemone-jev-qpa) |

The small model is Qwen2.5-1.5B-Instruct on nano-jev's top 2 paragraphs. The large model is Qwen2.5-7B-Instruct on all 10.

## Results

HotpotQA SLR split, 1,481 test questions, lenient accuracy, cost in AgentTTS Table 5 units. AIQ is the area under the
accuracy-vs-cost curve. The judges are 3-seed ensembles; intervals are 95% paired bootstrap.

| Policy | Decides | AIQ |
|---|---|---|
| Random | | 0.638 |
| Small-model entropy threshold | after | 0.665 |
| Logistic-regression router | before | 0.669 |
| Logistic-regression cascade | after | 0.681 |
| **qp** | before | **0.674** (single seeds 0.672 ± 0.001) |
| **qpa** | after | **0.674** (single seeds 0.671 ± 0.002) |
| Oracle | | 0.758 |

The routers tie with the entropy threshold and with the logistic-regression baselines. None of the differences is
significant, so these results do not show an accuracy gain. The router's own cost is about zero on this scale. On
1,303 test questions that nano-jev did not see in its own training, the judges score 0.670.

The saved scores were computed in bf16 on a GPU. On a CPU in fp32 they differ by up to 7.6e-3, and 2 of the 1,481
routing decisions flip at the published threshold.

Figures: [figures/aiq.png](figures/aiq.png), [figures/cost_accuracy.png](figures/cost_accuracy.png). Full reports:
[results/report.json](results/report.json) and [results/report-clean.json](results/report-clean.json).

## Usage

```python
from systemone_jev.route import Router

router = Router.from_pretrained("mohameddhameem/systemone-jev-qp")
passages = [
    "Scott Derrickson: Scott Derrickson is an American director ...",
    "Ed Wood: Edward Davis Wood Jr. was an American filmmaker ...",
]
question = "Were Scott Derrickson and Ed Wood of the same nationality?"
p = router.p_escalate(question, passages)   # P(large model)
use_large = router.route(question, passages) == "large"
```

`qpa` also takes the small model's answer: `router.p_escalate(question, passages, answer="...")`.

## Method

The judge is nano-jev with a two-way question. It scores the pair (question, state) for each option, "small model" and
"large model", and softmaxes the two scores. The label is `large-helps`: the small model is wrong and the large one is
right. Training uses grouped-softmax cross-entropy, with the epoch picked by dev NLL and a threshold picked on the
calibration split so that F1 stays within 1% of always-large.

## Reproducing

Training starts from cached answers of the two models on each training question. This repository does not generate
those caches, so a full retrain needs them from your own pipeline. With them:

```
uv sync
uv run judge all --plan configs/system-one-slr.toml --data-dir <dir with the HotpotQA parquet files> --out runs/judges --save-model
```

The release checks are in `scripts/`:

- `check_scores.py` runs `route.py` on the test questions and compares it with the saved scores. The tolerance is 1e-4 on
  a GPU and 1e-2 on a CPU.
- `make_figures.py` redraws the two figures from `results/`.
- `push_hf.py --org <name>` stages the Hub repositories, with cards rendered from `results/` and `model-cards/`. Add
  `--push` to upload.

If your Windows setup blocks pyarrow's Parquet module, set `SYSTEMONE_PARQUET_ENGINE=fastparquet`.

## Layout

| Path | Contents |
|---|---|
| `systemone_jev/` | judge training and evaluation, metrics, `route.py` for inference |
| `configs/` | the training plan of the published routers |
| `results/` | evaluation reports and the score check |
| `figures/` | the figures in the model card |
| `model-cards/` | the Hub model card template |
| `scripts/` | score check, figures, Hub upload |
| `tests/` | `uv run pytest` |

## Licence

MIT, the same as nano-jev v1.0. The training data is HotpotQA, which is CC BY-SA 4.0. Whether share-alike terms
apply to weights trained on that data is not settled, and this project does not claim that they do not.
