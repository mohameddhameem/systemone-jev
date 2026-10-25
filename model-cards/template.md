---
license: mit
base_model: sdmlai/nano-jev
datasets:
  - hotpotqa/hotpot_qa
language:
  - en
library_name: transformers
pipeline_tag: text-classification
tags:
  - llm-routing
  - cross-encoder
---

# SystemOne-Jev {inputs}

A router between a small and a large language model for multi-hop questions. It is a 33M MiniLM cross-encoder fine-tuned
from nano-jev v1.0. Given {inputs_text}, it returns P(large model): the probability that the large model should answer.
Send the question to the large model when that probability is at or above `threshold` in `systemone.json`, which is
chosen on validation.

- **Small model:** Qwen2.5-1.5B-Instruct on nano-jev's top-2 HotpotQA paragraphs. **Large model:** Qwen2.5-7B-Instruct on all 10.
- **Label:** the large model is needed, meaning the small one is wrong and the large one is right ({metric} scoring, tau = {tau}).
- **Seeds:** seed 0 is in the root folder; seeds 1 and 2 are in `seed1/` and `seed2/`.

## Results

HotpotQA SLR split, 1,481 test questions (178 of them were in nano-jev's own training, so the clean figure on 1,303
questions is shown too). Cost is {cost_unit}. AIQ is the area under the accuracy-vs-cost curve.

| | AIQ |
|---|---|
| Seed 0 (root folder, with its `systemone.json` threshold) | {aiq_seed0} |
| Seed 0, including its own cost ({per_call_cost} units a call) | {aiq_seed0_with_cost} |
| Three-seed ensemble, 95% CI | {aiq} {judge_ci} |
| Three-seed ensemble, 1,303 clean questions | {aiq_clean} |
| Single seeds, mean ± sd | {aiq_seeds} |
| Entropy threshold on the small model, 95% CI | {entropy_aiq} {entropy_ci} |
| {baseline_name}, 95% CI | {baseline_aiq} {baseline_ci} |
| Oracle (upper bound) | {oracle} |

Intervals are paired bootstrap over the {n_test} test questions.

Router minus baseline, 95% paired bootstrap:

- against the entropy threshold: {vs_entropy}
- against the {baseline_short}: {vs_baseline}

Neither difference is significant. The router ties with these baselines; the results do not show an accuracy gain.
Its own cost is {judge_cost} units a question on this scale.

The saved scores were computed in bf16 on a GPU. On a CPU in fp32 they differ by up to {cpu_max_diff}, and {cpu_flips} of
the {n_test} routing decisions flip at the published threshold.

![AIQ of each policy with its 95% interval](./figures/aiq.png)

![Accuracy against cost](./figures/cost_accuracy.png)

## Use

```python
# pip install git+https://github.com/mohameddhameem/systemone-jev
from systemone_jev.route import Router

router = Router.from_pretrained("{repo}")
p = router.p_escalate(question, passages{answer_arg})   # passages: ["title: text", ...], the small model's top 2
```

## Training

Grouped-softmax cross-entropy on the 4,814 training questions of the SLR split (HotpotQA dev). The label comes from
whether each model was right. lr 5e-5, batch 16, up to 3 epochs. The epoch with the lowest NLL on a 10% hold-out is kept.
`training.json` has the full record and the loss per epoch.

## Limits

Trained and tested on HotpotQA with one model pair. The threshold is chosen on validation for this pair; re-pick it on
your own validation data before using another pair or dataset.

## Licence

The weights and code are MIT-licensed, the same as nano-jev v1.0. HotpotQA, the training data, is CC BY-SA 4.0. Whether
its share-alike terms apply to weights trained on it is not settled, and this card does not claim that they do not.
