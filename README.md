# Flipkart Support Assistant — complete project implementation

This repository implements Parts 1–3 as one connected build. Run from the repository root.

## 1. Part 1 — return-risk model

```bash
python generate_orders.py
python part1/train_return_risk.py
```

The generator uses `np.random.default_rng(42)`, exactly 6,000 rows and 13 columns. The generated return rate is 22.75%. `rating_given` is missing for 13.05% overall. Missingness is **MAR conditional on the observed `payment_method`**: COD missingness is 22.83%, while non-COD missingness is 6.06%, a gap of about 16.77 percentage points. It is not MCAR because missingness depends on payment method; it is not MNAR because the missingness mask is generated from observed payment method rather than the unobserved rating value.

The baseline is intentionally weak: a most-frequent DummyClassifier has high accuracy but zero recall/F1 for returns. Logistic regression is class-weighted and its threshold is swept from 0.10–0.90. The final artifact is the tuned Random Forest pipeline, saved to `models/return_risk_model.pkl`; the Random Forest's own F1-maximizing threshold is saved separately to `models/return_risk_threshold.json`.

## 2. Part 2 — transfer learning

```bash
python part2/train_product_classifier.py
```

Fashion-MNIST is downloaded automatically. The pipeline replicates grayscale to three channels, resizes to 224×224, and applies ImageNet normalization. A pretrained ResNet-18 is frozen for feature extraction; cached 512-dimensional features are used to train the 10-class head. If validation accuracy is below 80%, `layer4` is unfrozen and fine-tuned at a lower learning rate. The final state dict is saved to `models/product_classifier.pt`, and five actual PNG samples are exported to `data/sample_images/`.

## 3. Part 3 — RAG + tools + LangGraph

First build the vector index after installing dependencies:

```bash
python -c "import part3.rag"
```

Then run retrieval evaluation:

```bash
python part3/evaluate_retrieval.py
```

The agent uses four graph nodes: intent, RAG retrieval, tool calling, and response generation, with a conditional edge from intent. `check_return_risk` loads the real Part 1 artifact and `classify_product_image` loads the real Part 2 artifact. The risk buckets are anchored to `t*_rf`: Low `< t*_rf`, Medium `t*_rf ≤ p < t*_rf+0.15`, High `≥ t*_rf+0.15`.

The response generator is deterministic/mock by default: no API key and no network call are needed once the local models/index have been built. The prompt is annotated with Specific, Short, Surround, Single, plus role prompting, and includes few-shot intent examples.

## Business interpretation

At a lower classification threshold, recall increases because more genuinely return-prone orders are flagged. The cost is lower precision and therefore more false alarms. In a support workflow this is reasonable when missing a high-risk return is more expensive than sending an additional proactive check.

For subgroup analysis, use `outputs/part1_report.json`. A concrete intervention should target the weakest subgroup rather than merely saying “collect more data”; for example, a category-specific threshold can be calibrated for a subgroup with substantially lower recall.

## Git requirement

Use a feature branch, make at least two commits on it, and merge it into main:

```bash
git checkout -b feature/support-agent
git add . && git commit -m "add return-risk pipeline"
git add . && git commit -m "add transfer learning and agent"
git checkout main
git merge --no-ff feature/support-agent -m "merge support-agent feature"
```
