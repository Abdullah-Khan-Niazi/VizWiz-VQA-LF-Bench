# VizWiz VQA LF Bench

**The first systematic benchmark for long-form, rationale-grounded Visual Question Answering on the VizWiz dataset.**

Three architecturally distinct vision-language models are fine-tuned, evaluated, and compared on the VizWiz Long-Form (LF) dataset using a unified expert-holdout evaluation protocol. This repository documents the complete experimental pipeline, from raw data parsing through supervised fine-tuning to multi-metric evaluation and cross-model analysis.

---

## Why This Work Matters

The VizWiz dataset was originally designed for accessibility: real users who are blind or visually impaired submitted photos and voice-recorded questions. Standard VQA benchmarks reward short, categorical answers. The Long-Form extension of VizWiz asks a fundamentally different question: can a model produce a coherent, informative rationale rather than a single-word answer?

To our knowledge, this is the first controlled multi-model fine-tuning study on the VizWiz-LF dataset. Prior work either evaluates on standard VizWiz with short answers or applies single-model adaptation without a cross-architecture comparison. This project fills that gap.

---

## Repository Name and Description

**Suggested Repository Name:** `vizwiz-lf-bench`

**Suggested Description:**
> First multi-model benchmark for long-form rationale VQA on VizWiz. Compares InstructBLIP, a custom LLaVA-style MLP projector, and Qwen3-VL-8B QLoRA across BLEU-4, METEOR, ROUGE-L, VQA accuracy, and F1 using a shared expert holdout split.

---

## Dataset

**Source:** VizWiz Long-Form (LF) dataset hosted on Kaggle/Hugging Face.

**Kaggle Dataset:**
`f230017abdullahkhan/viz-wiz-standard-with-lf`

This dataset contains:

- `LF.json`: The long-form rationale file. Each entry is keyed by image ID and contains a `question` field, an `image_url` field, and a `long_answers` dictionary. The `long_answers` dictionary maps answer sources to answer objects. Each answer object has an `answer_paragraph` field with the full rationale text.
- `train/train/`: Training split images (JPEG).
- `val/val/`: Validation split images (JPEG).

**Answer Sources in LF.json:**

The `long_answers` dictionary in each entry contains answers from multiple sources including GPT-4V, LLaVA-generated synthetic answers, and human expert annotations. The source key identifies the provenance.

**Data Splitting Strategy:**

This project implements a strict source-based split. Any answer whose source key contains `expert` or `human` is routed to a shared `expert_holdout.json` that is held out from all training. All synthetic answers (GPT-4V, LLaVA, and others) form the training corpus. This ensures that evaluation is always performed against human expert rationales, not synthetic ones, making the evaluation protocol meaningful.

The split is performed identically across all three notebooks to guarantee a common holdout set.

**Hugging Face Model Repositories:**
- InstructBLIP: `Abdullah-Khan-Niazi/instructblip-vizwiz-lf`
- LLaVA Projector: `Abdullah-Khan-Niazi/vizwiz-lf-llava-projector`
- Qwen3-VL: `Abdullah-Khan-Niazi/vizwiz-lf-qwen3vl`

---

## Architecture Overview

Three architecturally distinct multimodal models are trained and compared. Each makes different design choices about which components are frozen, how visual features are aligned to text, and how memory constraints are handled.

### Model 1: InstructBLIP with Vicuna-7B (Notebook 1)

**Base Model:** `Salesforce/instructblip-vicuna-7b`

InstructBLIP uses a three-component architecture: a frozen CLIP-style vision encoder, a trainable Q-Former (Querying Transformer), and a frozen Vicuna-7B language model. The Q-Former is a lightweight cross-attention module from BLIP-2 that compresses variable-length visual features into a fixed number of learned query tokens. Those query tokens are then projected into the language model's embedding space via a linear projection layer.

**Training Strategy:** Only the Q-Former and the language projection layer are trained. The vision encoder and Vicuna-7B backbone are fully frozen. This minimizes the number of trainable parameters and is the most memory-efficient configuration.

**Key Architectural Decision:** The Q-Former and projection layer are explicitly upcast to FP32 before training. The rest of the model loads in 8-bit via BitsAndBytes. This prevents the PyTorch `GradScaler` from encountering zero-scale gradients in the quantized layers, which would otherwise cause training to stall silently.

**Trainable Parameters:** A small fraction of total model parameters, roughly corresponding to the Q-Former's cross-attention weights and the linear projection.

**Hardware:** Kaggle Dual-T4 (2x16 GB VRAM). The model uses `device_map="auto"` with 8-bit quantization and gradient checkpointing to fit within this budget.

**Image Loading:** Images are loaded with OpenCV (`cv2.imread`), resized to 224x224 immediately, converted to PIL format, and then passed to the InstructBLIP processor. This avoids VRAM spikes from loading full-resolution images during batching.

---

### Model 2: LLaVA-Style MLP Projector (Notebook 2)

**Components:**
- Vision Encoder: `openai/clip-vit-large-patch14` (frozen)
- Language Model: `microsoft/Phi-3-mini-4k-instruct` (frozen)
- Projector: Custom 2-layer MLP (trainable only)

This notebook implements a from-scratch LLaVA-style architecture rather than using a pretrained LLaVA checkpoint. CLIP-ViT-Large-Patch14 encodes each image into patch-level feature vectors. A two-layer MLP projector with hidden dimension 2048 maps those visual features into the token embedding space of Phi-3-mini-4k-instruct. The projector is the only trainable module; both the vision encoder and the language model are kept frozen throughout training.

**Key Architectural Decision:** Because only a small projector is trained, this model has the fewest trainable parameters of any model in the study. The rationale is to test whether a lightweight alignment module can bridge a strong vision encoder and a strong language model without any backbone adaptation.

**Projector Design:** The MLP is upcast to FP32. The surrounding frozen modules remain in FP16. This mirrors the InstructBLIP approach: mixed precision is applied selectively to prevent gradient underflow in the only component that receives gradient updates.

**CLIP Precomputation:** Before training, all CLIP image tensors are precomputed and saved to disk as FP16 `.pt` files in a dedicated directory. During training, the dataset reads these cached tensors directly instead of running the CLIP encoder on the fly. This eliminates the CLIP forward pass from the training loop entirely, reducing CPU-to-GPU transfer overhead and enabling higher effective throughput.

**Prompt Template:** Inputs to the language model follow the structure `<image>\nQuestion: {question}\nAnswer: {answer}`, which is standard for instruction-tuned LLaVA-style models.

**Hardware:** Kaggle Dual-T4 (2x16 GB VRAM).

---

### Model 3: Qwen3-VL-8B-Instruct with QLoRA (Notebook 3)

**Base Model:** `Qwen/Qwen3-VL-8B-Instruct`

Qwen3-VL is a fully integrated vision-language model. Unlike InstructBLIP and the LLaVA projector, it does not use a separate alignment module. The vision encoder and language decoder are jointly trained in the base model and share a unified multimodal attention mechanism. The model uses Multimodal Rotary Position Embeddings (M-RoPE) to encode both spatial image positions and token sequence positions simultaneously.

**Training Strategy:** Parameter-Efficient Fine-Tuning via QLoRA. The model is loaded in 4-bit NF4 quantization with double quantization enabled. LoRA adapters are injected into `q_proj` and `v_proj` of the language decoder's attention layers only. The LoRA rank is 16 with alpha 32 and dropout 0.05.

**Key Architectural Decision:** Qwen3-VL uses dynamic resolution processing: the number of image patch tokens varies depending on the input image dimensions. This is powerful for high-resolution images but it makes batching non-trivial because different samples in the same batch have different numbers of visual tokens. The solution is a batch size of 1 with gradient accumulation of 16 to achieve an effective batch size of 16, combined with a custom data collator that uses `torch.cat` for variable-length visual tensors (`pixel_values`, `image_grid_thw`) and `torch.stack` for fixed-length sequence tensors. Without this collator, the Trainer crashes with tensor shape mismatches.

**M-RoPE Crash Fix:** Qwen3-VL requires `mm_token_type_ids` to be extracted in `__getitem__` and passed through the collator. If this tensor is absent, M-RoPE cannot assign the correct positional encoding types (text vs. image) and raises a runtime error during the forward pass.

**Image Preprocessing:** Images are resized with `thumbnail((512, 512))` before processing. Qwen3-VL's dynamic resolution engine can generate a very large number of patch tokens for high-resolution images, which causes OOM errors on 16 GB VRAM. The thumbnail cap limits the maximum number of visual tokens per sample while preserving aspect ratio.

**Hardware:** Kaggle Dual-T4 (2x16 GB VRAM). This is the most memory-intensive of the three configurations.

---

## Complete Pipeline

The pipeline has four stages across four notebooks.

### Stage 1: Per-Model Training (Notebooks 1, 2, 3)

Each notebook is self-contained and follows the same logical sequence:

1. Install dependencies and configure paths and hyperparameters.
2. Build a RAM-resident `IMAGE_PATH_MAP` dictionary mapping image filenames to absolute paths. This is built once at startup. It is never rebuilt inside `__getitem__`. Calling `os.path.exists()` or `glob` inside a PyTorch dataset's `__getitem__` causes severe CPU starvation at scale.
3. Parse `LF.json`. Route expert/human answers to `expert_holdout.json` and synthetic answers to the training corpus. Save the holdout immediately to disk.
4. Load the model with the appropriate quantization configuration.
5. Freeze all components except the trainable module(s) for that architecture.
6. Create a PyTorch `Dataset` and `DataLoader` with a custom `collate_fn` where needed.
7. Train with Hugging Face `Trainer` or a custom training loop. Log loss per step to a CSV file. Push checkpoints to the Hugging Face Hub at each epoch via a custom `TrainerCallback`.
8. After training, run inference on the expert holdout set and save predictions.

### Stage 2: Unified Evaluation and Comparison (Notebook 4 / VIZWIZ.ipynb)

This notebook loads the saved predictions from all three models against the shared `expert_holdout.json` and computes a standardized set of metrics for each model.

**Metrics Computed:**

All metrics are computed programmatically with no human judgment involved.

- **Exact Accuracy:** The fraction of predictions that match the normalized ground truth exactly after lowercasing and punctuation removal.
- **VQA Accuracy:** The standard VQA evaluation metric. For each sample, the score is `min(1.0, count / 3)` where `count` is the number of annotator answers that match the prediction. This is the canonical metric from the original VQA challenge.
- **BLEU-4:** Corpus-level BLEU with four-gram precision and Chen and Cherry smoothing (method 1) to handle zero counts in short outputs.
- **METEOR:** Mean METEOR score computed token-by-token using NLTK. Accounts for stemming and synonymy, making it more lenient than BLEU for paraphrase-heavy long-form outputs.
- **ROUGE-L:** F-measure of the longest common subsequence between prediction and reference. Better suited than ROUGE-N for long-form outputs where sentence-level recall matters.
- **Precision, Recall, F1:** Weighted macro-averaged classification metrics treating each unique normalized ground-truth string as a class label. Out-of-vocabulary predictions (predictions that do not appear in any ground truth) are assigned to a synthetic class.

**Text Normalization:** Before all metric computations, both predictions and references are normalized: lowercased, punctuation stripped with a regex, and whitespace collapsed. This ensures that minor formatting differences do not penalize otherwise correct answers.

**Visualization:** The notebook produces per-model metric bar charts, confusion matrices for the top-8 most frequent ground-truth categories, and qualitative prediction examples showing the image, question, ground truth, and prediction side by side.

---

## Key Architectural Decisions Summarized

**Selective Component Training:** All three models freeze the large pretrained backbone components and train only the alignment mechanism. For InstructBLIP this is the Q-Former and projection head. For the LLaVA projector this is the MLP. For Qwen3-VL this is the LoRA adapters injected into attention projections. This choice is driven by the VRAM budget of Dual-T4 hardware and by the principle that pretrained representations should be preserved when the fine-tuning dataset is relatively small.

**FP32 Upcasting of Trainable Modules:** In InstructBLIP and the LLaVA projector, the trainable module is explicitly upcast to FP32 while the frozen modules remain in FP16 or INT8. This prevents the `GradScaler` from producing zero-scale gradient updates for the trainable module when the surrounding quantized modules have very small gradient magnitudes.

**Expert Holdout Isolation:** All three training runs explicitly exclude expert-annotated samples from the training corpus. This is not automatic: `LF.json` mixes synthetic and expert answers in the same file. The holdout split is source-based and is applied identically in all three notebooks. This allows the evaluation in Notebook 4 to compare all models against the same reference set without any risk of training data contamination.

**RAM Image Indexing:** All three notebooks build the full image path index into a Python dictionary before any training begins. The dictionary lookup in `__getitem__` is O(1). This pattern is critical for throughput on the Kaggle multi-worker DataLoader configuration.

**Dynamic Resolution Batching (Qwen3-VL):** Qwen3-VL's native dynamic resolution processing produces variable-length visual token sequences. Batching these requires a custom collator that concatenates visual tensors along the batch dimension with `torch.cat` rather than stacking them. This is a non-obvious requirement that differs from standard HuggingFace collation and must be handled explicitly to avoid shape errors in the model forward pass.

---

## Training Hyperparameters

| Parameter | InstructBLIP | LLaVA Projector | Qwen3-VL QLoRA |
|---|---|---|---|
| Effective Batch Size | 32 | 32 | 16 |
| Per-Device Batch | 4 | 4 | 1 |
| Gradient Accumulation | 8 | 8 | 16 |
| Learning Rate | 1e-4 | 2e-4 | 2e-4 |
| Epochs | 3 | 3 | 3 |
| Max Sequence Length | 512 | 512 | 2048 |
| Warmup Steps | 50 | 50 | 50 |
| Quantization | INT8 | FP16 | 4-bit NF4 |
| LoRA Rank | N/A | N/A | 16 |
| LoRA Alpha | N/A | N/A | 32 |

---

## Evaluation Metrics Glossary

**Exact Accuracy:** Strict string match after normalization. Penalizes any lexical deviation from the reference, including valid paraphrases. Useful as a lower bound.

**VQA Accuracy:** The official VQA evaluation protocol. Designed to handle annotator disagreement by rewarding partial consensus. A prediction scores 1.0 if at least three annotators agreed on it, and scores proportionally less for fewer matches. This metric is the standard for comparing against VQA challenge baselines.

**BLEU-4:** Measures n-gram precision up to 4-grams with a brevity penalty. A standard metric for text generation. Less meaningful for long free-form rationales than for constrained generation tasks, but included for comparability with prior work.

**METEOR:** Extends BLEU by incorporating recall and by using WordNet synonymy and stemming when matching tokens. More appropriate than BLEU for long-form outputs where paraphrasing is expected.

**ROUGE-L:** Measures recall of the longest common subsequence. Captures sentence-level structure better than n-gram metrics. Standard for summarization evaluation and suitable for rationale generation.

**Weighted F1:** Treats the task as classification over unique answer strings. Measures how well the model assigns answers to the correct answer categories. Out-of-vocabulary predictions are penalized as a separate class.

---

## Environment and Dependencies

All experiments run on Kaggle Notebooks with Dual-T4 GPUs (2x16 GB VRAM).

Core Python packages:
- `transformers` (latest)
- `peft`
- `bitsandbytes`
- `accelerate`
- `torch`
- `datasets`
- `huggingface_hub`
- `opencv-python-headless`
- `Pillow` (do not upgrade on Kaggle; upgrading breaks `torchvision`)
- `evaluate`
- `rouge_score`
- `nltk`
- `bert_score`
- `sacrebleu`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## Notebooks

| Notebook | Model | Key Component Trained |
|---|---|---|
| `notebook_1_Instructblip.ipynb` | InstructBLIP + Vicuna-7B | Q-Former + Language Projection |
| `notebook_2_llava.ipynb` | CLIP-ViT-L + MLP + Phi-3-mini | 2-Layer MLP Projector |
| `notebook_3_qwen3.ipynb` | Qwen3-VL-8B-Instruct | LoRA on q_proj and v_proj |
| `VIZWIZ.ipynb` | All three (inference + evaluation) | N/A (evaluation only) |

---

## Reproducibility Notes

To reproduce the experiments:

1. Download the dataset from Kaggle: `f230017abdullahkhan/viz-wiz-standard-with-lf`
2. Set your Hugging Face token in Kaggle Secrets under the key `HF_TOKEN`.
3. Run notebooks 1, 2, and 3 independently. Each saves its `expert_holdout.json` and model predictions to `/kaggle/working/`.
4. Run `VIZWIZ.ipynb` to load all three models, run inference, and generate evaluation results and plots.

The expert holdout split is deterministic given the same `LF.json` source file because the routing logic is based on the source key string, not on random sampling.

---

## Citation

If you use this benchmark, dataset split, or evaluation protocol in your work, please cite this repository and the original VizWiz dataset:

```
Gurari, D., et al. "VizWiz grand challenge: Answering visual questions from blind people."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
```

---

## License

This project is released for academic and research use. The underlying VizWiz dataset is subject to its own license terms. The LF.json annotations are subject to their respective source licenses (OpenAI, Meta, and human annotator agreements). Model weights derived from Salesforce InstructBLIP, Microsoft Phi-3, and Qwen3-VL are subject to their respective upstream licenses.
