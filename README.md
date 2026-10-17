# Autoregressive Text Generation with SimpleRNN, BiRNN, LSTM and GRU

**Course:** Foundations of Generative AI (ICT4403) - Assignment 2
**Student:** Sunku Venkata Subhash (230911370)

## 1. Problem statement and objective
Build a word-level autoregressive language model that predicts the next word from the previous 20 words and generates text one word at a time. Four recurrent architectures (SimpleRNN, Bidirectional RNN, LSTM, GRU) are trained under identical conditions and compared quantitatively (loss, perplexity, accuracy, training time) and qualitatively (coherence, diversity, contextual consistency, repetition), including the effect of sampling temperature.

## 2. Dataset
- **Name:** Tiny Shakespeare (concatenated plays of Shakespeare)
- **Source:** https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- **Characteristics:** about 1.1 million characters and 200k whitespace-separated words of play text in dialogue format: a speaker label ending in a colon, followed by lines of verse or prose. Archaic vocabulary ("thou", "thee", "doth"), heavy punctuation, many proper names. Full statistics, word-frequency, Zipf and line-length plots: `results/dataset_stats.md` and `results/dataset_characteristics.png`.
- **Preprocessing:** lower-casing; regex tokenisation into words and punctuation marks; line breaks kept as a `<nl>` token (so the model can learn the play layout); vocabulary built from the training split only (frequency >= 3, others mapped to `<unk>`); `<pad>` used to left-pad short seeds.
- **Input-target sequences:** sliding window with stride 1. Input = 20 consecutive tokens, target = the next token. Contiguous 80/10/10 train/validation/test split, so windows never overlap across splits.

## 3. Models
All models: Embedding(128) -> one recurrent layer (256 units) -> Dropout(0.3) -> Linear(vocabulary).
- **SimpleRNN:** `nn.RNN` (tanh), last hidden state.
- **BiRNN:** bidirectional `nn.RNN`; final forward and backward states concatenated. Bidirectionality is used only over the input window, so there is no leakage of the target.
- **LSTM:** `nn.LSTM`.
- **GRU:** `nn.GRU`.

Shared training setup: Adam (lr 2e-3), batch size 256, cross-entropy loss, gradient clipping 1.0, up to 20 epochs, early stopping (patience 3) on validation loss with the best weights restored, identical random seed (42). Hidden size is the same for all models, so parameter counts differ naturally with the cell type.

## 4. Execution steps
1. Open `FGAI_Assignment2_TextGeneration.ipynb` in Google Colab, select a GPU runtime and Run all (about 5 minutes), **or** locally: `pip install -r requirements.txt` then `python fgai_textgen.py`.
2. The dataset is downloaded automatically. Outputs are written to `results/`.

Dependencies: Python 3.10+, torch, numpy, pandas, matplotlib.

## 5. Results (test set)

| Model | Parameters | Epochs run | Best epoch | Train time (s) | Time / epoch (s) | Test loss | Test perplexity | Top-1 acc (%) | Top-5 acc (%) |
|---|---|---|---|---|---|---|---|---|---|
| SimpleRNN | 1,754,316 | 7 | 4 | 27.6 | 3.9 | 4.1765 | 65.14 | 28.51 | 46.85 |
| BiRNN | 2,953,932 | 5 | 2 | 33.2 | 6.6 | 4.3137 | 74.72 | 27.59 | 45.27 |
| LSTM | 2,050,764 | 8 | 5 | 50.4 | 6.3 | 4.0192 | 55.66 | 30.44 | 48.99 |
| GRU | 1,951,948 | 6 | 3 | 34.6 | 5.8 | 4.0105 | **55.17** | 29.87 | **49.05** |

Plots: `results/loss_curves.png`, `results/metric_comparison.png`. Raw files: `results/model_comparison.csv`, `results/training_history.json`.

## 6. Model comparison and evaluation

**Quantitative.**
- **Perplexity / loss:** GRU (55.17) and LSTM (55.66) are clearly better than SimpleRNN (65.14), roughly 15% lower perplexity, and BiRNN is worst (74.72). The GRU-LSTM gap is under 1% from a single run, so they should be regarded as effectively tied.
- **Accuracy:** LSTM has the best top-1 accuracy (30.44%), GRU the best top-5 (49.05%). SimpleRNN and BiRNN trail by 1-3 points.
- **Training time:** SimpleRNN is cheapest (3.9 s/epoch). GRU (5.8) is slightly cheaper per epoch than LSTM (6.3). Total time depends on how many epochs ran before early stopping; approximate time to reach each model's best epoch is GRU about 17 s, LSTM about 32 s, SimpleRNN about 16 s, BiRNN about 13 s. Timings include validation and are from one run on one GPU.
- **Overfitting:** every model reached its best validation loss after only 2-5 epochs and then got worse (each run stopped exactly 3 epochs after its best). The corpus is small (about 200k words), so perplexities near 55 are largely data-limited. BiRNN, the largest model (2.95M parameters), overfit fastest (best epoch 2). A plausible reason it does not help is that next-word prediction depends mostly on the most recent words, so the backward pass over the window adds parameters but little useful signal; this explanation was not tested separately.

**Qualitative** (from reading the samples in `results/generated_samples.md`):

| Model | Coherence | Diversity | Contextual consistency | Repetition |
|---|---|---|---|---|
| SimpleRNN | Short grammatical fragments, drifts quickly | Medium | Learns the speaker-label layout, weak topic memory | Some echoes ("sir, sir") |
| BiRNN | Most fragmented, odd function-word sequences ("one the lady of norfolk") | Highest at low temperature, but not meaningful | Layout learned, topic lost | Low |
| LSTM | Best low-temperature phrases (e.g. "ay, my lord, i am not no more ... let me die") | Medium | Keeps the play format and plausible speakers | Some loops at T=0.5 |
| GRU | Comparable to LSTM, smooth "king / duke of york" style phrases | Medium-high at T=0.8-1.0 | Same as LSTM | Most repetition at T=0.5 (e.g. "warwick, warwick") |

Across all models, local grammar and the play layout (newline, speaker name, colon, speech) are learned well, but none of the models follows the seed's content or keeps a topic for more than a few words. This is expected from a 20-word window and a small model. Samples also show cosmetic artefacts of simple tokenisation: a stray `<nl>` token at the start of some samples and dash runs ("- - -").

## 7. Sampling-temperature analysis
Same three seeds, same random draws per seed for all models, temperatures 0.5, 0.8, 1.0, 1.5; 10 samples of 40 words per cell. Metrics: Distinct-1/2 (unique unigrams/bigrams, higher means more diverse), Repetition (share of repeated trigrams, lower is better), Bigram validity (share of generated bigrams that occur in the training text, used as a proxy for local coherence). Mean over the four models:

| Temperature | Distinct-1 | Distinct-2 | Repetition | Bigram validity |
|---|---|---|---|---|
| 0.5 | 0.628 | 0.891 | 0.034 | 0.968 |
| 0.8 | 0.691 | 0.930 | 0.013 | 0.888 |
| 1.0 | 0.734 | 0.954 | 0.009 | 0.798 |
| 1.5 | 0.861 | 0.992 | 0.000 | 0.475 |

Per-model numbers: `results/generation_metrics_by_temperature.csv`; plot: `results/temperature_analysis.png`.

- **T = 0.5 (conservative):** highest local coherence (bigram validity 0.95-0.98) and the most repetition (up to 0.05 trigram repeat for GRU). Text sticks to frequent phrases ("king richard iii:", "my lord", "the king"), is grammatical, and sometimes loops or repeats words ("warwick, warwick", "and, and").
- **T = 0.8 (balanced):** coherence is still high (about 0.89), diversity rises and repetition drops to about 0.013. Sentences stay plausible while using a wider vocabulary and more varied speakers.
- **T = 1.0:** more diverse (Distinct-2 0.95) but coherence falls to about 0.80; unusual word combinations and odd names start appearing.
- **T = 1.5 (too random):** Distinct-2 about 0.99 and almost no repetition, but bigram validity drops below 0.5 and the text is word salad ("give downfall but shall die herself pronounced added"). High diversity here reflects randomness, not quality.

Higher temperature monotonically increases diversity and decreases coherence and repetition for every model. The useful range for this dataset is T = 0.7-1.0, with 0.8 the best trade-off. Caveats: 40-word samples make trigram repetition low in general, and the repetition metric does not capture immediate word repeats well; bigram validity only measures local plausibility, not meaning.

## 8. Conclusion
**GRU is the most effective model for this dataset.** It has the lowest test loss and perplexity (4.0105 / 55.17), the best top-5 accuracy, reached its best validation loss in the fewest epochs among the gated models (3 vs 5 for LSTM), has fewer parameters and a lower per-epoch cost than LSTM, and at T = 0.8 gives a good balance (Distinct-2 0.935, Bigram validity 0.897, Repetition 0.011). LSTM is a very close second (best top-1 accuracy, slightly higher coherence at low temperature). The gap between them is small and comes from a single run, so the decision rests on the combination of accuracy and efficiency rather than on a large difference. SimpleRNN is cheapest but clearly worse, and BiRNN is the worst here: the extra parameters overfit and gave no benefit for next-word prediction. Recommended generation setting: GRU with temperature about 0.8.

**Limitations and future work:** one training run per model (no confidence intervals), small corpus with early overfitting, word-level vocabulary with `<unk>`, and a short 20-word context. Possible improvements: more dropout/weight decay, tied embeddings, a larger dataset, longer context, stacked layers, top-k/nucleus sampling, and multiple seeds for significance.

## Repository contents
`FGAI_Assignment2_TextGeneration.ipynb` (main notebook), `fgai_textgen.py` (same code as script), `requirements.txt`, `results/` (tables, plots, generated samples, dataset statistics).
