## Model comparison (test set)

| Model | Parameters | Epochs_run | Best_epoch | Train_time_s | Time_per_epoch_s | Test_loss | Test_perplexity | Top1_acc | Top5_acc |
|---|---|---|---|---|---|---|---|---|---|
| SimpleRNN | 1754316 | 7 | 4 | 27.6 | 3.9 | 4.1765 | 65.14 | 28.51 | 46.85 |
| BiRNN | 2953932 | 5 | 2 | 33.2 | 6.6 | 4.3137 | 74.72 | 27.59 | 45.27 |
| LSTM | 2050764 | 8 | 5 | 50.4 | 6.3 | 4.0192 | 55.66 | 30.44 | 48.99 |
| GRU | 1951948 | 6 | 3 | 34.6 | 5.8 | 4.0105 | 55.17 | 29.87 | 49.05 |

Lowest test perplexity: **GRU**

## Generation metrics by temperature (mean over 3 seeds x 10 samples)

| Model | Temperature | Distinct1 | Distinct2 | Repetition | BigramValidity |
|---|---|---|---|---|---|
| BiRNN | 0.5 | 0.653 | 0.915 | 0.02 | 0.95 |
| BiRNN | 0.8 | 0.714 | 0.94 | 0.012 | 0.862 |
| BiRNN | 1.0 | 0.734 | 0.956 | 0.011 | 0.798 |
| BiRNN | 1.5 | 0.869 | 0.994 | 0.0 | 0.45 |
| GRU | 0.5 | 0.587 | 0.861 | 0.05 | 0.969 |
| GRU | 0.8 | 0.682 | 0.935 | 0.011 | 0.897 |
| GRU | 1.0 | 0.741 | 0.962 | 0.007 | 0.79 |
| GRU | 1.5 | 0.863 | 0.989 | 0.001 | 0.44 |
| LSTM | 0.5 | 0.636 | 0.89 | 0.035 | 0.978 |
| LSTM | 0.8 | 0.674 | 0.919 | 0.017 | 0.898 |
| LSTM | 1.0 | 0.728 | 0.956 | 0.008 | 0.826 |
| LSTM | 1.5 | 0.842 | 0.987 | 0.001 | 0.521 |
| SimpleRNN | 0.5 | 0.635 | 0.897 | 0.029 | 0.974 |
| SimpleRNN | 0.8 | 0.694 | 0.926 | 0.012 | 0.896 |
| SimpleRNN | 1.0 | 0.733 | 0.943 | 0.009 | 0.776 |
| SimpleRNN | 1.5 | 0.87 | 0.996 | 0.0 | 0.488 |