# backtrace: supervised model on UAH

the model: score = 100 x (w1 speed + w2 accel² + w3 gap + w4 lane), weights learned per environment (add up to 1), P(aggressive) = sigmoid((score - threshold) / tau), trained on 5 drivers, tested on the 6th.

## learned weights per fold (speed / accel² / gap / lane) and result on the held out driver

| held out | highway and urban weights | AUC supervised | AUC formula |
|---|---|---|---|
| D1 | highway 0.09/0.83/0.05/0.03; urban 0.27/0.48/0.18/0.07 | 0.568 | 0.761 |
| D2 | highway 0.14/0.75/0.10/0.02; urban 0.33/0.34/0.27/0.06 | 0.603 | 0.906 |
| D3 | highway 0.11/0.79/0.06/0.05; urban 0.33/0.44/0.20/0.03 | 0.579 | 0.803 |
| D4 | highway 0.12/0.78/0.06/0.03; urban 0.32/0.37/0.25/0.06 | 0.594 | 0.869 |
| D5 | highway 0.11/0.75/0.10/0.05; urban 0.32/0.39/0.24/0.05 | 0.524 | 0.889 |
| D6 | highway 0.12/0.73/0.09/0.06; urban 0.32/0.40/0.23/0.06 | 0.467 | 0.822 |

## how well each single term separates aggressive from normal, per driver (AUC)

| driver | speed | accel² | gap | lane |
|---|---|---|---|---|
| D1 | 0.777 | 0.600 | 0.689 | 0.466 |
| D2 | 0.669 | 0.788 | 0.867 | 0.610 |
| D3 | 0.594 | 0.740 | 0.756 | 0.531 |
| D4 | 0.646 | 0.687 | 0.846 | 0.570 |
| D5 | 0.781 | 0.753 | 0.823 | 0.575 |
| D6 | 0.839 | 0.826 | 0.720 | 0.468 |

mean over drivers: speed 0.718, accel2 0.732, prox2 0.784, wave 0.537; spread (sd) speed 0.095, accel2 0.080, prox2 0.072, wave 0.060
