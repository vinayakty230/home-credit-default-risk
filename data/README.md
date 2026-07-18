# Data

The dataset is **not** included in this repository — the [Kaggle competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules) prohibit redistribution, and the files (~2.6 GB across 7 CSVs) exceed GitHub's file-size limits. This entire directory is git-ignored.

## Download

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (free account required — accept the competition rules on the Kaggle site once first):

```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p .
unzip home-credit-default-risk.zip -d home-credit-default-risk/
```

## Expected layout

The notebook reads from `data/home-credit-default-risk/`:

```
data/home-credit-default-risk/
├── application_train.csv        # main table, 307,511 applicants × 122 cols (has TARGET)
├── application_test.csv         # 48,744 applicants (no TARGET)
├── bureau.csv                   # 1.7M external-bureau credit records
├── bureau_balance.csv           # 27.3M monthly bureau statuses
├── previous_application.csv     # 1.7M previous Home Credit applications
├── POS_CASH_balance.csv         # 10.0M monthly POS/cash-loan snapshots
├── credit_card_balance.csv      # 3.8M monthly credit-card snapshots
└── installments_payments.csv    # 13.6M installment payment records
```

The feature-engineering section of the notebook also writes an intermediate `df_main_FINAL.csv` (~1.5 GB, 356,251 × 604) next to the notebook; it is git-ignored as well.
