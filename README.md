# SmartIntentNN V2

Leverage TensorFlow with Python for model training, replacing TensorFlow.js.

CuDNN support is insufficient in `tfjs-node-gpu`.

## JS Version

The following JS version is now used for web tool API and ever used for SmartIntentNN V1:

<https://github.com/web3se-lab/web3-sekit>

## Installation

```sh
pip install -r requirements.txt
```

## Dataset

- **Training:** 16,000 (from ID: 1) smart contracts
- **Evaluation:** 10,000 (from ID: 20000) smart contracts

## Execution

**Run in python for Model Training and Evaluation:**

```bash
# Summarize model
python3 ./smartbert_bilstm.py summary

# Train model
python3 ./smartbert_bilstm.py train

# For 2nd-round balance training
python3 ./smartbert_bilstm.py train2

# Evaluate model
python3 ./smartbert_bilstm.py evaluate
```


**Convert to TFJS model:**

First change env variables in `convert_tfjs.sh`: `MODEL`, `DIR`, `JS`.

Then run convert:

```bash
./convert_tfjs.sh
```

**Run in ipynb for GPT Experiment:**

- **Important**: You first need to fill in the following variables in Step 1.
```bash
# You need to define your chat API and request headers here
# chat_api_url = 
# chat_headers = {}
```
- **[GPT_Experiment.ipynb](./GPT_Experiment.ipynb)**: This file contains the code for GPT experiment.
- **[prompt.md](./prompt.md)**: The prompt given to the GPT.


## Settings

Pre-trained model is **[SmartBERT](https://github.com/web3se-lab/SmartBERT)**, used for code embedding, trained on **16000** smart contracts.

- **Loss:** BinaryFocalCrossentropy
- **LSTM/BiLSM Units:** 128/256
- **Dropout:** 0.5
- **Mask Pad:** 0.0
- **Seq Length:** 256
- **Embedding Dim:** 768

### Complete Training Turn

2 rounds for sequential complete data training

- **Epoch:** 100
- **Batch:** 80
- **Batch Size:** 200
- **Total:** 16000
- **Start ID:** 1
- **End ID:** 17197
- **Learning Rate:** 0.001

The results are:

| Category       | Accuracy | Precision  | Recall     | F1 Score   |
| -------------- | -------- | ---------- | ---------- | ---------- |
| fee            | 0.9394   | 0.9288248  | 0.9279471  | 0.9283857  |
| disableTrading | 0.9643   | 0.8526682  | 0.761658   | 0.8045977  |
| blacklist      | 0.9689   | 0.84528834 | 0.74937654 | 0.7944481  |
| reflect        | 0.9887   | 0.98382837 | 0.9789819  | 0.9813991  |
| maxTX          | 0.9623   | 0.9480992  | 0.9287565  | 0.9383281  |
| mint           | 0.9513   | 0.82222223 | 0.79753655 | 0.8096912  |
| honeypot       | 0.9933   | 0.84615386 | 0.49107143 | 0.62146884 |
| reward         | 0.9937   | 0.96196514 | 0.9396285  | 0.95066553 |
| rebase         | 0.9915   | 0.5294118  | 0.10465116 | 0.17475724 |
| maxSell        | 0.9971   | 1.0        | 0.06451613 | 0.12121211 |
|                |          |            |            |            |
| Average        | 0.975050 | 0.87184619 | 0.67441238 | 0.71249536 |
| Total          | 0.97505  | 0.9269809  | 0.89620465 | 0.911333   |

### Balanced Sample Tranining Turn

1 round for random balanced-sample completion training

- **Epoch:** 100
- **Batch:** 20
- **Batch Size:** 100 Random data
- **Total:** 2000
- **Start ID:** 1
- **End ID:** 17197
- **Learning Rate:** 0.0001

The results are:

| Category       | Accuracy | Precision  | Recall     | F1 Score   |
| -------------- | -------- | ---------- | ---------- | ---------- |
| fee            | 0.9452   | 0.91173184 | 0.96385545 | 0.9370693  |
| disableTrading | 0.9753   | 0.8953745  | 0.84248704 | 0.8681259  |
| blacklist      | 0.9813   | 0.8579744  | 0.91895264 | 0.8874172  |
| reflect        | 0.993    | 0.9806139  | 0.9967159  | 0.9885993  |
| maxTX          | 0.975    | 0.9610137  | 0.95790154 | 0.959455   |
| mint           | 0.9393   | 0.72322583 | 0.86297154 | 0.7869428  |
| honeypot       | 0.991    | 0.5833333  | 0.6875     | 0.6311475  |
| reward         | 0.9947   | 0.94452775 | 0.9752322  | 0.95963436 |
| rebase         | 0.9958   | 0.7        | 0.89534885 | 0.78571427 |
| maxSell        | 0.9987   | 0.71428573 | 0.9677419  | 0.8219178  |
|                |          |            |            |            |
| Average        | 0.97893  | 0.82720809 | 0.9068707  | 0.86260234 |
| Total          | 0.97893  | 0.9089568  | 0.947648   | 0.92789924 |
