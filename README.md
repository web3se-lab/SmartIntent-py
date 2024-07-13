# SmartIntentNN Python

Use Tensorflow Python to replace Tensorflow.js, for model training.

As CuDNN is not well applied on `tfjs-node-gpu`

## JS Version

JS version is only for web tool API and some model evaluation:

<https://github.com/decentralizedlab/web3-sekit>

## Install

```
pip install -r ./requirement.txt
```

## Dataset

Train on **20,000** rows: 1-20000
Evaluate on **10,000** rows: 21000-30999

## Use

```bash
# summary model
python3 ./lstm_smartbert.py summary

# train model
python3 ./lstm_smartbert.py train

# evaluate model
python3 ./lstm_smartbert.py evaluate
```

## Setting (New)

loss: BinaryFocalCrossentropy
LSTM Unit: 128
