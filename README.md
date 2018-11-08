# Machine Reading Comprehension
This repo is based on the following papers:
- [QANet](https://openreview.net/pdf?id=B14TlG-RW)
- [R-Net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

**NOTE**: Most of the code is inspired and/or collected and/or modified from various github repositories on the above 2 models. I thank the corresponding authors.
- [QA Net Python 2 implementation of NLP Learn Community](https://github.com/NLPLearn/QANet)
- [R-Net of NLP Learn Community](https://github.com/NLPLearn/R-net)
- [R-Net of JKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net)

## Contents
- Installation
- Training
- Evaluation
- Prediction or Demo
- Tensorboard

## Installation
### Prerequisites
There are few python 3 packages which needs to installed to use this package.
- tensorflow or tensorflow-gpu(recommended when GPU is available)
- numpy
- tqdm
- bottle
- codecs
- spacy

All the prerequisites have been stored in requirements.txt file. Run the following command to install the packages (assuming pip here is pip3).
```python
pip install -r requirements.txt
# if this doesn't work on the server due to proxy errors - run the following
https_proxy=https://172.16.2.30:8080/ pip install -r requirements.txt
```

### Download Datasets
Run the following commands in the terminal to download the required datasets and embeddings files
```bash
# download SQuAD dataset and Glove embeddings
sh download.sh
```

**Note**: If the above doesn't work because of proxy errors, run the following commands:
```
# Download SQuAD
PWD=$(pwd)
SQUAD_DIR=$PWD/datasets/squad
mkdir -p $SQUAD_DIR

https_proxy=https://172.16.2.30:8080/ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json

https_proxy=https://172.16.2.30:8080/ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

# Download GloVe
GLOVE_DIR=$PWD/datasets/glove
mkdir -p $GLOVE_DIR

https_proxy=https://172.16.2.30:8080/ wget https://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# # Download Spacy language models
python3 -m spacy download en
```

Now the data will be stored in ./datasets folder with minimum of the following subfolders:
- glove
- squad

### Preprocess Datasets
Run the following python script after an environment has been created with required prerequisite python 3 packages
```python
python main.py --mode preprocess
```
Now the preprocessed data will be stored in ./data folder

## Training
- The following commands are the ones we use to train/test/debug or demo the model
```
python main.py --mode debug/train/test/demo
```

- To use custom parameters, run the above command with the command line optional parameters like <parameter> <value> or <parameter>=<value>. An example to run with different num of steps than default is:
```
python main.py --mode train --num_steps 120000
```

All the optional parameters are defined in the get_flags function in main.py

Model running will take time.


## Evaluation
Use the following file to evaluate the model predictions
```
python evaluate-v1.1.py ~/data/squad/dev-v1.1.json train/{model_location}/answer/answer.json
```

## Prediction or Demo
- In order to use demo, we need to migrate the server port, so use this command to login if you working on a remote server.
```
ssh -L 16006:127.0.0.1:6006 <username>@10.35.32.101
```
This command migrates the default port to 16006 port of server.

Once this is done we can run the following command to start the demo.
```
python main.py --mode demo
```
![Sample Demo](https://i.imgur.com/GvCfHjR.png)

Enter the passage and question and click the get answer button to get the answer.

## Tensorboard
To view the model in tensorboard, first login using the following command.
```
ssh -L 16006:127.0.0.1:6006 <username>@10.35.32.101
```
The go to the model folder and then events folder.
```
tensorboard --logdir=./
```
