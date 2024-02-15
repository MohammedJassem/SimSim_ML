import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

data = pd.read_csv("//home/ubuntu/side_work/SimSim_ML/downloads/ner_datasetreference.csv", encoding='unicode_escape')


entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
data = data[~data.Tag.isin(entities_to_remove)]

data = data.fillna(method='ffill')

# let's create a new column called "sentence" which groups the words by sentence 
data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
# let's also create a new column called "word_labels" which groups the tags by sentence 
data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
print(data.head())
label2id = {k: v for v, k in enumerate(data.Tag.unique())}
id2label = {v: k for v, k in enumerate(data.Tag.unique())}
data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,wordpiece_tokenization = True):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.wordpiece_tokenization = wordpiece_tokenization
        
    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]  
        word_labels = self.data.word_labels[index]  
        tokenized_sentence, labels = self.tokenize_and_preserve_labels(sentence, word_labels)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]
        
        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(attn_mask, dtype=torch.long),
                #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
                'targets': torch.tensor(label_ids, dtype=torch.long)
        } 

    def __len__(self):
        return self.len

    def tokenize_and_preserve_labels(self,sentence, text_labels):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_sentence = []
        labels = []

        sentence = sentence.strip()

        for word, label in zip(sentence.split(), text_labels.split(",")):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            if self.wordpiece_tokenization:
                tokenized_sentence.extend(tokenized_word)
                labels.extend([label] * n_subwords)
            else:
                tokenized_sentence.extend(tokenized_word[0])
                labels.extend([label])

            
            

        return tokenized_sentence, labels

train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
