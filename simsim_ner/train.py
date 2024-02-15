import pandas as pd
from torch import cuda
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from core.data.dataloader.IBO_dataloader import dataset
from core.models.bert import BertForTokenClassification
from core.models.bert import NERTrainer, NERValidator
import torch
def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    # Command line arguments for file path and other parameters
    parser = argparse.ArgumentParser(description='NER Training and Validation')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file', required=True)
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-05, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=int, default=10, help='Maximum gradient norm')
    args = parser.parse_args()

    # Load data from CSV file
    data = pd.read_csv(args.file_path, encoding='unicode_escape')
    entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
    data = data[~data.Tag.isin(entities_to_remove)]
    data = data.fillna(method='ffill')
    data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
    print(data.head())

    # Label mapping
    label2id = {k: v for v, k in enumerate(data.Tag.unique())}
    id2label = {v: k for v, k in enumerate(data.Tag.unique())}
    data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)

    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split data into train and test sets
    train_size = 0.8
    train_dataset = data.sample(frac=train_size, random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Tokenize and create DataLoader for training and testing sets
    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': args.train_batch_size, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': args.valid_batch_size, 'shuffle': True, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # Initialize BERT model for token classification
    model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(id2label),
                                                      id2label=id2label,
                                                      label2id=label2id)

    model.to(device)

    # Train the model
    trainer = NERTrainer(model, training_loader, testing_loader, torch.optim.Adam(model.parameters(), lr=args.learning_rate), args.max_grad_norm)
    trainer.train(args.epochs)

    # Validate the model
    validator = NERValidator(model, testing_loader)
    labels, predictions = validator.validate()

if __name__ == "__main__":
    import argparse
    main()
