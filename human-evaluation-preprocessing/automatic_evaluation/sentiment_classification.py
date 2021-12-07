import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import random
import numpy as np
from textCNN import CNN
import torch.optim as optim
import time
import spacy
import pandas as pd
import pathlib
from tqdm import tqdm


nlp = spacy.load('en_core_web_sm')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def sentiment_classification_with_text_cnn(cuda, sentences):
    SEED = 1234
    TEXT = data.Field(tokenize='spacy', batch_first=True)
    SENTIMENT = data.LabelField(dtype=torch.float, batch_first=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    fields = {'text': ('text', TEXT), 'sentiment': ('label', SENTIMENT)}
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=pathlib.Path('yelp_data'),
        train='train.json',
        validation='valid.json',
        test='test.json',
        format='json',
        fields=fields
    )
    print(vars(train_data[0]))
    print(vars(valid_data[0]))
    print(vars(test_data[0]))

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    SENTIMENT.build_vocab(train_data)
    print(SENTIMENT.vocab.stoi)

    BATCH_SIZE = 64

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: x.text,
        batch_size=BATCH_SIZE,
        device=device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # Train the model
    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')
    print('Training...')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut4-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    def predict_sentiment(model, sentence, min_len=5):
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(model(tensor))
        return prediction.item()

    # sanity check
    sentiment_predictions = []
    for i, s in enumerate(sentences):
        res = predict_sentiment(model, s)
        sentiment_predictions.append((i, s, res))

    return sentiment_predictions


def load_model_and_predict(cuda, sentences, min_len=5):
    device = 'cpu'

    TEXT = data.Field(batch_first=True)
    SENTIMENT = data.LabelField(dtype=torch.float)

    fields = {'text': ('text', TEXT), 'sentiment': ('label', SENTIMENT)}
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path='yelp_data',
        train='train.json',
        validation='valid.json',
        test='test.json',
        format='json',
        fields=fields
    )

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    # initialise the model param
    SENTIMENT.build_vocab(train_data)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load('textcnn-model.pt', map_location=torch.device(device)))

    model.eval()
    sentiment_predictions = []
    is_prob_pos = SENTIMENT.vocab.stoi['pos']
    for i, s in tqdm(enumerate(sentences)):
        tokenized = [tok.text for tok in nlp.tokenizer(s)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(model(tensor))
        if is_prob_pos:
            sentiment_predictions.append((i, s, prediction.item()))
        else:
            sentiment_predictions.append((i, s, (1-prediction.item())))

    return sentiment_predictions


def main():
    gpu = 1
    cuda = torch.device(f'cuda:{gpu}')
    sentences = pd.read_csv(pathlib.Path('yelp_data', 'small', 'yelp_train.csv'))
    sample = [i for i in sentences.review[:10]]
    sentiment_classification_with_text_cnn(cuda, sample)


if __name__ == "__main__":
    main()
