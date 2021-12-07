import json
import os
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
import torch
import pandas as pd
import csv
import pathlib
import argparse
import random

# const yelp labels with three attributes
labels = ["PresPosPlural", "PresPosSingular", "PresPosNeutral",
         "PastPosPlural", "PastPosSingular", "PastPosNeutral",
         "PresNegPlural", "PresNegSingular", "PresNegNeutral",
         "PastNegPlural", "PastNegSingular", "PastNegNeutral"]


def init_model(exp_params, model_name):
    cuda2 = torch.device('cuda:0')

    embedding_size = exp_params["embedding_size"]
    hidden_size = exp_params["hidden_size"]
    rnn_type = exp_params['rnn_type']
    word_dropout = exp_params["word_dropout"]
    embedding_dropout = exp_params["embedding_dropout"]
    latent_size = exp_params["latent_size"]
    num_layers = 1
    batch_size = exp_params["batch_size"]
    bidirectional = bool(exp_params["bidirectional"])
    max_sequence_length = exp_params["max_sequence_length"]
    back = exp_params["back"]
    attribute_size = exp_params["attr_size"]
    wd_type = exp_params["word_drop_type"]
    num_samples = 2
    save_model_path = 'bin'

    vocab_dir = 'yelp_vocab.json'

    with open("bin/" + model_name + "/" + vocab_dir, 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=max_sequence_length,
        embedding_size=embedding_size,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        word_dropout=0,
        embedding_dropout=0,
        latent_size=latent_size,
        num_layers=num_layers,
        cuda=cuda2,
        bidirectional=bidirectional,
        attribute_size=attribute_size, # was fixed to 7
        word_dropout_type=wd_type, # was fixed to static
        back=back
    )
    print("initializing model:", model_name)
    print(model)

    return model, w2i, i2w


def get_generated_sentences(model, label, num_samples, w2i, i2w):
    generated_sent = []
    for s in range(num_samples):
        samples, z, l = model.inference(label)
        s = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
        generated_sent.append(s[0])

    return generated_sent


def generate_from_two_models(a, ai, label, epoch_a, epoch_ai, fraction_a, fraction_ai, num_samples):
    model_params = pd.read_csv(pathlib.Path('parameters', 'params.csv'))
    model_params = model_params.set_index('time')
    exp_params_model_a = model_params.loc[a]
    exp_params_model_ai = model_params.loc[ai]

    model_a, w2i_a, i2w_a = init_model(exp_params_model_a, a)
    model_ai, w2i_ai, i2w_ai = init_model(exp_params_model_ai, ai)

    assert(w2i_a == w2i_ai)
    assert(i2w_a == i2w_ai)
    w2i = w2i_a
    i2w = i2w_a

    checkpoint_a = pathlib.Path('bin', a, "E{0}.pytorch".format(epoch_a))
    checkpoint_ai = pathlib.Path('bin', ai, "E{0}.pytorch".format(epoch_ai))

    if not os.path.exists(checkpoint_a):
        raise FileNotFoundError(checkpoint_a)
    if not os.path.exists(checkpoint_ai):
        raise FileNotFoundError(checkpoint_ai)

    if torch.cuda.is_available():
        model_a = model_a.cuda()
        model_ai = model_ai.cuda()
        device = "cuda"
    else:
        device = "cpu"

    # load both pre-trained models
    model_a.load_state_dict(torch.load(checkpoint_a, map_location=torch.device(device)))
    model_ai.load_state_dict(torch.load(checkpoint_ai, map_location=torch.device(device)))

    model_a.eval()
    model_ai.eval()

    print('----------GENERATE-SAMPLES----------')
    num_samples_from_a = int(num_samples*fraction_a)
    num_samples_from_ai = int(num_samples * fraction_ai)
    print('Label:', label)
    print('Generate samples from A: {0}'.format(num_samples_from_a))
    print('Generate samples fromm Ai: {0}'.format(num_samples_from_ai))
    print('Total:', num_samples_from_a + num_samples_from_ai)

    generated_sent = []
    sentences_from_a = get_generated_sentences(model_a, label, num_samples_from_a, w2i, i2w)
    generated_sent.extend(sentences_from_a)
    sentences_from_ai = get_generated_sentences(model_ai, label, num_samples_from_ai, w2i, i2w)
    generated_sent.extend(sentences_from_ai)
    random.shuffle(generated_sent)

    return generated_sent


def main(args):

    output_file = pathlib.Path('{0}-{1}.txt'.format(args.model_A, args.model_Ai))
    with open(output_file, 'w') as txt_file:
        gen_writer = csv.writer(txt_file)
        gen_writer.writerow(["sentence", 'label'])

        for label in labels:
            print("Generating sentences for attribute:", label)
            print("Generating {0}% of sentences from model {1}".format(args.f_samples_A * 100, args.model_A))
            print("Generating {0}% of sentences from model {1}".format(args.f_samples_Ai * 100, args.model_Ai))
            generated_sentences = generate_from_two_models(args.model_A, args.model_Ai, label, args.epoch_A,
                                                           args.epoch_Ai, args.f_samples_A, args.f_samples_Ai,
                                                           args.samples_per_label,)
            for sent in generated_sentences:
                gen_writer.writerow([sent, label])


if __name__ == "__main__":

    # default model A
    model_a = '2021-Mar-04-14:08:40'
    # default model A'
    model_ai = '2021-Mar-05-11:33:06'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_A', type=str, default=model_a)
    parser.add_argument('--model_Ai', type=str, default=model_ai)
    parser.add_argument('--epoch_A', type=int, default=15)
    parser.add_argument('--epoch_Ai', type=int, default=15)
    parser.add_argument('--samples_per_label', type=int, default=100)
    parser.add_argument('--f_samples_A', type=float, default=0.20)
    parser.add_argument('--f_samples_Ai', type=float, default=0.80)

    args = parser.parse_args()
    main(args)