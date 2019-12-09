import json, os, pprint, time, sys, logging, argparse

import tensorflow as tf
import sentencepiece as sp

from google.cloud import storage
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token

def build_vocab(input_file, vocab_file, vocab_size, subsample_size, num_placeholders):
    print("Using sentencepiece to train and build the vocab file: "+vocab_file)
    command = ('--input={} --model_prefix={} --vocab_size={} --input_sentence_size={} --shuffle_input_sentence=true'
               ' --bos_id=-1 --eos_id=-1').format(input_file, vocab_file, vocab_size - num_placeholders, subsample_size)
    sp.SentencePieceTrainer.Train(command)
    print("Now converting "+vocab_file+" to BERT vocab")
    vocab = []
    with open(vocab_file + ".vocab", 'r') as f:
        for l in f:
            vocab.append(l.split('\t')[0])
    vocab = vocab[1:]
    print("Vocab length: " + str(len(vocab)))
    bert_vocab = list(map(parse_sentencepiece_token, vocab))
    ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    bert_vocab = ctrl_symbols + bert_vocab
    bert_vocab += ["[UNUSED_{}]".format(i) for i in range(vocab_size - len(bert_vocab))]
    print(len(bert_vocab))
    VOC_FNAME = "vocab.txt"  # @param {type:"string"}
    with open(VOC_FNAME, 'w') as fo:
        for token in bert_vocab:
            fo.write(token + "\n")
    print("BERT vocab file written in "+VOC_FNAME)
    print("Let's use BERT tokenizer using the vocab to tokenize the following:")
    print("Colorless geothermal substations are generating furiously")
    bert_tokenizer = tokenization.FullTokenizer(VOC_FNAME)
    print(bert_tokenizer.tokenize("Colorless geothermal substations are generating furiously"))

def main():
    parser = argparse.ArgumentParser(description="Train BERT model")
    parser.add_argument('-s', '--sent_file', help="Path to sentence file")
    parser.add_argument('-m', '--mode', required=True, help="Mode of operation")

    args = vars(parser.parse_args())
    sent_file = args['sent_file']
    mode = args['mode']
    config = dict()
    with open("config", 'r') as cf:
        for l in cf:
            config[l.split(':')[0]] = l.split(':')[1]
    if mode == 'v':
        build_vocab(sent_file, config['vocab_file'], int(config['vocab_size'],
        int(config['subsample_size']), int(config['num_placeholders'])))

if __name__ == '__main__':
    main()