import json, os, pprint, time, sys, logging, argparse

import tensorflow as tf

from google.cloud import storage
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

def train(config, input_dir_path, model_dir, train_data_dir, vocab_file, tpu_name):
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :  %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    log.handlers = [sh]

    BERT_GCS_DIR = "{}/{}".format(input_dir_path, model_dir)
    DATA_GCS_DIR = "{}/{}".format(input_dir_path, train_data_dir)

    VOCAB_FILE = os.path.join(BERT_GCS_DIR, vocab_file)
    CONFIG_FILE = os.path.join(BERT_GCS_DIR, "bert_config.json")

    INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)

    bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
    input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR, '*tfrecord'))

    log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
    log.info("Using {} data shards".format(len(input_files)))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=float(config['learning_rate']),
        num_train_steps=int(config['train_steps']),
        num_warmup_steps=10,
        use_tpu=True,
        use_one_hot_embeddings=True)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=BERT_GCS_DIR,
        save_checkpoints_steps=int(config['save_checkpoint_steps']),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=int(config['save_checkpoint_steps']),
            num_shards=int(config['num_tpu_cores'])))
            #per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=int(config['train_batch_size']),
        eval_batch_size=int(config['eval_batch_size']))

    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=int(config['max_seq_length']),
        max_predictions_per_seq=int(config['max_predictions']),
        is_training=True)

    estimator.train(input_fn=train_input_fn, max_steps=int(config['train_steps']))

def main():
    parser = argparse.ArgumentParser(description="Train BERT model")
    parser.add_argument('-i', '--input_dir_path', help="Path to input data directory")
    parser.add_argument('-m', '--model_dir', help="Name of model directory")
    parser.add_argument('-d', '--data_dir', help="Name of the train data directory")
    parser.add_argument('-v', '--vocab_file', help="Name of vocab file")
    parser.add_argument('-t', '--tpu_name', help="Name of the tpu")

    args = vars(parser.parse_args())
    input_dir_path = args['input_dir_path']
    model_dir = args['model_dir']
    data_dir = args['data_dir']
    vocab = args['vocab_file']
    tpu = args['tpu_name']
    config = dict()
    with open("config", 'r') as cf:
        for l in cf:
            if l[0] == '#':
                continue
            config[l.split('=')[0]] = l.split('=')[1].rstrip()

    train(config, input_dir_path, model_dir, data_dir, vocab, tpu)

if __name__ == '__main__':
    main()