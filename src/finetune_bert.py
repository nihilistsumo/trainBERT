import json, os, pprint, datetime, sys, logging, argparse

import tensorflow as tf

from google.cloud import storage
from bert import modeling, optimization, tokenization, run_classifier
from bert.run_pretraining import input_fn_builder, model_fn_builder

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def model_predict(estimator, processor, input_dir, predict_batch_size, label_list, max_sequence_length, tokenizer):
    prediction_examples = processor.get_dev_examples(input_dir)[:predict_batch_size]
    input_features = run_classifier.convert_examples_to_features(prediction_examples, label_list, max_sequence_length,
                                                                 tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=max_sequence_length,
                                                       is_training=False, drop_remainder=True)
    predictions = estimator.predict(predict_input_fn)
    for example, prediction in zip(prediction_examples, predictions):
        print('text_a: %s\ntext_b: %s\nlabel:%s\nprediction:%s\n' % (example.text_a, example.text_b, str(example.label), prediction['probabilities']))

def model_eval(estimator, processor, input_dir, label_list, max_sequence_length, tokenizer, eval_batch_size, output_dir):
    eval_examples = processor.get_dev_examples(input_dir)
    eval_features = run_classifier.convert_examples_to_features(
        eval_examples, label_list, max_sequence_length, tokenizer)
    print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(eval_examples)))
    print('  Batch size = {}'.format(eval_batch_size))

    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / eval_batch_size)
    eval_input_fn = run_classifier.input_fn_builder(
        features=eval_features,
        seq_length=max_sequence_length,
        is_training=False,
        drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

def get_run_config(output_dir, tpu_cluster_resolver, config):
    return tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=output_dir,
        save_checkpoints_steps=config["save_checkpoint_steps"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=config["iterations_per_loop"],
            num_shards=config["num_tpu_cores"],
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

def finetune(config, bucket_path, input_dir, model_dir, output_dir, vocab_file, tpu_name):
    processor = run_classifier.MrpcProcessor()
    label_list = processor.get_labels()

    # Compute number of train and warmup steps from batch size
    train_examples = processor.get_train_examples(input_dir)
    num_train_steps = int(len(train_examples) / int(config["train_batch_size"]) * float(config["num_train_epochs"]))
    num_warmup_steps = int(num_train_steps * float(config["warmup_proportion"]))
    train_batch_size = int(config["train_batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    predict_batch_size = int(config["predict_batch_size"])
    do_lowercase = bool(config["do_lowercase"])
    max_sequence_length = int(config["max_seq_length"])

    BERT_GCS_DIR = "{}/{}".format(bucket_path, model_dir)
    INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)
    CONFIG_FILE = os.path.join(BERT_GCS_DIR, "bert_config.json")
    VOCAB_FILE = os.path.join(BERT_GCS_DIR, vocab_file)
    bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name)

    model_fn = run_classifier.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=float(config["learning_rate"]),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=True,
        use_one_hot_embeddings=True
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=get_run_config(output_dir, tpu_cluster_resolver, config),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size
    )

    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE,
                                           do_lower_case=do_lowercase)
    train_features = run_classifier.convert_examples_to_features(
        train_examples, label_list, max_sequence_length, tokenizer)

    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(train_examples)))
    print('  Batch size = {}'.format(train_batch_size))
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=max_sequence_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.datetime.now()))

    model_eval(estimator, processor, input_dir, label_list, max_sequence_length, tokenizer, eval_batch_size, output_dir)
    model_predict(estimator, processor, input_dir, predict_batch_size, label_list, max_sequence_length, tokenizer)

def main():
    del_all_flags(tf.flags.FLAGS)
    parser = argparse.ArgumentParser(description="Fine-tune BERT model")
    parser.add_argument('-b', '--bucket_path', help="Path to google bucket")
    parser.add_argument('-m', '--model_dir', help="Name of model directory")
    parser.add_argument('-d', '--data_dir', help="Name of the train data directory")
    parser.add_argument('-o', '--output_dir', help="Path to output directory")
    parser.add_argument('-v', '--vocab_file', help="Name of vocab file")
    parser.add_argument('-t', '--tpu_name', help="Name of the tpu")

    args = vars(parser.parse_args())
    bucket_path = args['bucket_path']
    model_dir = args['model_dir']
    data_dir = args['data_dir']
    out_dir = args['output_dir']
    vocab = args['vocab_file']
    tpu = args['tpu_name']
    config = dict()
    with open("config", 'r') as cf:
        for l in cf:
            if l[0] == '#':
                continue
            config[l.split('=')[0]] = l.split('=')[1].rstrip()

    finetune(config, bucket_path, data_dir, model_dir, out_dir, vocab, tpu)

if __name__ == '__main__':
    main()