import os
import sys
import tensorflow as tf
from datetime import datetime
import ujson as json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'src'))

from prepro import prepro as preprocess
from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset

def get_new_folder():
    return datetime.now().strftime("%y-%d-%m-%H-%M-%S")

def get_flags():
    flags = tf.flags

    home = os.getcwd()
    train_file = os.path.join(home, "datasets", "squad", "train-v1.1.json")
    dev_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
    test_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
    glove_word_file = os.path.join(home, "datasets", "glove", "glove.840B.300d.txt")

    train_dir = "train"
    # model_name = get_new_folder()
    model_name = "FRC"
    dir_name = os.path.join(train_dir, model_name)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
        os.mkdir(os.path.join(os.getcwd(),dir_name))
    target_dir = "data"
    log_dir = os.path.join(dir_name, "event")
    save_dir = os.path.join(dir_name, "model")
    answer_dir = os.path.join(dir_name, "answer")
    train_record_file = os.path.join(target_dir, "train.tfrecords")
    dev_record_file = os.path.join(target_dir, "dev.tfrecords")
    test_record_file = os.path.join(target_dir, "test.tfrecords")
    word_emb_file = os.path.join(target_dir, "word_emb.json")
    char_emb_file = os.path.join(target_dir, "char_emb.json")
    train_eval = os.path.join(target_dir, "train_eval.json")
    dev_eval = os.path.join(target_dir, "dev_eval.json")
    test_eval = os.path.join(target_dir, "test_eval.json")
    dev_meta = os.path.join(target_dir, "dev_meta.json")
    test_meta = os.path.join(target_dir, "test_meta.json")
    word_dictionary = os.path.join(target_dir, "word_dictionary.json")
    char_dictionary = os.path.join(target_dir, "char_dictionary.json")
    answer_file = os.path.join(answer_dir, "answer.json")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(answer_dir):
        os.makedirs(answer_dir)

    flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

    flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
    flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
    flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
    flags.DEFINE_string("train_file", train_file, "Train source file")
    flags.DEFINE_string("dev_file", dev_file, "Dev source file")
    flags.DEFINE_string("test_file", test_file, "Test source file")
    flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

    flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
    flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
    flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
    flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
    flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
    flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
    flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
    flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
    flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
    flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
    flags.DEFINE_string("answer_file", answer_file, "Out file for answer")
    flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
    flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")


    flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
    flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
    flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
    flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

    flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
    flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
    flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
    flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
    flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
    flags.DEFINE_integer("char_limit", 16, "Limit length for character")
    flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
    flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

    flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
    flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
    flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
    flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

    flags.DEFINE_integer("batch_size", 32, "Batch size")
    flags.DEFINE_integer("num_steps", 60000, "Number of steps")
    flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
    flags.DEFINE_integer("period", 100, "period to save batch loss")
    flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
    flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
    flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
    flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
    flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
    flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
    flags.DEFINE_integer("hidden", 96, "Hidden size")
    flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
    flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")

    # Extensions (Uncomment corresponding code in download.sh to download the required data)
    glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
    flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
    flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")

    fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
    flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
    flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")
    return flags

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save, patience, best_f1, best_em = 100.0, 0, 0., 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir, graph=graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)

            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                          handle: train_handle, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2 = sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2])
                answer_dict_, remapped_dict_ = convert_tokens(
                    eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)
                losses.append(loss)
            loss = np.mean(losses)
            metrics = evaluate(eval_file, answer_dict)
            with open(config.answer_file, "w") as fh:
                json.dump(remapped_dict, fh)
            print("Exact Match: {}, F1: {}".format(
                metrics['exact_match'], metrics['f1']))

def main(_):
    flags = get_flags()
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "preprocess":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "demo":
        demo(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
