from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
import numpy as np

import logging

from IPython import embed

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 8, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 300, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkout)points to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

# added
tf.app.flags.DEFINE_string("model_type", "gru", "specify either gru or lstm cell type for encoding")
tf.app.flags.DEFINE_integer("debug", 1, "whether to set debug or not")
tf.app.flags.DEFINE_integer("grad_clip", 1, "whether to clip gradients or not")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def initialize_data(data_path,keep_as_string=False):
    if tf.gfile.Exists(data_path):
        print ("LOADING:",data_path)
        dataset = []
        with tf.gfile.GFile(data_path, mode="rb") as f:
            dataset.extend(f.readlines())
        dataset = [line.strip('\n').split() for line in dataset]
        if not keep_as_string:
            dataset = [[int(num) for num in line] for line in dataset]
        return dataset
    else:
        raise ValueError("Vocabulary file %s not found.", data_path)

    
def initialize_embeddings(embed_path):
    
    if tf.gfile.Exists(embed_path):
        embeddings=np.load(embed_path)
        return embeddings['glove']

    else:
        raise ValueError("Embeddings file %s not found.", embed_path)

    
def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None


    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    context_ids_path = pjoin(FLAGS.data_dir, "train.ids.context")
    question_ids_path = pjoin(FLAGS.data_dir, "train.ids.question")
    answer_span_path = pjoin(FLAGS.data_dir, "train.span")
    val_context_ids_path = pjoin(FLAGS.data_dir, "val.ids.context")
    val_question_ids_path = pjoin(FLAGS.data_dir, "val.ids.question")
    val_answer_span_path = pjoin(FLAGS.data_dir, "val.span")

    context_path = pjoin(FLAGS.data_dir, "train.context")
    val_context_path = pjoin(FLAGS.data_dir, "val.context")

    context_ids = initialize_data(context_ids_path)
    question_ids = initialize_data(question_ids_path)
    answer_spans = initialize_data(answer_span_path)
    context = initialize_data(context_path, keep_as_string=True)
    val_context_ids = initialize_data(val_context_ids_path)
    val_question_ids = initialize_data(val_question_ids_path)
    val_answer_spans = initialize_data(val_answer_span_path)
    val_context = initialize_data(val_context_path, keep_as_string=True)

    # Reducing context length to the specified max in FLAGS.output_size
    paragraph_lengths = []
    for i in range(0,len(context_ids)):
        paragraph_lengths.append(len(context_ids[i]))
        context_ids[i] = context_ids[i][:FLAGS.output_size]
        context[i] = context[i][:FLAGS.output_size]
        answer_spans[i] = np.clip(answer_spans[i],0,FLAGS.output_size-1)
    for j in range(0,len(val_context_ids)):
        paragraph_lengths.append(len(val_context_ids[j]))
        val_context_ids[j] = val_context_ids[j][:FLAGS.output_size]
        val_context[j] = val_context[j][:FLAGS.output_size]
        val_answer_spans[j] = np.clip(val_answer_spans[j],0,FLAGS.output_size-1)

    #print ('Par_mean:',np.mean(paragraph_lengths),'Par_std:',np.std(paragraph_lengths))

    par_mean = np.mean(paragraph_lengths)
    par_max_len = max(paragraph_lengths)
    std = np.std(paragraph_lengths)

    train_dataset = [context_ids,question_ids,answer_spans]
    val_dataset = [val_context_ids,val_question_ids,val_answer_spans]
    contexts = [context,val_context]
    dataset = (train_dataset,val_dataset)

    max_ctx_len = max(max(map(len, context_ids)), max(map(len, val_context_ids)))
    max_q_len = max(max(map(len, question_ids)), max(map(len, val_question_ids)))
    
    embeddings = initialize_embeddings(embed_path)
    
    assert len(vocab) == embeddings.shape[0], "Mismatch between embedding shape and vocab length"
    assert embeddings.shape[1] == FLAGS.embedding_size, "Mismatch between embedding shape and FLAGS"
    assert len(context_ids) == len(question_ids) == len(answer_spans), "Mismatch between context, questions, and answer lengths"

    print("Using model type : {}".format(FLAGS.model_type))

    question_encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, name="question_encoder")
    context_encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, name="context_encoder")
    decoder = Decoder(output_size=FLAGS.output_size, name="decoder")

    qa = QASystem(encoder=(question_encoder,context_encoder), 
                  decoder=decoder, 
                  pretrained_embeddings=embeddings,
                  max_ctx_len=max_ctx_len,
                  max_q_len=max_q_len,
                  flags=FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        saver = tf.train.Saver()

        qa.train(sess, saver, dataset, contexts, save_train_dir)

if __name__ == "__main__":
    tf.app.run()
