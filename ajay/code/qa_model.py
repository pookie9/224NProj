from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell()
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell()

        # note: inputs need to be zero-padded to max_time length
        # note: masks must contain actual length of every input in the batch
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                                   sequence_length=masks, 
                                                                   initial_state_fw=encoder_state_input, 
                                                                   initial_state_bw=encoder_state_input)

        # TODO: see shape of output_states
        # concatenate hidden vectors of both directions together
        encoded_outputs = tf.concat(outputs, 2)

        # return all hidden states and the final hidden state
        return encoded_outputs, encoded_outputs[:, -1, :]


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        return

class QASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, max_ctx_len, max_q_len):
        """
        Initializes your System

        :param encoder: tuple of 2 encoders that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.pretrained_embeddings = pretrained_embeddings
        self.question_encoder, self.context_encoder = encoder # unpack tuple of encoders
        self.decoder = decoder
        self.max_ctx_len = max_ctx_len
        self.max_q_len = max_q_len
        self.embed_size = encoder[0].vocab_dim
        # ==== set up placeholder tokens ========

        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, max_ctx_len), name='context_placeholder')
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_q_len), name='question_placeholder')
        self.answer_span_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length), name='answer_span_placeholder')
        self.mask_q_placeholder = tf.placeholder(tf.int32, shape=(None,), name='mask_q_placeholder')
        self.mask_ctx_placeholder = tf.placeholder(tf.int32, shape=(None,), name='mask_ctx_placeholder')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass

    def pad(self, sequence):
        # assumes sequence is a list of lists of word, pads to the longest "sentence"
        # returns (padded_sequence, mask)
        from qa_data import PAD_ID
        max_length = max(map(len, sequence))
        padded_sequence = []
        mask = []
        for sentence in sequence:
            mask.append(len(sentence))
            sentence.extend([PAD_ID] * (max_length-len(sequence)))
            padded_sentence.append(sentence)
        return (padded_sequence, mask)

    def setup_attention_vector(self, context_vectors, question_rep):
        #context_vectors is a list of the hidden states of the context
        #question_rep are the final forward and backward states of the encoder for the question concatenated
        #Does part 3 in original handout
        W = tf.get_variable("W", shape=[context_vectors[0].get_shape()[0], question_rep.get_shape()[0]],
                                 initializer=tf.contrib.layers.xavier_initializer())
        #attention = [tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(ctx), W), question_rep)) for ctx in context_vectors]
        

        # TODO: ask TA how to handle batch size stuff here...
        attention = tf.nn.softmax(tf.sum(tf.matmul(tf.matmul(question_rep, W), context_vectors)))
        return attention

    def concat_most_aligned(self, question_states, cur_ctx):
        #Does part 4 in original handout
        #question_states is a list of all of the hidden states for the question, cur_ctx is the current context word
        #returns a concatenation of [cur_ctx, q*] where q* is the most aligned question word
        U = tf.get_variable("U", shape=[cur_ctx.get_shape()[0], question_states[0].get_shape()[0]],
                                 initializer=tf.contrib.layers.xavier_initializer())#maybe need to add reuse variable?
        attention = [tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(cur_ctx), W), q)) for q in question_states]
        most_aligned = (0.0, None)

        # TODO: change this to completely use tensorflow functions (like argmax)
        for i in range(len(attention)):
            if attention[i] > most_aligned:
                most_aligned = (attention[i], question_states[i])
        return tf.concat([cur_ctx,most_aligned[0]], 1)


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question_states, question_rep = self.question_encoder.encode(self.question_placeholder, self.mask_q_placeholder, None)
        ctx_states, ctx_rep = self.context_encoder.encode(self.context_placeholder, self.mask_ctx_placeholder, None)
        attention = setup_attention_vector(question_rep, ctx_states)
        weighted_ctx = tf.matmul(self.question_placeholder, attention)#(hidden_size x max_ctx_len) (max_ctx_len x 1)=>(hidden_size x 1)
        
        # TODO: how to do stuff like packing operations together
        new_ctx = [self.concat_most_aligned(question_states, ctx) for ctx in ctx_states]

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding = tf.Variable(self.pretrained_embeddings,name='embedding') #only learn one common embedding

            #context_embedding=tf.Variable(self.pretrained_embeddings,name="context_embedding")
            #question_embedding=tf.Variable(self.pretrained_embeddings,name="question_embedding")

            context_embeddings = tf.nn.embedding_lookup(embedding, self.context_placeholder)
            self.context_embeddings = tf.reshape(embeddings, [-1, self.max_ctx_len, self.embed_size])

            question_embeddings = tf.nn.embedding_lookup(embedding, self.question_placeholder)
            self.question_embeddings = tf.reshape(embeddings, [-1, self.max_q_len, self.embed_size])



    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        mask = [None, None]
        dataset[0], mask[0] = self.pad(dataset[0]) #context_ids
        dataset[1], mask[1] = self.pad(dataset[1]) #question_ids
        for i in range(1,len(dataset[0])):
            assert len(dataset[0][i]) == len(dataset[0][i - 1]), "Incorrectly padded context"
            assert len(dataset[1][i]) == len(dataset[1][i - 1]), "Incorrectly padded question"
