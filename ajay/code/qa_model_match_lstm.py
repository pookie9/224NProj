from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar, minibatches

from evaluate import exact_match_score, f1_score

from IPython import embed

from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

logging.basicConfig(level=logging.INFO)


### Match LSTM:

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
    """
    Arguments:
        -num_units: hidden state dimensions
        -encoder_output: hidden states to compute attention over
        -scope: lol who knows
    """
    def __init__(self, num_units, encoder_output, scope=None):
        # attn_states is shape (batch_size, N, hid_dim)
        self.attn_states = encoder_output
        super(GRUAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            # compute scores using hs.T * W * ht
            with vs.variable_scope("Attn"):
                # ht is shape (batch_size, hid_dim)
                ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)

                # ht is shape (batch_size, 1, hid_dim)
                ht = tf.expand_dims(ht, axis=1)

            # scores is shape (batch_size, N, 1)
            scores = tf.reduce_sum(self.attn_states * ht, reduction_indices=2, keep_dims=True)

            # do a softmax over the scores
            scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
            scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

            # compute context vector using linear combination of attention states with
            # weights given by attention vector.
            # context is shape (batch_size, hid_dim)
            context = tf.reduce_sum(self.attn_states * scores, reduction_indices=1)

            with vs.variable_scope("AttnConcat"):
                out = tf.nn.tanh(tf.nn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))

            return (out, gru_state)

class LSTMAttnCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Arguments:
        -num_units: hidden state dimensions
        -encoder_output: hidden states to compute attention over
        -scope: lol who knows
    """
    def __init__(self, num_units, encoder_output, scope=None):
        # attn_states is shape (batch_size, N, hid_dim)
        self.attn_states = encoder_output
        super(LSTMAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        lstm_out, lstm_state = super(LSTMAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            # compute scores using hs.T * W * ht
            with vs.variable_scope("Attn"):
                # ht is shape (batch_size, hid_dim)
                ht = tf.nn.rnn_cell._linear(lstm_out, self._num_units, True, 1.0)

                # ht is shape (batch_size, 1, hid_dim)
                ht = tf.expand_dims(ht, axis=1)

            # scores is shape (batch_size, N, 1)
            scores = tf.reduce_sum(self.attn_states * ht, reduction_indices=2, keep_dims=True)

            # do a softmax over the scores
            scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
            scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

            # compute context vector using linear combination of attention states with
            # weights given by attention vector.
            # context is shape (batch_size, hid_dim)
            context = tf.reduce_sum(self.attn_states * scores, reduction_indices=1)

            with vs.variable_scope("AttnConcat"):
                out = tf.nn.tanh(tf.nn.rnn_cell._linear([context, lstm_out], self._num_units, True, 1.0))

            return (out, lstm_state)

class MatchLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Arguments:
        -num_units: hidden state dimensions
        -encoder_output: hidden states to compute attention over
        -scope: lol who knows
    """
    def __init__(self, num_units, encoder_output, scope=None):
        # shape (batch_size, Nq, hid_dim), these are question encodings
        self.attn_states = encoder_output
        self.q_size = int(self.attn_states.get_shape()[1]) # convert from Dimension type to int

        super(MatchLSTMCell, self).__init__(num_units)

    # note: inputs should be paragraph encodings
    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):

            with vs.variable_scope("ParAttn"):
                par_attn = tf.nn.rnn_cell._linear([inputs, state[-1]], self._num_units, True, 1.0)

            with vs.variable_scope("QuesAttn"):
                # hit attention_states with a weight matrix (3d * 2d)
                attn_states_reshaped = tf.reshape(self.attn_states, [-1, self._num_units])
                ques_attn = tf.nn.rnn_cell._linear(attn_states_reshaped, output_size=self._num_units, bias=True)
                ques_attn = tf.reshape(ques_attn, [-1, self.q_size, self._num_units])

                # use expand_dims with broadcasting, this is shape [?, 1, hidden_size] now
                par_attn = tf.expand_dims(par_attn, 1)

                g_i = tf.nn.tanh(ques_attn + par_attn)

            with vs.variable_scope("Scores"):
                g_i_r = tf.reshape(g_i, [-1, self._num_units])
                scores = tf.nn.rnn_cell._linear(g_i_r, output_size=1, bias=True)
                scores = tf.reshape(scores, [-1, self.q_size])
                scores = tf.nn.softmax(scores)

            # do a softmax over the scores
            # scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
            # scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

            # compute context vector using linear combination of attention states with
            # weights given by attention vector.
            # context is shape (batch_size, hid_dim)
            scores = tf.expand_dims(scores, 2)
            context = tf.reduce_sum(g_i * scores, reduction_indices=1)

            z = tf.concat(1, [inputs, context])

        lstm_out, lstm_state = super(MatchLSTMCell, self).__call__(z, state, scope)
        return (lstm_out, lstm_state)


class Encoder(object):
    """
    Arguments:
        -size: dimension of the hidden states
        -vocab_dim: dimension of the embeddings
    """
    def __init__(self, hidden_size, dropout):
        self.hidden_size = hidden_size
        self.dropout = dropout

    def encode(self, inputs, masks, encoder_state_input=None, attention_inputs=None, model_type="gru", bidir=True, name="encoder", reuse=False, concat=True):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :param attention_inputs: (Optional) pass this to compute attention and context
                                    over these encodings
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with tf.variable_scope(name, reuse=reuse):

            ### Define the correct cell type.
            if attention_inputs is None:
                if model_type == "gru":
                    if bidir:
                        fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                        bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                    else:
                        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                elif model_type == "lstm":
                    if bidir:
                        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                    else:
                        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                else:
                    raise Exception('Must specify model type.')
            else:
                # use an attention cell - each cell uses attention to compute context
                # over the @attention_inputs
                if model_type == "gru":
                    if bidir:
                        fw_cell = GRUAttnCell(self.hidden_size, attention_inputs[0])
                        bw_cell = GRUAttnCell(self.hidden_size, attention_inputs[1])
                    else:
                        cell = GRUAttnCell(self.hidden_size, attention_inputs)
                elif model_type == "lstm":
                    if bidir:
                        fw_cell = LSTMAttnCell(self.hidden_size, attention_inputs[0])
                        bw_cell = LSTMAttnCell(self.hidden_size, attention_inputs[1])
                    else:
                        cell = LSTMAttnCell(self.hidden_size, attention_inputs)
                elif model_type == "match":
                    if bidir:
                        fw_cell = MatchLSTMCell(self.hidden_size, attention_inputs[0])
                        bw_cell = MatchLSTMCell(self.hidden_size, attention_inputs[1])
                    else:
                        cell = MatchLSTMCell(self.hidden_size, attention_inputs)
                else:
                    raise Exception('Must specify model type.')

            ### Define correct RNN.
            if bidir:
                if encoder_state_input is None:
                    encoder_state_input = (None, None)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                       sequence_length=masks,
                                                                       dtype=tf.float32,
                                                                       initial_state_fw=encoder_state_input[0],
                                                                       initial_state_bw=encoder_state_input[1])
                # get concatenated stuff
                if concat:
                    if model_type == "gru":
                        final_state = tf.concat(1, final_state)
                        outputs = tf.concat(2, outputs)
                    elif model_type == "lstm" or model_type == "match":
                        # get rid of "c"
                        final_state = tf.concat(1, [final_state[0][-1], final_state[1][-1]])
                        outputs = tf.concat(2, outputs)
                    else:
                        raise Exception('Must specify model type.')
                # add forward and backward hidden states together
                else:
                    outputs = outputs[0] + outputs[1]
                    final_state = final_state[0] + final_state[1]

                return outputs, final_state

            else:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
                                           sequence_length=masks,
                                           dtype=tf.float32,
                                           initial_state=encoder_state_input)
                # get rid of "c"
                if model_type == "lstm" or model_type == "match":
                    final_state = final_state[-1]

                return outputs, final_state


class Decoder(object):
    def __init__(self, hidden_size, output_size, dropout):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

    def decode(self, knowledge_rep, masks):
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

        with vs.variable_scope("decoder"):

            # TODO: use correct masks...since this is bidirectional...

            with vs.variable_scope("answer_start"):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout)
                all_start_probs, _ = tf.nn.dynamic_rnn(cell, knowledge_rep, sequence_length=masks, dtype=tf.float32)

                # P is (batch_size, output_dim, hid_dim), reshape to (batch_size * output_dim, hid_dim)
                all_start_probs_reshaped = tf.reshape(all_start_probs, [-1, self.hidden_size])
                # weights W = (hid_dim, 1)
                start_probs = tf.nn.rnn_cell._linear(all_start_probs_reshaped, output_size=1, bias=True)
                # P is now (batch_size * output_dim, 1), so reshape to get start_probs
                start_probs = tf.reshape(start_probs, [-1, self.output_size])

            with vs.variable_scope("answer_end"):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout)
                all_end_probs, _ = tf.nn.dynamic_rnn(cell, all_start_probs, sequence_length=masks, dtype=tf.float32)

                # do same trick as above on LSTM output to get end_probs
                all_end_probs_reshaped = tf.reshape(all_end_probs, [-1, self.hidden_size])
                end_probs = tf.nn.rnn_cell._linear(all_end_probs_reshaped, output_size=1, bias=True)
                end_probs = tf.reshape(end_probs, [-1, self.output_size])

            # Masking
            bool_masks = tf.cast(tf.sequence_mask(masks, maxlen=self.output_size), tf.float32)
            add_mask = (-1e30 * (1.0 - bool_masks))
            # add_mask = tf.log(bool_masks)
            start_probs = tf.add(start_probs, add_mask)
            end_probs = tf.add(end_probs, add_mask)

        return start_probs, end_probs


class QASystem(object):
    def __init__(self, pretrained_embeddings, flags):
        """
        Initializes your System

        :param args: pass in more arguments as needed
        """
        self.pretrained_embeddings = pretrained_embeddings
        self.flags = flags
        self.h_size = self.flags.state_size
        self.p_size = self.flags.output_size
        self.q_size = self.flags.question_size
        self.embed_size = self.flags.embedding_size

        self.encoder = Encoder(hidden_size=self.h_size,
                               dropout=(1.0 - self.flags.dropout))

        self.decoder = Decoder(hidden_size=self.h_size,
                               output_size=self.p_size,
                               dropout=(1.0 - self.flags.dropout))

        # ==== set up placeholder tokens ========

        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.p_size), name='context_placeholder')
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, self.q_size), name='question_placeholder')
        self.answer_span_placeholder = tf.placeholder(tf.int32, shape=(None, 2), name='answer_span_placeholder')
        self.mask_q_placeholder = tf.placeholder(tf.int32, shape=(None,), name='mask_q_placeholder')
        self.mask_ctx_placeholder = tf.placeholder(tf.int32, shape=(None,), name='mask_ctx_placeholder')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.flags.learning_rate

        self.learning_rate = self.starter_learning_rate

        # learning rate decay
        # self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
        #                                    1000, 0.96, staircase=True)

        self.optimizer = get_optimizer("adam")

        if self.flags.grad_clip:
            # gradient clipping
            self.optimizer = self.optimizer(self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.flags.max_gradient_norm), var)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        else:
            # no gradient clipping
            self.train_op = self.optimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def pad(self, sequence, max_length):
        # assumes sequence is a list of lists of word, pads to the longest "sentence"
        # returns (padded_sequence, mask)
        from qa_data import PAD_ID
        padded_sequence = []
        mask = []
        for sentence in sequence:
            mask.append(len(sentence))
            sentence.extend([PAD_ID] * (max_length - len(sentence)))
            padded_sequence.append(sentence)
        return (padded_sequence, mask)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        # first do a basic LSTM encoding of questions and contexts
        question_states, final_question_state = self.encoder.encode(self.question_embeddings,
                                                                    self.mask_q_placeholder,
                                                                    attention_inputs=None,
                                                                    model_type=self.flags.model_type,
                                                                    bidir=False,
                                                                    name="question_encoder",
                                                                    reuse=False,
                                                                    concat=False)

        # TODO: try with and without attention here
        ctx_states, final_ctx_state = self.encoder.encode(self.context_embeddings,
                                                          self.mask_ctx_placeholder,
                                                          attention_inputs=None,
                                                          model_type=self.flags.model_type,
                                                          bidir=False,
                                                          name="context_encoder",
                                                          reuse=False,
                                                          concat=False)

        # match LSTM layer
        # TODO: make this bidirectional
        match_states, final_match_state = self.encoder.encode(self.context_embeddings,
                                                              self.mask_ctx_placeholder,
                                                              encoder_state_input=None, #final_question_state,
                                                              attention_inputs=question_states,
                                                              model_type="match",
                                                              bidir=False,
                                                              name="match_encoder",
                                                              reuse=False,
                                                              concat=False)


        # decoder takes encoded representation to probability dists over start / end index
        self.start_probs, self.end_probs = self.decoder.decode(knowledge_rep=match_states,
                                                               masks=self.mask_ctx_placeholder)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_probs, self.answer_span_placeholder[:, 0])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_probs,self.answer_span_placeholder[:, 1]))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings, name='embedding', dtype=tf.float32,
                                     trainable=False)  # only learn one common embedding

            self.question_embeddings = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

            self.context_embeddings = tf.nn.embedding_lookup(embeddings, self.context_placeholder)

    def optimize(self, session, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout
        input_feed[self.answer_span_placeholder] = answer_span_batch

        output_feed = [self.train_op, self.loss]

        _, loss = session.run(output_feed, input_feed)

        return loss

    def test(self, session, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout
        input_feed[self.answer_span_placeholder] = answer_span_batch

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def decode(self, session, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout

        output_feed = [self.start_probs, self.end_probs]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, data):

        yp_lst = []
        yp2_lst = []
        prog_train = Progbar(target=1 + int(len(data[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(data, self.flags.batch_size, shuffle=False)):
            yp, yp2 = self.decode(session, *batch)
            yp_lst.append(yp)
            yp2_lst.append(yp2)
            prog_train.update(i + 1, [("computing F1...", 1)])
        print("")
        yp_all = np.concatenate(yp_lst, axis=0)
        yp2_all = np.concatenate(yp2_lst, axis=0)

        a_s = np.argmax(yp_all, axis=1)
        a_e = np.argmax(yp2_all, axis=1)

        return (a_s, a_e)

    def validate(self, sess, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """

        return self.test(session=sess,
                         context_batch=context_batch,
                         question_batch=question_batch,
                         answer_span_batch=answer_span_batch,
                         mask_ctx_batch=mask_ctx_batch,
                         mask_q_batch=mask_q_batch)

    def evaluate_answer(self, session, dataset, context, sample=100, log=False, eval_set='train'):
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

        if sample is None:
            sampled = dataset
            sample = len(dataset[0])
        else:
            # np.random.seed(0)
            inds = np.random.choice(len(dataset[0]), sample)
            sampled = [elem[inds] for elem in dataset]
            context = [context[i] for i in inds]

        a_s, a_e = self.answer(session, sampled)

        context_ids, question_ids, answer_spans, ctx_mask, q_mask = sampled

        f1 = []
        em = []
        # #embed()
        for i in range(len(sampled[0])):
            pred_words = ' '.join(context[i][a_s[i]:a_e[i] + 1])
            actual_words = ' '.join(context[i][answer_spans[i][0]:answer_spans[i][1] + 1])
            f1.append(f1_score(pred_words, actual_words))
            cur_em = exact_match_score(pred_words, actual_words)
            em.append(float(cur_em))

        if log:
            logging.info("{},F1: {}, EM: {}, for {} samples".format(eval_set, np.mean(f1), np.mean(em), sample))

        return np.mean(f1), np.mean(em)

    ### Imported from NERModel
    def run_epoch(self, sess, train_set, val_set, train_context, val_context):
        prog_train = Progbar(target=1 + int(len(train_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(train_set, self.flags.batch_size)):
            loss = self.optimize(sess, *batch)
            prog_train.update(i + 1, [("train loss", loss)])
        print("")

        # if self.flags.debug == 0:
        prog_val = Progbar(target=1 + int(len(val_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(val_set, self.flags.batch_size)):
            val_loss = self.validate(sess, *batch)
            prog_val.update(i + 1, [("val loss", val_loss)])
        print("")

        self.evaluate_answer(session=sess,
                             dataset=train_set,
                             context=train_context,
                             sample=len(val_set[0]),
                             log=True,
                             eval_set="-Epoch TRAIN-")

        self.evaluate_answer(session=sess,
                             dataset=val_set,
                             context=val_context,
                             sample=None,
                             log=True,
                             eval_set="-Epoch VAL-")
        # train_f1, train_em = self.evaluate_answer(sess,train_set, context=context[0], sample=100, log=True, eval_set="-Epoch TRAIN-")
        # val_f1, val_em = self.evaluate_answer(sess,val_set, context=context[1], sample=100, log=True, eval_set="-Epoch VAL-")

    def train(self, session, dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
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

        context_ids, question_ids, answer_spans, ctx_mask, q_mask, train_context = dataset
        train_dataset = [context_ids, question_ids, answer_spans, ctx_mask, q_mask]
        # train_dataset = [np.array(col) for col in zip(*train_dataset)]

        val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask, val_context = val_dataset
        val_dataset = [val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask]
        # val_dataset = [np.array(col) for col in zip(*val_dataset)]

        num_epochs = self.flags.epochs

        if self.flags.debug:
            train_dataset = [elem[:self.flags.batch_size] for elem in train_dataset]
            val_dataset = [elem[:self.flags.batch_size] for elem in val_dataset]
            num_epochs = 100
            # num_epochs = 1

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset,
                           val_set=val_dataset,
                           train_context=train_context,
                           val_context=val_context)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir)

    def minibatches(self, data, batch_size, shuffle=True):
        num_data = len(data[0])
        context_ids, question_ids, answer_spans, ctx_mask, q_mask = data
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, num_data, batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
            yield [context_ids[minibatch_indices], question_ids[minibatch_indices], answer_spans[minibatch_indices],
                   ctx_mask[minibatch_indices], q_mask[minibatch_indices]]





