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


### Baseline 2:

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn
class Encoder(object):
    """
    Arguments:
        -size: dimension of the hidden states
    """
    def __init__(self, hidden_size, dropout):
        self.hidden_size = hidden_size
        self.dropout = dropout

    def encode(self, inputs, masks, attention_inputs=None, initial_state=(None, None),model_type="gru", name="encoder", reuse=False):
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

            # Define the correct cell type.
            if attention_inputs is None:
                if model_type == "gru":
                    cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                elif model_type == "lstm":
                    cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                else:
                    raise Exception('Must specify model type.')
            else:
                assert(model_type == "lstm")
                cell = LSTMAttnCell(self.hidden_size, attention_inputs)
            
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)

            # TODO: should we create two different LSTM cells here, one for fw, one for bw?
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs,
                                                                   sequence_length=masks,
                                                                   initial_state_fw=initial_state[0],
                                                                   initial_state_bw=initial_state[1],
                                                                   dtype=tf.float32)

            # TODO: try concatenating instead of adding
            # final_outputs = tf.concat(2, outputs)

            # add forward and backward hidden states together
            final_outputs = outputs[0] + outputs[1]

            return final_outputs, final_state

class Decoder(object):
    def __init__(self, hidden_size, output_size, dropout):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

    def decode(self, knowledge_rep, masks, initial_state=(None,None)):
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
            #initial_state=(None,None)
            with vs.variable_scope("answer_start"):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                
                start_states, start_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, knowledge_rep,
                                                                                  sequence_length=masks,
                                                                                  initial_state_fw=initial_state[0],
                                                                                  initial_state_bw=initial_state[1],
                                                                                  dtype=tf.float32)
                start_states=start_states[0]+start_states[1]
                
                #start_states, start_final_state = tf.nn.dynamic_rnn(cell, knowledge_rep, sequence_length=masks, dtype=tf.float32)
                start_states_reshaped = tf.reshape(start_states, [-1, self.hidden_size])
                start_probs = tf.nn.rnn_cell._linear(start_states_reshaped, output_size=1, bias=True)
                start_probs = tf.reshape(start_probs, [-1, self.output_size])

                
            with vs.variable_scope("answer_end"):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                end_states, end_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, knowledge_rep,
                                                                                  sequence_length=masks,
                                                                                  initial_state_fw=start_final_state[0],
                                                                                  initial_state_bw=start_final_state[1],
                                                                                  dtype=tf.float32)

                end_states=end_states[0]+end_states[1]
                #end_states, end_final_state = tf.nn.dynamic_rnn(cell, knowledge_rep,initial_state=start_final_state, sequence_length=masks, dtype=tf.float32)
                end_states_reshaped = tf.reshape(end_states, [-1, self.hidden_size])
                end_probs = tf.nn.rnn_cell._linear(end_states_reshaped, output_size=1, bias=True)
                end_probs = tf.reshape(end_probs, [-1, self.output_size])

            # Masking
            bool_masks = tf.cast(tf.sequence_mask(masks, maxlen=self.output_size), tf.float32)
            add_mask = (-1e30 * (1.0 - bool_masks))
            #add_mask = tf.log(bool_masks)
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
        self.dropout = self.flags.dropout
        self.encoder = Encoder(hidden_size=self.h_size,
                               dropout=(1.0-self.flags.dropout))

        self.decoder = Decoder(hidden_size=self.h_size,
                               output_size=self.p_size,
                               dropout=(1.0-self.flags.dropout))


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

        self.saver=tf.train.Saver()
        
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

    def mixer(self, q_states, ctx_states):
        # Compute attention of each context word representation with respect to the question final hidden states


        with vs.variable_scope("mixer"):
            # to calculate affinity matrix, need P * Q^T
            # P is shape (?, max_p_len, hid_size), Q is shape (?, max_q_len, hid_size)
            # A will be shape (?, max_p_len, max_q_len)
            A = tf.nn.softmax(batch_matmul(ctx_states, tf.transpose(q_states, perm=[0, 2, 1])))

            # C_P is shape (?, max_p_len, hid_size) = lin. comb. of weights from A over question states
            # These are the context vectors.
            C_P = batch_matmul(A, q_states)

            # First, reshape both C_P and P to make them 2-D
            C_P = tf.reshape(C_P, [-1, self.h_size])
            P = tf.reshape(ctx_states, [-1, self.h_size])

            # Next, use a linear layer to concatenate them along hid_size, and apply a weight matrix
            P_final = tf.nn.rnn_cell._linear([C_P, P], output_size=self.h_size, bias=True)

            # Finally, reshape the output to the correct shape
            P_final = tf.reshape(P_final, [-1, self.p_size, self.h_size])

            return P_final

    # Compute relevancy matrix to determine how relevant each word in paragraph is to each word in question.
    # Then filter the paragraph word embeddings on relevancy
    def filter(self, Q, P):
        with vs.variable_scope("filter"):
            # Q is (batch_size, q_size, embed_size)
            # P is (batch_size, p_size, embed_size)

            # normalize all embeddings to unit norm so that dot product is cosine similarity
            Qn = tf.nn.l2_normalize(Q, dim=2)
            Pn = tf.nn.l2_normalize(P, dim=2)

            # R is shape (batch_size, q_size, p_size), R_ij = q_i dot p_j
            R = batch_matmul(Qn, tf.transpose(Pn, perm=[0, 2, 1]))

            # collect maximum relevancy over the questions per paragraph word, shape (batch_size, p_size)
            r = tf.reduce_max(R, axis=1)
            r = tf.expand_dims(r, axis=2)  # shape (batch_size, p_size, 1) to take advantage of broadcasting

            # re-weight paragraph embeddings with relevancy scores
            P_filtered = P * r

            return P_filtered


    # Does coattention encoding
    def coattention(self, P, Q, masks):
        # P is shape (batch_size, p_size, hid_size)
        # Q is shape (batch_size, q_size, hid_size)

        P_t = tf.transpose(P, perm=[0, 2, 1])
        Q_t = tf.transpose(Q, perm=[0, 2, 1])

        # compute affinity matrix, shape (batch_size, p_size, q_size)
        L = tf.batch_matmul(P, Q_t)

        # attention weights
        AQ = tf.nn.softmax(L,dim=1)
        AD = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]),dim=1)

        # attention contexts
        CQ = tf.batch_matmul(P_t, AQ)
        # mapf = lambda x : tf.batch_matmul(x, AQ)
        # CQ = tf.map_fn(mapf, P_t) # use map to multiply 3-d tensor by 2-d

        # CQ should be shape (batch_size, hid_size, q_size + 1)

        # contexts
        contexts = tf.concat(1, [Q_t, CQ])  # shape (batch_size, 2 * hid_size, q_size)
        # mapf = lambda x : tf.batch_matmul(x, AD)
        # CD = tf.map_fn(mapf, contexts) # shape (batch_size, 2 * hid_size, p_size)
        CD = tf.batch_matmul(contexts, AD)

        # shape (batch_size, p_size, 4 * hid_size)
        lstm_inputs = tf.transpose(tf.concat(1, [P_t, CD]), perm=[0, 2, 1])

        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.h_size)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.h_size)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout)
        # cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.dropout)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout)
        # cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.dropout)
        all_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, lstm_inputs,
                                                        sequence_length=masks,
                                                        dtype=tf.float32)

        U = tf.concat(2, all_states)  # concatenate fwd and bck states
        U = U[:, :self.p_size, :]  # cut off extra dimension from sentinel

        return U


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        # TODO: can use filter here
        P_filtered = self.context_embeddings
        # # first filter paragraph embeddings on relevancy
        #P_filtered = self.filter(Q=self.question_embeddings,
        #                         P=self.context_embeddings)

        # note that we reuse the SAME encoder for both question and paragraph
        question_states, final_question_state = self.encoder.encode(self.question_embeddings,
                                                                    self.mask_q_placeholder,
                                                                    attention_inputs=None,
                                                                    model_type=self.flags.model_type,
                                                                    reuse=False,name='q_encoder')

        # TODO: can update question encodings with tanh layer here
        # question_states = tf.reshape(question_states, [-1, self.h_size])
        # question_states = tf.nn.rnn_cell._linear(question_states, output_size=self.h_size, bias=True)
        # question_states = tf.nn.tanh(tf.reshape(question_states, [-1, self.q_size + 1, self.h_size]))

        # TODO: can use attention over the question here too (set attention_inputs)
        ctx_states, final_ctx_state = self.encoder.encode(P_filtered,
                                                          self.mask_ctx_placeholder,
                                                          attention_inputs=None,
                                                          # attention_inputs=question_states
                                                          initial_state=final_question_state,
                                                          model_type=self.flags.model_type,
                                                          reuse=False, name='ctx_encoder')

        # TODO: can use coattention here
        #feed_states = self.coattention(P=ctx_states,
        #                     Q=question_states,
        #                     masks=self.mask_ctx_placeholder)

        feed_states = self.mixer(q_states=question_states,
                                 ctx_states=ctx_states)

        # decoder takes encoded representation to probability dists over start / end index
        self.start_probs, self.end_probs = self.decoder.decode(knowledge_rep=feed_states,
                                                               masks=self.mask_ctx_placeholder,
                                                               initial_state=final_question_state)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_probs, self.answer_span_placeholder[:, 0])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_probs, self.answer_span_placeholder[:, 1]))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings, name='embedding', dtype=tf.float32, trainable=False) #only learn one common embedding
            self.question_embeddings = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            self.context_embeddings = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            """
            q_mean,q_var = tf.nn.moments(question_embeddings, axes=[2], shift=None, name=None, keep_dims=True)
            ctx_mean,ctx_var = tf.nn.moments(context_embeddings, axes=[2], shift=None, name=None, keep_dims=True)



            # q_scale = tf.Variable(tf.zeros,name='q_scale',dtype=tf.float32,trainable=True)
            # ctx_scale = tf.Variable(1.0,name='ctx_scale',dtype=tf.float32,trainable=True)
            # q_offset = tf.Variable(1.0,name='q_offset',dtype=tf.float32,trainable=True)
            # ctx_offset = tf.Variable(1.0,name='ctx_offset',dtype=tf.float32,trainable=True) 


            q_scale = tf.get_variable("q_scale", shape=[self.q_size,self.embed_size],initializer=tf.ones_initializer(tf.float32))
            ctx_scale = tf.get_variable("ctx_scale", shape=[self.p_size,self.embed_size],initializer=tf.ones_initializer(tf.float32))
            q_offset = tf.get_variable("q_offset", shape=[self.q_size,self.embed_size],initializer=tf.ones_initializer(tf.float32))
            ctx_offset = tf.get_variable("ctx_offset", shape=[self.p_size,self.embed_size],initializer=tf.ones_initializer(tf.float32))


            self.question_embeddings = tf.nn.batch_normalization(question_embeddings, q_mean, q_var, q_offset,q_scale, variance_epsilon=0.0000001)
            self.context_embeddings = tf.nn.batch_normalization(context_embeddings, ctx_mean, ctx_var, ctx_offset, ctx_scale, variance_epsilon=0.0000001)
            """

           


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
            prog_train.update(i + 1, [("Answering Questions....",0.0)])
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
            #np.random.seed(0)
            inds = np.random.choice(len(dataset[0]), sample)
            
            sampled = [elem[inds] for elem in dataset]
            context=[context[i] for i in inds]
            
        a_s, a_e = self.answer(session, sampled)

        context_ids, question_ids, answer_spans, ctx_mask, q_mask = sampled

        f1 = []
        em = []
        # #embed()
        for i in range(len(sampled[0])):
            pred_words=' '.join(context[i][a_s[i]:a_e[i]+1])
            actual_words=' '.join(context[i][answer_spans[i][0]:answer_spans[i][1]+1])
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

        #if self.flags.debug == 0:
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
        #self.saver=saver
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        context_ids, question_ids, answer_spans, ctx_mask ,q_mask, train_context = dataset
        train_dataset = [context_ids, question_ids, answer_spans, ctx_mask ,q_mask]
        #train_dataset = [np.array(col) for col in zip(*train_dataset)]

        val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask, val_context = val_dataset
        val_dataset = [val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask]
        #val_dataset = [np.array(col) for col in zip(*val_dataset)]
        
        num_epochs = self.flags.epochs

        if self.flags.debug:
            train_dataset = [elem[:self.flags.batch_size*1] for elem in train_dataset]
            val_dataset = [elem[:self.flags.batch_size] for elem in val_dataset]
            num_epochs = 100
            #num_epochs = 1

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset, 
                           val_set=val_dataset,
                           train_context=train_context,
                           val_context=val_context)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")

    def minibatches(self, data, batch_size, shuffle=True):
        num_data = len(data[0])
        context_ids, question_ids, answer_spans, ctx_mask, q_mask = data
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, num_data, batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
            yield [context_ids[minibatch_indices], question_ids[minibatch_indices], answer_spans[minibatch_indices], ctx_mask[minibatch_indices], q_mask[minibatch_indices]]





