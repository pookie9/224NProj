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

#from IPython import embed

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

#The DropoutWrapper class isn't working on my tensorflow,https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
class DropoutCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,cell,keep_probs):
        self.keep_probs=None
        self._cell=cell
    def state_size(self):
        return self._cell.state_size
    def output_size(self):
        return self._cell.output_size
    def __call__(self, inputs,state,scope=None):
        inputs=tf.nn.dropout(inputs,self.keep_probs)
        output,new_state=self._cell(inputs,state,scope)
        output=tf.nn.dropout(output,self.keep_probs)
        return output,new_state

class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
    """
    Arguments:
        -num_units: hidden state dimensions
        -encoder_output: hidden states to compute attention over
        -scope: lol who knows
    """
    def __init__(self, num_units, encoder_output, scope=None):
        self.attn_states = encoder_output
        super(GRUAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with tf.variable_scope(scope or type(self).__name__):
            # compute scores using hs.T * W * ht
            with tf.variable_scope("Attn"):
                # ht is shape (batch_size, hid_dim)
                W_score = tf.get_variable("W_score", shape=(self._num_units, self._num_units),
                                          initializer=tf.contrib.layers.xavier_initializer())
                b_score = tf.get_variable("b_score", shape=(self._num_units))
                ht = tf.matmul(gru_out, W_score) + b_score

                #ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)

                # ht is shape (batch_size, 1, hid_dim)
                ht = tf.expand_dims(ht, axis=1)

            # scores is shape (batch_size, N, 1)
            scores = tf.reduce_sum(self.attn_states * ht, reduction_indices=2, keep_dims=True)

            # compute context vector using linear combination of attention states with
            # weights given by attention vector.
            # context is shape (batch_size, hid_dim)
            context = tf.reduce_sum(self.attn_states * scores, reduction_indices=1)

            with tf.variable_scope("AttnConcat"):
                W_c = tf.get_variable("W_c", shape=(2 * self._num_units, self._num_units),
                                          initializer=tf.contrib.layers.xavier_initializer())
                b_c = tf.get_variable("b_c", shape=(self._num_units))

                # print(context.get_shape())
                # print(gru_out.get_shape())

                concat_vec = tf.concat(1, [context, gru_out])

                out = tf.nn.relu(tf.matmul(concat_vec, W_c) + b_c)

            return (out, out)

class LSTMAttnCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, encoder_output, scope=None):
        pass

class Encoder(object):
    """
    Arguments:
        -size: dimension of the hidden states
        -vocab_dim: dimension of the embeddings
    """
    def __init__(self, size, vocab_dim, name):
        self.size = size
        self.vocab_dim = vocab_dim
        self.name = name

    def encode(self, inputs, masks, encoder_state_input=None, attention_inputs=None, model_type="gru",dropout=0.0):
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
        with tf.variable_scope(self.name):

            # Define the correct cell type.
            if attention_inputs is None:
                if model_type == "gru":
                    cell = tf.nn.rnn_cell.GRUCell(self.size)
                elif model_type == "lstm":
                    cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
                else:
                    raise Exception('Must specify model type.')
            else:
                # use an attention cell - each cell uses attention to compute context
                # over the @attention_inputs
                if model_type == "gru":
                    cell = GRUAttnCell(self.size, attention_inputs)
                elif model_type == "lstm":
                    cell = LSTMAttnCell(self.size, attention_inputs)
                else:
                    raise Exception('Must specify model type.')                
            #cell=DropoutCell(cell,1.0-dropout)
            #outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,sequence_length=masks,dtype=tf.float32,initial_state=encoder_state_input)
            outputs, final_state = tf.nn.dynamic_bidirectional_rnn(cell, inputs,sequence_length=masks,dtype=tf.float32,initial_state=encoder_state_input)
            
        return outputs, final_state

        # cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        # cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)


        # # TODO: see shape of output_states
        # # concatenate hidden vectors of both directions together
        # encoded_outputs = tf.concat(outputs, 2)

        # # return all hidden states and the final hidden state
        # return encoded_outputs, encoded_outputs[:, -1, :]


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, model_type="gru"):
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

        if model_type == "gru":
            pass
        elif model_type == "lstm":
            # take 2nd part of state params, since that corresponds to hidden state h
            knowledge_rep = knowledge_rep[-1]
        else:
            raise Exception('Must specify model type.')

        input_size = knowledge_rep.get_shape()[-1]
        W_start = tf.get_variable("W_start", shape=(input_size, self.output_size),
                initializer=tf.contrib.layers.xavier_initializer())
        b_start = tf.get_variable("b_start", shape=(self.output_size))

        W_end = tf.get_variable("W_end", shape=(input_size, self.output_size),
                initializer=tf.contrib.layers.xavier_initializer())
        b_end = tf.get_variable("b_end", shape=(self.output_size))

        start_probs = tf.matmul(knowledge_rep, W_start) + b_start
        end_probs = tf.matmul(knowledge_rep, W_end) + b_end

        return start_probs, end_probs

class QASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, max_ctx_len, max_q_len, flags):
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
        self.flags = flags
        # ==== set up placeholder tokens ========

        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_ctx_len), name='context_placeholder')
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_q_len), name='question_placeholder')
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
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                           100000, 0.96, staircase=True)
        self.optimizer = get_optimizer("adam")
        self.optimizer=self.optimizer(self.learning_rate)
        gvs=self.optimizer.compute_gradients(self.loss)
        capped_gvs=[(tf.clip_by_value(grad,-self.flags.max_gradient_norm, self.flags.max_gradient_norm),var)for grad,var in gvs]
        self.train_op=self.optimizer.apply_gradients(capped_gvs)#NOTE: I don't specify to minimize anywhere...Not sure if I should...
        #self.train_op = self.optimizer(self.learning_rate).minimize(self.loss)

    def pad(self, sequence, max_length):
        # assumes sequence is a list of lists of word, pads to the longest "sentence"
        # returns (padded_sequence, mask)
        from qa_data import PAD_ID
        #max_length = max(map(len, sequence))
        #print('max_length : {}'.format(max_length))
        padded_sequence = []
        mask = []
        for sentence in sequence:
            mask.append(len(sentence))
            sentence.extend([PAD_ID] * (max_length - len(sentence)))
            padded_sequence.append(sentence)
        return (padded_sequence, mask)

    # def setup_attention_vector(self, context_vectors, question_rep):
    #     #context_vectors is a list of the hidden states of the context
    #     #question_rep are the final forward and backward states of the encoder for the question concatenated
    #     #Does part 3 in original handout
    #     W = tf.get_variable("W", shape=[context_vectors[0].get_shape()[0], question_rep.get_shape()[0]],
    #                              initializer=tf.contrib.layers.xavier_initializer())
    #     #attention = [tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(ctx), W), question_rep)) for ctx in context_vectors]
        

    #     # TODO: ask TA how to handle batch size stuff here...
    #     attention = tf.nn.softmax(tf.sum(tf.matmul(tf.matmul(question_rep, W), context_vectors)))
    #     return attention

    # def concat_most_aligned(self, question_states, cur_ctx):
    #     #Does part 4 in original handout
    #     #question_states is a list of all of the hidden states for the question, cur_ctx is the current context word
    #     #returns a concatenation of [cur_ctx, q*] where q* is the most aligned question word
    #     U = tf.get_variable("U", shape=[cur_ctx.get_shape()[0], question_states[0].get_shape()[0]],
    #                              initializer=tf.contrib.layers.xavier_initializer())#maybe need to add reuse variable?
    #     attention = [tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(cur_ctx), W), q)) for q in question_states]
    #     most_aligned = (0.0, None)

    #     # TODO: change this to completely use tensorflow functions (like argmax)
    #     for i in range(len(attention)):
    #         if attention[i] > most_aligned:
    #             most_aligned = (attention[i], question_states[i])
    #     return tf.concat([cur_ctx,most_aligned[0]], 1)


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        # simple encoder stuff here
        question_states, final_question_state = self.question_encoder.encode(self.question_embeddings, self.mask_q_placeholder, 
                                                                             encoder_state_input=None, 
                                                                             attention_inputs=None, 
                                                                             model_type=self.flags.model_type,dropout=self.flags.dropout)
        ctx_states, final_ctx_state = self.context_encoder.encode(self.context_embeddings, self.mask_ctx_placeholder, 
                                                                             encoder_state_input=None,#final_question_state, 
                                                                             attention_inputs=question_states,
                                                                             model_type=self.flags.model_type,dropout=self.flags.dropout)

        # decoder takes encoded representation to probability dists over start / end index
        self.start_probs, self.end_probs = self.decoder.decode(final_ctx_state)

        # TODO: put predictions here?




        # TODO: is this correct for the baseline?
        # question_states, question_rep = self.question_encoder.encode(self.question_placeholder, self.mask_q_placeholder, None)
        # ctx_states, ctx_rep = self.context_encoder.encode(self.context_placeholder, self.mask_ctx_placeholder, None)
        # attention = setup_attention_vector(question_rep, ctx_states)
        # weighted_ctx = tf.matmul(self.question_placeholder, attention)#(hidden_size x max_ctx_len) (max_ctx_len x 1)=>(hidden_size x 1)
        
        # TODO: how to do stuff like packing operations together
        #new_ctx = [self.concat_most_aligned(question_states, ctx) for ctx in ctx_states]

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_probs, self.answer_span_placeholder[:, 0])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_probs, self.answer_span_placeholder[:, 1]))
            #pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = tf.Variable(self.pretrained_embeddings, name='embedding', dtype=tf.float32) #only learn one common embedding

            question_embeddings = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            self.question_embeddings = tf.reshape(question_embeddings, [-1, self.max_q_len, self.embed_size])

            context_embeddings = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            self.context_embeddings = tf.reshape(context_embeddings, [-1, self.max_ctx_len, self.embed_size])


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

        # TODO: compute cost for validation set, tune hyperparameters

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

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        
        #mask_ctx_batch=np.reshape(mask_ctx_batch,(context_batch.shape[0]))
        #mask_q_batch=np.reshape(mask_q_batch,(context_batch.shape[0]))
        

        input_feed[self.context_placeholder] = np.array(map(np.array,context_batch))
        input_feed[self.question_placeholder] = np.array(map(np.array,question_batch))
        input_feed[self.mask_ctx_placeholder] = np.array(map(np.array,mask_ctx_batch))
        input_feed[self.mask_q_placeholder] = np.array(mask_q_batch)
        input_feed[self.dropout_placeholder] = self.flags.dropout


        output_feed = [self.start_probs, self.end_probs]

        
        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, data):

        data = np.array(data).T
        yp, yp2 = self.decode(session, *data)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

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

        return self.test(sess, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch)

    def evaluate_answer(self, session, dataset, context, sample=100, log=False):
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

        # iterate over dataset, calling answer method

        # TODO: do we have to loop here over batches?
        # TODO: be explicit about structure of dataset input for all of these functions.


        # idx = np.random.randint(100, size=2)
        # sampled = dataset[idx]

        if sample is None:
            sampled = dataset
        else:
            #np.random.seed(0)
            sampled = dataset[np.random.choice(dataset.shape[0], sample)]

        a_s, a_e = self.answer(session, sampled)

        f1=[]
        em=[]
        #embed()
        sampled = sampled.T
        for i in range(len(sampled[0])):
            pred_words=' '.join(context[i][a_s[i]:a_e[i]+1])
            actual_words=' '.join(context[i][sampled[2][i][0]:sampled[2][i][1]+1])
            # print('I:',i)
            # print ("INDICES",a_s[i],a_e[i])
            # print ("PRED_WORDS:",pred_words)
            # print ("ACTUAL WORD",actual_words)
            f1.append(f1_score(pred_words,actual_words))
            # print('F1:',f1)
            em.append(exact_match_score(pred_words,actual_words))
            # print('EM:',em)
            # print (" ")

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(np.mean(f1), None , sample))
        f1=sum(f1)/len(f1)
        em=sum(em)/len(em)
        return f1, em

    ### Imported from NERModel
    def run_epoch(self, sess, train_examples, dev_set, context):
        prog_train = Progbar(target=1 + int(len(train_examples) / self.flags.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.flags.batch_size)):
            loss = self.optimize(sess, *batch)
            prog_train.update(i + 1, [("train loss", loss)])
        print("")

        prog_val = Progbar(target=1 + int(len(dev_set) / self.flags.batch_size))
        for i, batch in enumerate(minibatches(dev_set, self.flags.batch_size)):
            break
            val_loss = self.validate(sess, *batch)
            prog_val.update(i + 1, [("val loss", val_loss)])
            # prog_val.update(i + 1, [("val f1", val_f1)])
            # prog_val.update(i + 1, [("val em", val_em)])
        val_f1, val_em = self.evaluate_answer(sess,train_examples, context=context, sample=100, log=True)#CHANGE HERE
        #print("validation F1 : {}".format(np.mean(val_f1)))


    def train(self, session, saver, dataset, contexts, train_dir):
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

        train_dataset, val_dataset = dataset
        train_mask = [None, None]
        val_mask = [None, None]
        train_dataset[0], train_mask[0] = self.pad(train_dataset[0], self.max_ctx_len) #train_context_ids
        train_dataset[1], train_mask[1] = self.pad(train_dataset[1], self.max_q_len) #train_question_ids

        val_dataset[0], val_mask[0] = self.pad(val_dataset[0], self.max_ctx_len) #val_context_ids
        val_dataset[1], val_mask[1] = self.pad(val_dataset[1], self.max_q_len) #val_question_ids


        for i in range(1,len(train_dataset[0])):
            assert len(train_dataset[0][i]) == len(train_dataset[0][i - 1]), "Incorrectly padded train context"
            assert len(train_dataset[1][i]) == len(train_dataset[1][i - 1]), "Incorrectly padded train question"

        for i in range(1,len(val_dataset[0])):
            assert len(val_dataset[0][i]) == len(val_dataset[0][i - 1]), "Incorrectly padded val context"
            assert len(val_dataset[1][i]) == len(val_dataset[1][i - 1]), "Incorrectly padded val question"

        print("Training/val data padding verification completed.")
        
        train_dataset.extend(train_mask)
        val_dataset.extend(val_mask)
        train_dataset = np.array(train_dataset).T
        val_dataset = np.array(val_dataset).T

        train_context = contexts[0]
        dev_context = contexts[1]

        num_epochs = self.flags.epochs

        if self.flags.debug:
            train_dataset = train_dataset[:self.flags.batch_size]
            num_epochs = 100

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session, train_examples=train_dataset, dev_set=val_dataset, context = train_context)
            logging.info("Saving model in %s", train_dir)
            saver.save(session, train_dir)

        self.evaluate_answer(session, val_dataset, dev_context, sample=None, log=True)




