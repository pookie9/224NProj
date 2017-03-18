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
        -vocab_dim: dimension of the embeddings
    """
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input=None, attention_inputs=None, model_type="gru"):
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
        with tf.variable_scope("Encoder"):

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

            outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, 
                                           sequence_length=masks, 
                                           dtype=tf.float32,
                                           initial_state=encoder_state_input)
        return outputs, final_state



class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self,knowledge_rep,masks,state_size,model_type="gru"):
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

        batch_size = tf.shape(knowledge_rep)[0]
        with vs.variable_scope("answer_start"):            
            W_start = tf.get_variable("W_start", shape=(1,1,state_size),initializer=tf.contrib.layers.xavier_initializer())
            W_start = tf.tile(W_start,[batch_size,1,1])
            print ("W_START",W_start)
            print ("knowledge_rep",knowledge_rep)
            start_probs = batch_matmul(knowledge_rep,tf.transpose(W_start,perm=[0,2,1]))
            

        with vs.variable_scope("answer_end"):
            cell = tf.nn.rnn_cell.GRUCell(state_size)
            all_end_probs, _ = tf.nn.dynamic_rnn(cell, knowledge_rep,sequence_length=masks,dtype=tf.float32,initial_state=None)
            W_end = tf.get_variable("W_end", shape=(1, 1, state_size),initializer=tf.contrib.layers.xavier_initializer())
            print ("all_end_probs",all_end_probs)
            print ("W_END",W_end)
            W_end = tf.tile(W_end,[batch_size,1,1])
            end_probs = batch_matmul(all_end_probs,tf.transpose(W_end,perm=[0,2,1]))
            #end_probs = tf.reduce_sum(all_end_probs * W_end, reduction_indices=2)

        start_probs = tf.squeeze(start_probs,2)
        end_probs = tf.squeeze(end_probs,2)

        bool_masks = tf.cast(tf.sequence_mask(masks,maxlen=self.output_size),tf.float32)
        a = tf.constant(-1e30)
        b = tf.constant(1.0)
        add_mask = (a*(b-bool_masks))
        start_probs = start_probs + add_mask
        end_probs = end_probs + add_mask
        return start_probs, end_probs

class QASystem(object):
    def __init__(self, encoder, decoder, flags,embeddings):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.flags=flags
        self.embeddings=embeddings
        self.encoder=encoder
        self.decoder=decoder
        self.max_ctx_len=max_ctx_len
        self.max_q_len=max_q_len
        assert self.max_ctx_len==self.flags.output_size, "WRONG MAX_CTX_LEN OR OUTPUT SIZE "+str(max_ctx_len)+","+str(self.flags.output_size)
        
        # ==== set up placeholder tokens ========
        self.context_placeholder=tf.placeholder(tf.int32,shape=(None,self.max_ctx_len),name="context_placeholder")
        self.question_placeholder=tf.placeholder(tf.int32,shape=(None,self.max_q_len),name="question_placeholder")
        self.answer_span_placeholder=tf.placeholder(tf.int32,shape=(None,2),name="answer_span_placeholder")
        self.mask_ctx_placeholder=tf.placeholder(tf.int32,shape=(None,),name="mask_ctx_placeholder")
        self.mask_q_placeholder=tf.placeholder(tf.int32,shape=(None,),name="mask_q_placeholder")
        self.dropout_placeholder=tf.placeholder(tf.float32,shape=(),name='dropout_placeholder')
        
        # ==== assemble pieces ====
        #with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        self.setup_embeddings()
        self.setup_system()
        self.setup_loss()

        # ==== set up training/updating procedure ====
        self.learning_rate=self.flags.learning_rate
        self.optimizer=get_optimizer("adam")
        self.optimizer=self.optimizer(self.learning_rate)
        gvs=self.optimizer.compute_gradients(self.loss)
        capped_gvs=[]
        for grad,var in gvs:
            if grad is None:
                capped_gvs.append((grad,var))
            else:
                capped_gvs.append((tf.clip_by_norm(grad,self.flags.max_gradient_norm),var))
        self.train_op=self.optimizer.apply_gradients(capped_gvs)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question_states,final_question_state=self.encoder.encode(self.question_embeddings,
                                                                 self.mask_q_placeholder,
                                                                 encoder_state_input=None,
                                                                 attention_inputs=None,
                                                                 model_type=self.flags.model_type,
                                                                 dropout=self.flags.dropout)

        ctx_states,final_ctx_state=self.encoder.encode(self.context_embeddings,
                                                       self.mask_ctx_placeholder,
                                                       encoder_state_input=None,
                                                       attention_inputs=None,
                                                       model_type=self.flags.model_type,
                                                       dropout=self.flags.dropout)

        weighted_ctx_states=self.mixer(final_question_state,ctx_states)

        self.start_probs,self.end_probs=self.decoder.decode(weighted_ctx_states,self.mask_ctx_placeholder,initial_state=None)
                                                                 

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        #with vs.variable_scope("loss"):
        print ("START PROBS SHAPE",self.start_probs)
        print ("START_ANSWER_SPAN",self.answer_span_placeholder[:,0])
        print ("END PROBS SHAPE",self.end_probs)
        print ("END_ANSWER_SPAN",self.answer_span_placeholder[:,1])
        start_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.start_probs,self.answer_span_placeholder[:,0]))
        end_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.end_probs,self.answer_span_placeholder[:,1]))
        self.loss=start_loss+end_loss
        
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        #with vs.variable_scope("embeddings"):

        embedding=tf.Variable(self.pretrained_embeddings,name="embedding",trainable=False,dtype=tf.float32)
        self.question_embeddings=tf.nn.embedding_lookup(self.embeddings, self.question_ids_placeholder)
        self.context_embeddings=tf.nn.embedding_lookup(self.embeddings, self.context_ids_placeholder)

    def optimize(self, session, context_batch,question_batch,answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.context_placeholder]=context_batch
        input_feed[self.question_placeholder]=question_batch
        input_feed[self.mask_ctx_placeholder]=mask_ctx_batch
        input_feed[self.mask_q_placeholder]=mask_q_batch
        input_feed[self.dropout_placeholder]=self.flags.dropout
        input_feed[self.answer_span_placeholder]=answer_span_batch
        
        output_feed = [self.train_op,self.loss]

        _,loss = session.run(output_feed, input_feed)

        return loss

    def test(self, sess, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        input_feed[self.context_placeholder]=context_batch
        input_feed[self.question_placeholder]=question_batch
        input_feed[self.mask_ctx_placeholder]=mask_ctx_batch
        input_feed[self.mask_q_placeholder]=mask_q_batch
        input_feed[self.dropout_placeholder]=self.flags.dropout
        input_feed[self.answer_span_placeholder]=answer_span_batch

        
        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, context_batch, question_batch, mask_ctx_batch, mask_q_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.context_placeholder]=context_batch
        input_feed[self.question_placeholder]=question_batch
        input_feed[self.mask_ctx_placeholder]=mask_ctx_batch
        input_feed[self.mask_q_placeholder]=mask_q_batch
        input_feed[self.dropout_placeholder]=self.flags.dropout

        output_feed = [self.start_probs,self.end_probs]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context_ids,question_ids,ctx_mask,q_mask):

        yp, yp2 = self.decode(session, context_ids,question_ids,ctx_mask,q_mask)

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


        return test(self, sess, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch)
    
    def evaluate_answer(self, session, context_ids,question_ids,ctx_mask,q_mask,answer_span, context, sample=100, log=False,eval_set="train"):
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
        if sample is not None:
            sample_indices=p.random.choice(dataset.shape[0], sample)
            context_ids=context_ids[sample_indices,:]
            question_ids=question_ids[sample_indices,:]
            ctx_mask=ctx_mask[sample_indices,:]
            q_mask=q_mask[sample_indices,:]
            answer_span=answer_span[sample_indices,:]
            context=context[sample_indices,:]

        a_s, a_e = self.answer(session, sampled)

        f1=[]
        em=[]
        print ("SIZE SAMPLED",sampled)

        for i in range(sample):
            pred_words=' '.join(context[i][a_s[i]:a_e[i]+1])
            actual_words=' '.join(context[i][answer_span[i][0]:answer_span[i][1]+1])
            f1.append(f1_score(pred_words,actual_words))
            cur_em=(exact_match_score(pred_words,actual_words))
            val=1.0
            for word in cur_em:
                if not word:
                    val=0.0
            em.append(val)
        f1=sum(f1)/len(f1)
        em=sum(em)/len(em)
        if log:
            logging.info("{},F1: {}, EM: {}, for {} samples".format(eval_set,np.mean(f1), None , sample))
        return f1, em

    def run_epoch(self, sess, train_set, val_set, train_context,val_context):
        prog_train = Progbar(target=1 + int(len(train_set) / self.flags.batch_size))
        for i, batch in enumerate(minibatches(train_set, self.flags.batch_size)):
            loss = self.optimize(sess, *batch)
            prog_train.update(i + 1, [("train loss", loss)])
        print("")

        prog_val = Progbar(target=1 + int(len(val_set) / self.flags.batch_size))
        for i, batch in enumerate(minibatches(val_set, self.flags.batch_size)):
            break
            val_loss = self.validate(sess, *batch)
            prog_val.update(i + 1, [("val loss", val_loss)])

        train_f1, train_em = self.evaluate_answer(sess,train_set, context=train_context, sample=100, log=True, eval_set="-TRAIN-")
        val_f1, val_em = self.evaluate_answer(sess,val_set, context=val_context, sample=100, log=True, eval_set="-VAL-")

    
    def train(self, session, dataset, val_dataset,train_dir):
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

        context_ids,question_ids,answer_spans,ctx_mask,q_mask,context=dataset
        val_context_ids,val_question_ids,val_answer_spans,val_ctx_mask,val_q_mask,val_context=val_dataset
        
        for epoch in range(self.flags.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session, train_set=train_dataset, val_set=val_dataset, train_context=train_context,val_context=val_context)
            logging.info("Saving model in %s", train_dir)
            saver.save(session, train_dir)

        
