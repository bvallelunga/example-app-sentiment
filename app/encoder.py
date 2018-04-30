"""
This file is a slightly modified version of the file, of the same name, found on the openai repo:
https://github.com/openai/generating-reviews-discovering-sentiment
"""

import numpy as np
import tensorflow as tf

# Add Current Directory to Path
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import HyperParams
from utils import preprocess

MODEL_PARAMS_PATH = 'model'
SENTIMENT_NEURON_IDX = 2388

global num_loaded
num_loaded = 0


def load_params(shape, dtype, *args, **kwargs):
    global num_loaded
    num_loaded += 1
    return params[num_loaded - 1]


def embed(X, num_dim, scope='embedding'):
    """Build the embedding layer subgraph in tensorflow."""
    with tf.variable_scope(scope):
        embedding = tf.get_variable(
            "w", [hps.num_vocab, num_dim], initializer=load_params)
        h = tf.nn.embedding_lookup(embedding, X)
        return h


def fc(x, num_outputs, activation, wn=False, bias=True, scope='fc'):
    """Build the fully connected layer subgraph in tensorflow."""
    with tf.variable_scope(scope):
        num_inputs = x.get_shape()[-1].value
        w = tf.get_variable("w", [num_inputs, num_outputs], initializer=load_params)
        if wn:
            g = tf.get_variable("g", [num_outputs], initializer=load_params)
        if wn:
            w = tf.nn.l2_normalize(w, dim=0) * g
        z = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("b", [num_outputs], initializer=load_params)
            z = z + b
        h = activation(z)
        return h


def mlstm(inputs, c, h, m, num_dims, scope='lstm', wn=False):
    """Build the mlstm (multiplicative Long-Short Term Memory) subgraph in tensorflow.

    An mlstm is a type of recurrent neural network. For more information see the arxiv paper
    "Multiplicative LSTM for Sequence Modeling" by Krause et al (https://arxiv.org/pdf/1609.07959.pdf).
    """
    num_inputs = inputs[0].get_shape()[1].value
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [num_inputs, num_dims * 4], initializer=load_params)
        wh = tf.get_variable("wh", [num_dims, num_dims * 4], initializer=load_params)
        wmx = tf.get_variable("wmx", [num_inputs, num_dims], initializer=load_params)
        wmh = tf.get_variable("wmh", [num_dims, num_dims], initializer=load_params)
        b = tf.get_variable("b", [num_dims * 4], initializer=load_params)

        if wn:
            gx = tf.get_variable("gx", [num_dims * 4], initializer=load_params)
            gh = tf.get_variable("gh", [num_dims * 4], initializer=load_params)
            gmx = tf.get_variable("gmx", [num_dims], initializer=load_params)
            gmh = tf.get_variable("gmh", [num_dims], initializer=load_params)

    if wn:
        wx = tf.nn.l2_normalize(wx, dim=0) * gx
        wh = tf.nn.l2_normalize(wh, dim=0) * gh
        wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh

    cs = []
    for idx, x in enumerate(inputs):
        prev_m = tf.matmul(x, wmx) * tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(prev_m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)

        if m is not None:
            ct = f * c + i * u
            ht = o * tf.tanh(ct)
            prev_m = m[:, idx, :]
            c = ct * prev_m + c * (1 - prev_m)
            h = ht * prev_m + h * (1 - prev_m)
        else:
            c = f * c + i * u
            h = o * tf.tanh(c)

        inputs[idx] = h
        cs.append(c)

    cs = tf.stack(cs)
    return inputs, cs, c, h


def model(x, s, m=None, reuse=False):
    """Build the entire graph, for the model, in tensorflow."""
    num_steps = x.get_shape()[1]
    c_start, h_start = tf.unstack(s, num=hps.num_states)

    with tf.variable_scope('model', reuse=reuse):
        words = embed(x, hps.num_embed)
        inputs = tf.unstack(words, num_steps, 1)
        hs, cells, c_final, h_final = mlstm(inputs, c_start, h_start, m, hps.num_hidden, scope='rnn', wn=hps.rnn_wn)
        hs = tf.reshape(tf.concat(hs, 1), [-1, hps.num_hidden])
        logits = fc(hs, hps.num_vocab, activation=lambda el: el, wn=hps.out_wn, scope='out')

    states = tf.stack([c_final, h_final], 0)
    return cells, states, logits


def ceil_round_step(n, step):
    return int(np.ceil(n / step) * step)


def batch_pad(xs, num_batch, num_steps):
    """Pads a batch of data.

    Given a batch of data with varying sequence lengths, pad the seqs with zeros such that all sequence lengths
    are equal to the max sequence length. This is a necessary data preprocessing step since tensorflow models can not
    handle varied sequence lengths due to the static nature of the underlying computation graphs.
    """
    xmb = np.zeros((num_batch, num_steps), dtype=np.int32)
    mmb = np.ones((num_batch, num_steps, 1), dtype=np.float32)
    for i, x in enumerate(xs):
        num_pad = num_steps - len(x)
        xmb[i, -len(x):] = list(x)
        mmb[i, :num_pad] = 0
    return xmb, mmb


class Model(object):
    """The sentiment model."""

    def __init__(self, model_params, num_batch=128, num_steps=64):
        global hps
        hps = HyperParams(
            load_path='model_params/params.jl',
            num_hidden=4096,
            num_embed=64,
            num_steps=num_steps,
            num_batch=num_batch,
            num_states=2,
            num_vocab=256,
            out_wn=False,
            rnn_wn=True,
            rnn_type='mlstm',
            embed_wn=True,
        )

        global params
        # moved param loading to separate utility function in order to speed up tests i.e. it's quicker to load
        # the params once and passed them to each instance of the model rather than loading them each time a model
        # is instantiated.
        params = model_params
        # params = [np.load(os.path.join(data_path, '%d.npy'%i)) for i in range(15)]
        # params[2] = np.concatenate(params[2:6], axis=1)
        # params[3:6] = []

        x = tf.placeholder(tf.int32, [None, hps.num_steps])
        m = tf.placeholder(tf.float32, [None, hps.num_steps, 1])
        s = tf.placeholder(tf.float32, [hps.num_states, None, hps.num_hidden])
        cells, states, logits = model(x, s, m, reuse=False)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.get_variable_scope().reuse_variables()

        def seq_rep(xmb, mmb, smb):
            """Given processed text sequences and model states, get the sequence representations.

            This will execute the tensorflow graph and return the numerical representations/features
            of our input texts (e.g. the sentiment value).
            """
            return sess.run(states, {x: xmb, m: mmb, s: smb})

        def predict(xs):
            """Transform raw text sequences into their numerical features/representations."""
            xs = [preprocess(x) for x in xs]
            lens = np.asarray([len(x) for x in xs])
            sorted_idxs = np.argsort(lens)
            unsort_idxs = np.argsort(sorted_idxs)
            sorted_xs = [xs[i] for i in sorted_idxs]
            max_len = np.max(lens)
            offset = 0
            n = len(xs)
            smb = np.zeros((2, n, hps.num_hidden), dtype=np.float32)

            for step in range(0, ceil_round_step(max_len, num_steps), num_steps):
                start = step
                end = step + num_steps
                x_subseq = [x[start:end] for x in sorted_xs]
                num_done = sum([x == b'' for x in x_subseq])
                offset += num_done
                x_subseq = x_subseq[num_done:]
                sorted_xs = sorted_xs[num_done:]
                num_subseq = len(x_subseq)
                xmb, mmb = batch_pad(x_subseq, num_subseq, num_steps)
                for batch in range(0, num_subseq, num_batch):
                    start = batch
                    end = batch + num_batch
                    batch_smb = seq_rep(xmb[start:end], mmb[start:end], smb[:, offset+start:offset+end, :])
                    smb[:, offset+start:offset+end, :] = batch_smb

            features = smb[0, unsort_idxs, :]
            return features[:, SENTIMENT_NEURON_IDX]

        self.predict = predict
