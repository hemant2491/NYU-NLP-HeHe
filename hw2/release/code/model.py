from mxnet import np, npx, gluon
from mxnet.gluon import nn, rnn
npx.set_np()

class UnigramModel(nn.Block):
    """Label the sequence by classifying each input symbol.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(UnigramModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dense1 = nn.Dense(num_hiddens, flatten=False)
        self.dense2 = nn.Dense(num_labels, flatten=False)

    def forward(self, inputs):
        X = npx.one_hot(inputs, self.vocab_size)  # (B, T, H)
        Y = self.dense1(X)  # (B, T, H)
        output = self.dense2(Y)  # (B, T, H)
        return output

class RNNModel(nn.Block):
    """Label the sequence by independent prediction at each time step
    using all input context.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn.LSTM(num_hiddens, bidirectional=True, layout='NTC')
        self.vocab_size = vocab_size
        self.dense = nn.Dense(num_labels, flatten=False)

    def forward(self, inputs):
        """
        Parameters:
            inputs : (batch_size, seq_lens, num_hidden_units)
            state : (batch_size, num_hidden_units)
                initial state of RNN
        Returns:
            output : (seq_lens, batch_size, num_labels)
                predicted scores for labels at each time step
        """
        # Set initial state to zero
        init_state = self.begin_state(inputs.shape[0])
        # One hot representation of input symbols
        X = npx.one_hot(inputs, self.vocab_size)  # (B, T, H)
        Y, state = self.rnn(X, init_state)
        output = self.dense(Y)  # (B, T, H)
        return output

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

class CRFRNNModel(RNNModel):
    """Add a CRF layer on top of the RNN model.
    """
    def __init__(self, num_hiddens, vocab_size, num_labels, **kwargs):
        super(CRFRNNModel, self).__init__(num_hiddens, vocab_size, num_labels, **kwargs)
        self.bigram_scores = gluon.Parameter('weights', shape=(num_labels, num_labels))
        self.num_labels = num_labels

    def forward(self, inputs):
        unigram_scores = super(CRFRNNModel, self).forward(inputs)  # RNN outputs
        batch_size, seq_len, vocab_size = unigram_scores.shape
        bigram_scores = np.broadcast_to(self.bigram_scores.data(), (batch_size, seq_len, self.num_labels, self.num_labels))
        return unigram_scores, bigram_scores
