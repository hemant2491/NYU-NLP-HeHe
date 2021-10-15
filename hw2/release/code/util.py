import mxnet as mx
mx.random.seed(42)
device = mx.cpu()

from mxnet import np, gluon, init, autograd
import numpy as onp

from model import *

def logsumexp(scores, axis=-1, keepdims=False):
    """
    Parameters:
        scores : (..., vocab_size)
    Returns:
        normalizer : (..., 1)
            same dimension as scores
    """
    m = 0
    return m + np.log(np.sum(np.exp(scores - m), axis=axis, keepdims=keepdims))

def generate_dataset_identity(vocab_size, num_labels, length, size):
    """Generate a simple dataset where the output equals the input,
    e.g. [1,2,3] -> [1,2,3]
    """
    assert vocab_size == num_labels
    X = np.random.randint(low=1, high=vocab_size, size=(size, length))
    Y = np.copy(X)
    # Set first symbol to START (0)
    X[:, 0] = 0
    Y[:, 0] = 0
    return X, Y

def generate_dataset_rnn(vocab_size, num_labels, length, size):
    """Generate a dataset where the RNNModel achieves much lower loss than the UnigramModel.
    Parameters:
        vocab_size : int
            size of the output space for each input symbol
        num_labels: int
            size of the label set
        length : int
            the input sequence length
        size : int
            number of examples to generate
    Returns:
        X : mxnet.numpy.ndarray (size, length)
        Y : mxnet.numpy.ndarray (size, length)
            each element in Y must be an integer in [0, num_labels).
    """
    X = np.random.randint(low=1, high=vocab_size, size=(size, length))
    Y = np.zeros(X.shape)
    Y[:, :-1] = np.mod(X[:, 1:], num_labels - 1) + 1
    X[:, 0] = 0
    Y[:, 0] = 0
    return X, Y

def generate_dataset_hmm(vocab_size, num_labels, length, size):
    num_states = num_labels - 1  # don't count START
    generate_multinomial = lambda num_outcomes: onp.random.dirichlet(onp.random.randint(1, 10, num_outcomes))
    #start_prob = np.array([generate_multinomial(num_states)])
    #trans_prob = np.array([generate_multinomial(num_states) for _ in range(num_states)])
    #emiss_prob = np.array([generate_multinomial(vocab_size - 1) for _ in range(num_states)])
    start_prob = np.array([[0.3, 0.3, 0.4]])
    trans_prob = np.array([
        [0.1, 0.2, 0.7],
        [0.6, 0.1, 0.3],
        [0.1, 0.6, 0.4]
        ])
    emiss_prob = np.array([
        [0.6, 0.3, 0.1, 0.0],
        [0.3, 0.6, 0.0, 0.1],
        [0.1, 0.0, 0.5, 0.4],
        [0.1, 0.1, 0.1, 0.6],
        ])
    X, Y = _sample_from_hmm(start_prob, trans_prob, emiss_prob, length - 1, size)
    # Offset START
    X = X + 1
    Y = Y + 1
    # Add START
    X = np.hstack([np.zeros((size, 1)), X])
    Y = np.hstack([np.zeros((size, 1)), Y])
    return X, Y

def _sample_from_hmm(start_prob, trans_prob, emiss_prob, length, size):
    """Generate sequences of fixed length according to the start state probability
    and the transition matrix.
    Parameters:
        start_prob : start_prob[i] = p(state=i | START)
        trans_prob : trans_prob[i][j] = p(state=j | prev_state=i)
        emiss_prob : emiss_prob[i][j] = p(data=j | state=i)
    Returns:
        obs : (size, length)
        state : (size, length)
    """
    def _sample(state, cdf):
        cdf_ = cdf[state, :]
        a = np.random.rand(state.shape[0], 1)
        outcome = np.argmax((cdf_ > a).astype(np.int32), axis=-1)
        return outcome

    data = np.zeros((size, length))
    states = np.zeros((size, length))
    start_prob_cumsum = np.cumsum(start_prob, axis=1)
    trans_prob_cumsum = np.cumsum(np.array(trans_prob), axis=1)
    emiss_prob_cumsum = np.cumsum(np.array(emiss_prob), axis=1)
    states[:, 0] = _sample(np.zeros(size), start_prob_cumsum)
    for t in range(length):
        curr_state = states[:, t]
        data[:, t] = _sample(curr_state, emiss_prob_cumsum)
        if t + 1 < length:
            states[:, t+1] = _sample(curr_state, trans_prob_cumsum)
    return data, states

def compute_loss(X, Y, model, loss):
    total_loss = 0
    for j in range(X.shape[0]):
        x, y = X[j:j+1, :], Y[j:j+1, :] # (B, T)
        x, y = x.as_in_ctx(device), y.as_in_ctx(device)
        scores = model(x) # (B, T, vocab_size)
        l = loss(scores, y).mean()
        total_loss += l
    return total_loss / X.shape[0]

def train(X_train, Y_train, X_valid, Y_valid, model, loss, learning_rate, num_epochs):
    model.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(),
                                'adam', {'learning_rate': learning_rate})
    batch_size = 1
    step = 0
    for i in range(num_epochs):
        for j in range(X_train.shape[0]):
            x, y = X_train[j:j+1, :], Y_train[j:j+1, :] # (B, T)
            x, y = x.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                scores = model(x) # (B, T, vocab_size)
                l = loss(scores, y).mean()
            l.backward()
            trainer.step(1)
            step += 1
            if step % 100 == 0:
                valid_loss = compute_loss(X_valid, Y_valid, model, loss)
                print('step={}, curr_loss={}, valid_loss={}'.format(step, l, valid_loss))
