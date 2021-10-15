import argparse

from util import *
from model import *
from submission import greedy_decode, hamming_loss, generate_dataset_rnn, compute_normalizer, bruteforce_normalizer, viterbi_decode, bruteforce_decode, crf_loss


def evaluate(X, Y, model, decoder, loss):
    total_loss = 0
    for j in range(X.shape[0]):
        x, y = X[j:j+1, :], Y[j:j+1, :] # (B, T)
        x, y = x.as_in_ctx(device), y.as_in_ctx(device)
        scores = model(x)
        _, y_hat = decoder(scores)
        total_loss += loss(y, y_hat)
    return total_loss / X.shape[0]

def test_unigram():
    print('testing unigram model')
    num_hiddens = 5
    model = UnigramModel(num_hiddens, vocab_size, num_labels)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 10)
    error = evaluate(X_valid, Y_valid, model, greedy_decode, hamming_loss)
    print('0-1 error={}'.format(error))

def test_rnn():
    print('testing RNN model')
    num_hiddens = 5
    model = RNNModel(num_hiddens, vocab_size, num_labels)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 10)
    error = evaluate(X_valid, Y_valid, model, greedy_decode, hamming_loss)
    print('0-1 error={}'.format(error))

def test_hamming_loss():
    a = np.array([[2, 3, 1, 0]])
    b = np.array([[1, 3, 1, 0]])
    assert hamming_loss(a, b) == 0.25

def test_score_sequence():
    unigram_scores = np.zeros((1, 3, 3))
    unigram_scores[0, 0, :] = [0.1, 0.2, 0.7]
    unigram_scores[0, 1, :] = [0.5, 0.2, 0.3]
    unigram_scores[0, 2, :] = [0.3, 0.2, 0.5]
    bigram_scores = np.array([
        [0.5, 0.1, 0.3],
        [0.1, 0.4, 0.2],
        [0.2, 0.2, 0.3]
        ])
    bigram_scores = np.broadcast_to(bigram_scores, (1, 3, 3, 3))
    seqs = np.array([[0, 1, 2]])
    # 0.1 + 0.2 + 0.5 + 0.1 + 0.2 = 1.1
    score = score_sequence(seqs, unigram_scores, bigram_scores)
    onp.testing.assert_array_equal(score, np.array([[1.1]]))

def test_viterbi():
    print('testing viterbi decoding')
    for _ in range(3):
        unigram_scores = np.random.uniform(0, 1, (1, 3, 3))
        bigram_scores = np.random.uniform(0, 1, (1, 3, 3, 3))
        score, y_hat = viterbi_decode((unigram_scores, bigram_scores))
        score_brute, y_brute = bruteforce_decode(unigram_scores, bigram_scores)
        # Note that the paths found may not be the same
        onp.testing.assert_array_almost_equal(score, score_brute)

def test_normalizer():
    print('testing normalizer computation')
    for _ in range(3):
        unigram_scores = np.random.uniform(0, 1, (1, 3, 3))
        bigram_scores = np.random.uniform(0, 1, (1, 3, 3, 3))
        score = compute_normalizer(unigram_scores, bigram_scores)
        score_brute = bruteforce_normalizer(unigram_scores, bigram_scores)
        onp.testing.assert_array_almost_equal(score, score_brute)

def test_crfrnn():
    print('testing CRFRNN model')
    num_hiddens = 5
    model = CRFRNNModel(num_hiddens, vocab_size, num_labels)
    loss = crf_loss
    train(X_train, Y_train, X_valid, Y_valid, model, loss, 0.01, 5)
    error = evaluate(X_valid, Y_valid, model, viterbi_decode, hamming_loss)
    print('0-1 error={}'.format(error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test', help='test name')
    parser.add_argument('--data', help='dataset name', default='identity')
    args = parser.parse_args()

    vocab_size = 5
    N = 100
    length = 10
    num_labels = 4
    if args.data == 'identity':
        generate_dataset = generate_dataset_identity
        num_labels = vocab_size
    elif args.data == 'rnn':
        generate_dataset = generate_dataset_rnn
    elif args.data == 'hmm':
        generate_dataset = generate_dataset_hmm
    else:
        raise ValueError
    X_train, Y_train = generate_dataset(vocab_size, num_labels, length, N)
    X_valid, Y_valid = generate_dataset(vocab_size, num_labels, length, N)

    if args.test == 'unigram':
        test_unigram()
    elif args.test == 'rnn':
        test_rnn()
    elif args.test == 'normalizer':
        test_normalizer()
    elif args.test == 'viterbi':
        test_viterbi()
    elif args.test == 'crfrnn':
        test_crfrnn()
    else:
        raise ValueError
