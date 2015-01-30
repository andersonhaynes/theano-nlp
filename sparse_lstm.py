# -*- coding: utf-8 -*-
'''
    Theano implementation of a LSTM token sequence classifier

    This is a refactor of the code found at:
        http://deeplearning.net/tutorial/lstm.html

    The goal of the refactor is to make the code simpler, modular, and reusable.
    The LSTM classifier now has a sklearn-like interface, making it practical to 
    put to use in new projects (demonstrated here on two different datasets).

    Contact: Francois Chollet (fchollet@sphere-engineering.com)

    --------------------------------------------------
    Usage:
    --------------------------------------------------

    We are trying to classify sequences of tokens. 
    The total number of tokens (dimensionality of the input space) is noted "dim", 
    the number of classes is noted "ydim" (dimensionality of the output space).
    Each sequence is a list of integers in [0, dim[, 
    where each integer is the index of a token in the sparse input space.

        X_train # list of training sequences
        y_train # labels of training sequences (integers in [0, ydim[)
        X_test # list of sequences to classify

        clf = SparseLSTM(dim = dim, ydim = ydim)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

    if you want early stopping of training based on error
    on a holdout validation set (recommended), do:

        clf.fit(X_train, y_train, X_validation, y_validation)

    otherwise the training will run for max_epochs epochs (non-optimal).


    --------------------------------------------------
    Testing:
    --------------------------------------------------

    Available test datasets: 
        - IMDB movie review sentiment classification
        - Reuters newswire topic classification


    --------------------------------------------------
    References:
    --------------------------------------------------

        - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. 
            Neural computation, 9(8), 1735-1780.
            http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf 

        - Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. 
            Neural computation, 12(10), 2451-2471.
            http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015

        - Graves, Alex. Supervised sequence labelling with recurrent neural networks. 
            Vol. 385. Springer, 2012.
            http://www.cs.toronto.edu/~graves/preprint.pdf

        - http://www.iro.umontreal.ca/~lisa/pointeurs/nips2012_deep_workshop_theano_final.pdf
            Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Bergstra, James, Goodfellow, Ian, 
            Bergeron, Arnaud, Bouchard, Nicolas, and Bengio, Yoshua. 
            Theano: new features and speed improvements. 
            NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2012.

        - http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf
            Bergstra, James, Breuleux, Olivier, Bastien, Frédéric, Lamblin, Pascal, Pascanu, 
            Razvan, Desjardins, Guillaume, Turian, Joseph, Warde-Farley, David, and Bengio, Yoshua. 
            Theano: a CPU and GPU math expression compiler. 
            In Proceedings of the Python for Scientific Computing Conference (SciPy), June 2010.

'''

import random
from datetime import datetime
import numpy as np
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj
    

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')



def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd than needed, mirroring 
        the code for adadelta and rmsprop

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup+rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update



def pad_sequences(seqs, labels, maxlen=None):
    """
    pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    """
    # seqs: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


class SparseLSTM():
    '''
        Classify sequences of integers (= indexes of "1" values in a large sparse vector).

    '''

    def __init__(self,
            dim, # dimensionality of the sparse input space
            ydim, # number of classes to classify
            dim_proj=128,  # hidden units: sparse space embedding dimension and LSTM number of hidden units.
            max_epochs=50,  # maximum number of epoch to run
            decay_c=0.,  # weight decay for the classifier applied to the U weights.
            lrate=0.0001,  # learning rate for sgd (not used for adadelta and rmsprop)
            optim='adadelta',  # sgd, adadelta and rmsprop available, sgd very hard to use (probably needs momentum and decaying learning rate).     
            maxlen=200,  # input sequences longer then this get ignored
            batch_size=16,  # batch size during training.
            noise_std=0., 
            seed=113, # random generator seed
            use_dropout=True,  # if False slightly faster, but worse test error
        ):
        
        self.dim = dim
        self.ydim = ydim
        self.dim_proj = dim_proj
        self.max_epochs = max_epochs
        self.decay_c = decay_c
        self.lrate = lrate
        self.optim = optim
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.seed = seed
        self.use_dropout = use_dropout

        if optim == 'adadelta':
            optimizer = adadelta
        elif optim == 'rmsprop':
            optimizer = rmsprop
        elif optim == 'sgd':
            optimizer = sgd
        else:
            raise Exception('Optimiser not recognized')

        print 'Building model'
        # Generate all the theano shared variables we'll need (list below)
        self.init_params()
        # put the generated theano vars in a dict for easy manipulation 
        tparams = {
            'W_emb':self.W_emb,
            'W_lstm':self.W_lstm, 
            'U_lstm':self.U_lstm,
            'b_lstm':self.b_lstm,
            'U':self.U,
            'b':self.b
        }

        # use_noise is for dropout
        (use_noise, x, mask, y, f_pred_prob, cost) = self.build_model()

        # weights decay
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            weight_decay += (self.U**2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        # generate theano functions
        f_cost = theano.function([x, mask, y], cost, name='f_cost')

        grads = tensor.grad(cost, wrt=tparams.values())
        f_grad = theano.function([x, mask, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')
        f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

        # this is what we'll need to train the model
        self.use_noise = use_noise
        self.f_grad_shared = f_grad_shared
        self.f_update = f_update
        self.f_pred_prob = f_pred_prob


    def init_params(self):

        # embedding
        randn = np.random.rand(self.dim, self.dim_proj)        
        W_emb = (0.01 * randn).astype('float32')
        self.W_emb = theano.shared(W_emb, name='W_emb')

        # lstm
        W_lstm = np.concatenate([ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj)], axis=1)
        self.W_lstm = theano.shared(W_lstm, name='W_lstm')

        U_lstm = np.concatenate([ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj),
                               ortho_weight(self.dim_proj)], axis=1)
        self.U_lstm = theano.shared(U_lstm, name='U_lstm')

        b_lstm = np.zeros((4 * self.dim_proj,)).astype('float32')
        self.b_lstm = theano.shared(b_lstm, name='b_lstm')

        # classifier
        U = 0.01 * np.random.randn(self.dim_proj, self.ydim).astype('float32')
        self.U = theano.shared(U, name='U')
        b = np.zeros((self.ydim,)).astype('float32')
        self.b = theano.shared(b, name='b')


    def build_model(self):

        trng = RandomStreams(self.seed)

        use_noise = theano.shared(np.float32(0.)) # Used for dropout

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # embedding of input: matrix timesteps * samples * embedding dimensions 
        emb = self.W_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_proj])

        nsteps = emb.shape[0]
        if emb.ndim == 3:
            n_samples = emb.shape[1]
        else:
            n_samples = 1 # case embedding = matrix timesteps * embedding dimensions

        emb = (tensor.dot(emb, self.W_lstm) + self.b_lstm)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, self.U_lstm)
            preact += x_
            preact += self.b_lstm

            i = tensor.nnet.sigmoid(_slice(preact, 0, self.dim_proj))
            f = tensor.nnet.sigmoid(_slice(preact, 1, self.dim_proj))
            o = tensor.nnet.sigmoid(_slice(preact, 2, self.dim_proj))
            c = tensor.tanh(_slice(preact, 3, self.dim_proj))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        rval, updates = theano.scan(_step,
                                    sequences=[mask, emb],
                                    outputs_info=[tensor.alloc(0., n_samples,
                                                               self.dim_proj),
                                                  tensor.alloc(0., n_samples,
                                                               self.dim_proj)],
                                    name='lstm_layers',
                                    n_steps=nsteps)
        proj = rval[0]
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

        if self.use_dropout: 
            # apply dropout
            proj = dropout_layer(proj, use_noise, trng)

        # apply classifier
        pred = tensor.nnet.softmax(tensor.dot(proj, self.U) + self.b)
        f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')

        # cost: average -log(err)
        cost = -tensor.log(pred[tensor.arange(n_samples), y] + 1e-8).mean()

        return use_noise, x, mask, y, f_pred_prob, cost


    def fit(self, X_train, y_train, X_valid=None, y_valid=None, patience=4):
        '''
            train_X: sequences of integers in [0, dim[. 
            train_y: classes of the sequences, integers in [0, ydim[.
        '''
        decay_c = self.decay_c
        max_epochs = self.max_epochs
        batch_size = self.batch_size
        maxlen = self.maxlen
        lrate = self.lrate

        use_noise = self.use_noise
        f_grad_shared = self.f_grad_shared
        f_update = self.f_update
        
        uidx = 0  # number of updates so far
        best_valid_err = 1. # best validation error
        no_improv = 0 # nb of epochs with no improvements

        print 'Training...'
        start_time = datetime.now()
        
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(X_train), batch_size, shuffle=True)
            log_cost = 0.
            log_samples = 0
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [X_train[t]for t in train_index]
                y = [y_train[t] for t in train_index]

                # Get the data in np.ndarray format
                # return something of the shape (minibatch maxlen, n samples)
                x, mask, y = pad_sequences(x, y, maxlen=maxlen)
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                log_cost += cost
                log_samples += x.shape[1]

                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if not uidx % 100:
                    print 'Epoch:', eidx, 'Updates:', uidx, 'Cost:', log_cost/log_samples

            print 'Samples seen:', n_samples, 'Elapsed:', datetime.now() - start_time

            if X_valid is not None: 
                # check current validation error
                preds = self.predict(X_valid)

                valid_err = 1. - float((preds == y_valid).sum()) / len(y_valid)
                print 'Validation error: %.3f' % valid_err
                if valid_err < best_valid_err:
                    best_valid_err = valid_err
                    no_improv = 0
                else:
                    no_improv += 1
                    if no_improv > patience:
                        print 'Early stopping'
                        break

        print 'Training finished in', datetime.now() - start_time


    def predict_proba(self, X_test):
        preds = np.zeros((len(X_test), self.ydim))
        batch_size = self.batch_size

        for batch_index in range(0, len(X_test)/batch_size+1):

            batch = []
            for i in range(batch_index*batch_size, (batch_index+1)*batch_size):
                if i < len(X_test):
                    batch.append(i)
            if not batch:
                break

            x, mask, y = pad_sequences([X_test[t] for t in batch], [0. for t in batch], maxlen=None)

            batch_preds = self.f_pred_prob(x, mask)
            preds[batch_index*batch_size : (batch_index+1)*batch_size, :] = batch_preds

        return preds


    def predict(self, X_test):
        preds = self.predict_proba(X_test)
        return preds.argmax(axis=1)



def imdb_test():
    '''
        Binary sentiment classification across a IMDB movie review dataset
    '''
    import imdb_dataset as imdb

    n_words = 20000
    maxlen = 400

    print '-'*40
    print 'Imdb sentiment classification...'
    print '-'*40

    print 'Loading data...'
    (X_train, y_train), (X_test, y_test) = imdb.load_data(n_words=n_words, maxlen=maxlen)

    print 'Train samples:', len(X_train)
    print 'Positive training samples: %.3f' % (float(np.array(y_train).sum())/len(y_train))
    print 'Test samples:', len(X_test)
    print 'Positive test samples: %.3f' % (float(np.array(y_test).sum())/len(y_test))

    print 'Fitting model...'
    clf = SparseLSTM(dim=n_words, ydim=max(y_train)+1, maxlen=maxlen)
    clf.fit(X_train, y_train, X_valid=X_test, y_valid=y_test, patience=4)


def reuters_test():
    '''
        Topic classification over Reuters newswires
    '''
    import reuters_dataset as reuters

    print '-'*40
    print 'Reuters topic classification...'
    print '-'*40
    
    min_samples_per_topic = 20
    n_words = 20000
    maxlen = 500

    print 'Loading data...'
    (X_train, y_train), (X_test, y_test) = reuters.load_data(n_words=n_words, maxlen=maxlen, seed=111)

    print 'Train samples:', len(X_train)
    print 'Test samples:', len(X_test)

    print 'Fitting model...'
    clf = SparseLSTM(dim=n_words, ydim=max(y_train)+1, maxlen=maxlen)
    clf.fit(X_train, y_train, X_valid=X_test, y_valid=y_test, patience=4)


if __name__ == "__main__":
    theano.config.floatX = "float32"
    imdb_test()
    reuters_test()

