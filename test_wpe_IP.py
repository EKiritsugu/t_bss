##这个代码用于测试ILRMA-T-IP算法


import os
import soundfile as sf
from nara_wpe.utils import stft, istft
from toolbox import projection_back
import numpy as np

n_sources = 3
mixed_sig_path = 'mixed/'+str(n_sources) + 'ch/'
save_path = 'wped/'+str(n_sources) + 'ch/'

file_list = os.listdir(mixed_sig_path)

stft_options = dict(size=1024, shift=1024//4)
sampling_rate = 16000
delay = 2
iterations = 100
taps = 5

wav_name = file_list[4]
y = sf.read(mixed_sig_path+wav_name)[0]
y = y.T
Y = stft(y, **stft_options)
X = Y.transpose(2, 0, 1).copy()
del Y, y# 把可能混淆的变量删除
#######################################################################################################################
# input&output X: n_freq, n_ch, n_frame

X = X.transpose(2, 0, 1)
n_iter = 20
n_components = 2
if True:
    """
    X: ndarray (nframes, nfrequencies, nchannels)
    Returns an (nframes, nfrequencies, nsources) array.
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case

    n_src = X.shape[2]

    # Only supports determined case
    assert n_chan == n_src, "There should be as many microphones as sources"

    W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=X.dtype)


    # initialize the nonnegative matrixes with random values
    T = np.array(0.1 + 0.9 * np.random.rand(n_src, n_freq, n_components))
    V = np.array(0.1 + 0.9 * np.random.rand(n_src, n_frames, n_components))
    R = np.zeros((n_src, n_freq, n_frames))

    lambda_aux = np.zeros(n_src)
    eps = 1e-15
    eyes = np.tile(np.eye(n_chan, n_chan), (n_freq, 1, 1))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X

    X = X.transpose([1, 2, 0]).copy()
    print(np.shape(X))# n_freq, n_ch, n_frames

    np.matmul(T, V.swapaxes(1, 2), out=R)

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)

    demix(Y, X, W)

    # Y2.shape == R.shape == (n_src, n_freq, n_frames)
    Y2 = np.power(abs(Y.transpose([1, 0, 2])), 2.0)
    iR = 1 / R

    for epoch in range(n_iter):

        # simple loop as a start
        for s in range(n_src):
            ## NMF
            ######
            T[s, :, :] *= np.sqrt(np.dot(Y2[s, :, :]* iR[s, :, :] ** 2, V[s, :, :])/np.dot(iR[s, :, :], V[s, :, :]))
            T[T < eps] = eps
            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]
            V[s, :, :] *= np.sqrt(np.dot(Y2[s, :, :].T * iR[s, :, :].T ** 2, T[s, :, :])/np.dot(iR[s, :, :].T, T[s, :, :]))
            V[V < eps] = eps
            R[s, :, :] = np.dot(T[s, :, :], V[s, :, :].T)
            R[R < eps] = eps
            iR[s, :, :] = 1 / R[s, :, :]
            ##########

            ## IVA
            ######

            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            C = np.matmul((X * iR[s, :, None, :]), np.conj(X.swapaxes(1, 2))) / n_frames

            WV = np.matmul(W, C)
            W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            # normalize
            denom = np.matmul(
                np.matmul(W[:, None, s, :], C[:, :, :]), np.conj(W[:, s, :, None])
            )
            W[:, s, :] /= np.sqrt(denom[:, :, 0])

        demix(Y, X, W)
        np.power(abs(Y.transpose([1, 0, 2])), 2.0, out=Y2)

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(Y2[s, :, :]))
            W[:, :, s] *= lambda_aux[s]
            Y2[s, :, :] *= lambda_aux[s] ** 2
            R[s, :, :] *= lambda_aux[s] ** 2
            T[s, :, :] *= lambda_aux[s] ** 2


    Y = Y.transpose([2, 0, 1]).copy()
    z = projection_back(Y, X_original[:, :, 0])
    Y *= np.conj(z[None, :, :])




Z = Y.transpose(1, 2, 0)

# baseline
from nara_wpe.wpe import wpe
Z_wpe = wpe(X,    taps=taps,    delay=delay,    iterations=5,    statistics_mode='full')
#######################################################################################################################
z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
for i in range(n_sources):
    sf.write('test_wpe_'+str(i)+'.wav', z[i] , 16000)