import numpy as np

def auxiva_iss(
    X,
    n_iter=20,
    proj_back=True,
    model="laplace",
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    **kwargs,
):


    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    n_src = X.shape[2]

    # for now, only supports determined case
    assert n_chan == n_src

    # pre-allocate arrays
    r_inv = np.zeros((n_src, n_frames))
    v_num = np.zeros((n_freq, n_src), dtype=X.dtype)
    v_denom = np.zeros((n_freq, n_src), dtype=np.float64)
    v_iss = np.zeros((n_freq, n_src), dtype=X.dtype)

    # Things are more efficient when the frequencies are over the first axis
    X = X.transpose([1, 2, 0]).copy()

    # Initialize the demixed outputs
    Y = X.copy()

    n_frames_sqrt = np.sqrt(n_frames)

    for epoch in range(n_iter):

        # shape: (n_src, n_frames)
        # OP: n_frames * n_src
        eps = 1e-10
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        for s in range(n_src):

            # OP: n_frames * n_src
            v_num = (Y * r_inv[None, :, :]) @ np.conj(
                Y[:, s, :, None]
            )  # (n_freq, n_src, 1)
            # OP: n_frames * n_src
            v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2
            # (n_freq, n_src, 1)

            # OP: n_src
            v_iss[:, :] = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            v_iss[:, s] -= n_frames_sqrt / np.sqrt(v_denom[:, s, 0])

            # update demixed signals
            # OP: n_frames * n_src
            Y[:, :, :] -= v_iss[:, :, None] * Y[:, s, None, :]

        # Monitor the algorithm progression
        if callback is not None and (epoch + 1) in callback_checkpoints:
            Y_tmp = Y.transpose([2, 0, 1]).copy()
            callback(Y_tmp, model)

    if return_filters is not None:
        # Demixing matrices were not computed explicitely so far,
        # do it here, if necessary
        W = Y[:, :, :n_chan] @ np.linalg.inv(X[:, :, :n_chan])

    Y = Y.transpose([2, 0, 1]).copy()
    X = X.transpose([2, 0, 1])

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y