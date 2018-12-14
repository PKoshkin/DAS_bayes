import numpy as np
import scipy.stats as st


def expand_last(tensor, times):
    return np.repeat(np.reshape(tensor, np.shape(tensor) + (1, )), times, axis=-1)


def expand_first(tensor, times):
    return np.repeat(np.reshape(tensor, (1, ) + np.shape(tensor)), times, axis=0)


def pa(params, model):
    k_a = params["amax"] - params["amin"] + 1
    return np.full([k_a], 1.0 / k_a), np.arange(params["amin"], params["amax"] + 1)


def pb(params, model):
    k_b = params["bmax"] - params["bmin"] + 1
    return np.full([k_b], 1.0 / k_b), np.arange(params["bmin"], params["bmax"] + 1)


def cartesian_product(*arrays):
    arrays = list(arrays)
    arrays[0], arrays[1] = arrays[1], arrays[0]
    mesh = np.meshgrid(*arrays)
    result = list(map(lambda tensor: np.reshape(tensor, [-1]), mesh))
    result[0], result[1] = result[1], result[0]
    return tuple(result)


def pc(params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_range = np.arange(params["bmax"] + params["amax"] + 1)

    a_size, b_size, c_size = len(a_range), len(b_range), len(c_range)

    if model == 3:
        a, c = cartesian_product(a_range, c_range)
        mult_1 = st.binom.pmf(c, a, params["p1"]).reshape([a_size, c_size])  # shape: (a, c)
        mult_1 = mult_1.sum(axis=0)  # shape: (c,)

        b, c = cartesian_product(b_range, c_range)
        mult_2 = st.binom.pmf(c, b, params["p2"]).reshape([b_size, c_size])  # shape: (b, c)
        mult_2 = mult_2.sum(axis=0)  # shape: (c,)

        probs = np.array([
            np.sum(mult_1[:t] * np.flip(mult_2[:t], axis=0))
            for t in c_range + 1
        ])  # shape: (c,)
        return probs / a_size / b_size, c_range
    elif model == 4:
        a, b, c = cartesian_product(a_range, b_range, c_range)
        pmf = st.poisson.pmf(c, a * params["p1"] + b * params["p2"]).reshape([a_size, b_size, c_size])  # shape: (a, b, c)
        probs = pmf.sum(axis=(0, 1))  # shape: (c,)
        return probs / a_size / b_size, c_range
    else:
        raise ValueError("Unsupported model {}".format(model))


def pd(params, model):
    c_probs, c_vals = pc(params, model)
    d_range = np.arange(2 * (params["amax"] + params["bmax"]) + 1)

    probs = np.array([
        np.sum(st.binom.pmf(d - c_vals, c_vals, params["p3"]) * c_probs)
        for d in d_range
    ])
    return probs, d_range


def pb_d(d, params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    flat_d = d.reshape([-1])
    probs = pb_ad_numerator(a_range, flat_d, params, model)  # shape: (a, b, k_d * N)
    probs = probs.reshape(probs.shape[:-1] + d.shape)  # shape: (a, b, k_d, N)
    probs = probs.prod(axis=-1).sum(axis=0)  # shape: (b, k_d)
    denum = probs.sum(axis=0)
    return probs / denum, b_range


def pb_ad_numerator(a, d, params, model):
    a_range = a
    d_range = d
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_range = np.arange(params["bmax"] + params["amax"] + 1)

    a_size, b_size, c_size, d_size = len(a_range), len(b_range), len(c_range), len(d_range)

    if model == 3:
        x, y = cartesian_product(a_range, np.arange(params["amax"] + 1))
        mult_1 = st.binom.pmf(y, x, params["p1"]).reshape([a_size, 1, -1]).repeat(b_size, axis=1)
        mult_1 = mult_1.reshape([-1, mult_1.shape[-1]])

        x, y = cartesian_product(b_range, np.arange(params["bmax"] + 1))
        mult_2 = st.binom.pmf(y, x, params["p2"]).reshape([1, b_size, -1]).repeat(a_size, axis=0)
        mult_2 = mult_2.reshape([-1, mult_2.shape[-1]])

        pmf_1 = np.array([
            np.convolve(mult_1[i], mult_2[i])
            for i in range(a_size * b_size)
        ]).reshape([a_size, b_size, -1])  # shape: (a, b, c)
        del x, y
    elif model == 4:
        a, b, c = cartesian_product(a_range, b_range, c_range)
        pmf_1 = st.poisson.pmf(c, a * params["p1"] + b * params["p2"]).reshape([a_size, b_size, c_size])  # shape: (a, b, c)
        del a, b, c
    else:
        raise ValueError("Unsupported model {}".format(model))

    c, d = cartesian_product(c_range, d_range)
    pmf_2 = st.binom.pmf(d - c, c, params["p3"]).reshape([c_size, d_size])  # shape: (c, d)
    pmf_2 = expand_first(pmf_2, b_size)  # shape: (b, c, d)

    result = np.array([[
            (pmf_1[a] * pmf_2[:, :, d]).sum(axis=-1)
            for d in range(d_size)
        ]
        for a in range(a_size)
    ])  # shape: (a, d, b)
    return result.transpose([0, 2, 1])


def pb_ad(a, d, params, model):
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    flat_d = d.reshape([-1])
    probs = pb_ad_numerator(a, flat_d, params, model)  # shape: (a, b, k_d * N)
    probs = probs.reshape(probs.shape[:-1] + d.shape)  # shape: (a, b, k_d, N)
    probs = probs.prod(axis=-1)  # shape: (a, b, k_d)
    probs = probs.transpose([1, 0, 2])
    denum = probs.sum(axis=0)
    return probs / denum, b_range


def generate(N, a, b, params, model):
    a_vals = a
    b_vals = b
    a, b = cartesian_product(a_vals, b_vals)
    if model == 3:
        c = st.binom.rvs(a, params["p1"], size=(N, len(a))) + st.binom.rvs(b, params["p2"], size=(N, len(a)))  # shape: (N, a * b)
        c = np.reshape(c, [N, len(a_vals), len(b_vals)])
    elif model == 4:
        c = st.poisson.rvs(a * params["p1"] + b * params["p2"], size=(N, len(a)))
        c = np.reshape(c, [N, len(a_vals), len(b_vals)])
    return c + st.binom.rvs(c, params["p3"])
