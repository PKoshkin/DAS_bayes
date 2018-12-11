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
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)

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
    d_range = np.arange(0, 2 * (params["amax"] + params["bmax"]) + 1)

    probs = np.array([
        np.sum(st.binom.pmf(d - c_vals, c_vals, params["p3"]) * c_probs)
        for d in d_range
    ])
    return probs, d_range


def pb_d(d, params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)

    flat_d = np.reshape(d, [-1])

    a_size, b_size, c_size, d_size = len(a_range), len(b_range), len(c_range), len(flat_d)
    k_d, N = d.shape

    if model == 3:
        a, b, c = cartesian_product(a_range, b_range, c_range)
        pmf_1 = st.poisson.pmf(c, a * params["p1"] + b * params["p2"]).reshape([a_size, b_size, c_size])  # shape: (a, b, c)
        mult_1 = pmf_1.sum(axis=0)  # shape: (b, c)
    elif model == 4:
        a, b, c = cartesian_product(a_range, b_range, c_range)
        pmf_1 = st.poisson.pmf(c, a * params["p1"] + b * params["p2"]).reshape([a_size, b_size, c_size])  # shape: (a, b, c)
        mult_1 = pmf_1.sum(axis=0)  # shape: (b, c)
    else:
        raise ValueError("Unsupported model {}".format(model))

    mult_1 = expand_last(mult_1, d_size)  # shape: (b, c, d)

    c, d = cartesian_product(c_range, flat_d)
    mult_2 = st.binom.pmf(d - c, c, params["p3"]).reshape([c_size, d_size])  # shape: (c, d)
    mult_2 = expand_first(mult_2, b_size)  # shape: (b, c, d)

    probs = np.sum(mult_1 * mult_2, axis=1)  # shape: (b, d)
    probs = probs.reshape([b_size, k_d, N]).prod(axis=-1)  # shape: (b, k_d)

    d_probs, _ = pd(params, model)  # shape: (d_distr,)
    d_probs = d_probs[flat_d].reshape([k_d, N])  # shape: (k_d, N)

    d_probs = expand_first(d_probs, b_size).prod(axis=-1)  # shape: (b, k_d)

    probs /= d_probs
    denum = probs.sum(axis=0)

    return probs / denum, b_range


def pb_ad(a, d, params, model):
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)
    a_vals = a
    d_vals = np.reshape(d, [-1])
    if model == 3:
        a, b, t = cartesian_product(a_vals, b_range, c_range)
        mult_1 = st.binom.pmf(t, a, params["p1"])
        mult_1 = np.reshape(mult_1, [len(a_vals), len(b_range), len(c_range)])
        mult_2 = st.binom.pmf(np.flip(t, axis=0), np.flip(b, axis=0), params["p2"])
        mult_2 = np.reshape(mult_2, [len(a_vals), len(b_range), len(c_range)])
        pc_ab = np.cumsum(mult_1 * mult_2, axis=-1)  # shape: (a, b, c)
    elif model == 4:
        a, b, c = cartesian_product(a_vals, b_range, c_range)
        pc_ab = st.poisson.pmf(c, a * params["p1"] + b * params["p2"])
        pc_ab = np.reshape(pc_ab, [len(a_vals), len(b_range), len(c_range)])  # shape: (a, b, c)
    else:
        raise ValueError("Unsupported model {}".format(model))
    c_d = cartesian_product(c_range, d_vals)
    pd_c = st.binom.pmf(c_d[0] - c_d[1], c_d[1], params["p3"])
    pd_c = np.reshape(pd_c, [len(c_range), len(d_vals)])
    pd_c = expand_first(pd_c, len(b_range))
    pd_c = expand_first(pd_c, len(a_vals))  # shape: (a, b, c, d)

    pc_ab = expand_last(pc_ab, len(d_vals))  # shape: (a, b, c, d)

    numerator = np.sum(pd_c * pc_ab, axis=2)  # shape: (a, b, d)
    denominator = np.sum(numerator, axis=1)  # shape: (a, d)
    denominator = np.transpose(expand_last(denominator, len(b_range)), [0, 2, 1])  # shape: (a, b, d)

    result = numerator / np.clip(denominator, 1e-100, 1)  # shape: (a, b, d)
    result = np.transpose(result, [1, 0, 2])  # shape: (b, a, d)
    result = np.reshape(result, np.shape(result)[:-1] + np.shape(d))
    return np.prod(result, axis=-1), b_range


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
