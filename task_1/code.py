import numpy as np
import scipy.stats as st


def pa(params, model):
    k_a = params["amax"] - params["amin"] + 1
    return np.full([k_a], 1.0 / k_a), np.arange(params["amin"], params["amax"] + 1)


def pb(params, model):
    k_b = params["bmax"] - params["bmin"] + 1
    return np.full([k_b], 1.0 / k_b), np.arange(params["bmin"], params["bmax"] + 1)


def cartesian_product(*arrays):
    mesh = np.meshgrid(*arrays)
    mesh = tuple(map(lambda tensor: np.reshape(tensor, [-1]), mesh))
    return np.dstack(mesh).reshape(-1, len(arrays))


def pc(params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)
    if model == 3:
        a_t_cartesian = cartesian_product(a_range, c_range)
        mult_1 = st.binom.pmf(a_t_cartesian[:, 1], a_t_cartesian[:, 0], params["p1"])
        b_t_cartesian = cartesian_product(b_range, np.arange(c_max, -1, -1))
        mult_2 = st.binom.pmf(b_t_cartesian[:, 1], b_t_cartesian[:, 0], params["p2"])

        mult_1 = np.reshape(mult_1, [len(c_range), len(a_range)])
        mult_2 = np.reshape(mult_2, [len(c_range), len(b_range)])
        mult_1 = np.sum(mult_1, axis=1)
        mult_2 = np.sum(mult_2, axis=1)
        probs = np.cumsum(mult_1 * mult_2)
        return np.clip(probs, 1e-100, 1) / len(a_range) / len(b_range), c_range
    elif model == 4:
        a_b_c_cartesian = cartesian_product(a_range, b_range, c_range)
        a = a_b_c_cartesian[:, 0]
        b = a_b_c_cartesian[:, 1]
        c = a_b_c_cartesian[:, 2]
        pmf = st.poisson.pmf(c, a * params["p1"] + b * params["p2"])
        pmf = np.reshape(pmf, [len(b_range), len(a_range), len(c_range)])

        probs = np.sum(pmf, axis=(0, 1))
        return probs / len(a_range) / len(b_range), c_range
    else:
        raise ValueError("Unsupported model {}".format(model))


def pd(params, model):
    c_probs, c_vals = pc(params, model)
    d_range = np.arange(0, 2 * (params["amax"] + params["bmax"]) + 1)
    probs = np.array([
        np.sum(st.binom.pmf(d, c_vals, params["p3"]) * c_probs)
        for d in d_range
    ])
    return probs, d_range


def pb_d(d, params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)

    start_d_shape = np.shape(d)
    d = np.reshape(d, [-1])
    d_c_cartesian = cartesian_product(d, c_range)
    raw_d_probs = st.binom.pmf(d_c_cartesian[:, 0], d_c_cartesian[:, 1], params["p3"])
    raw_d_probs = np.reshape(raw_d_probs, [len(c_range), len(d)])
    d_probs = np.repeat(np.reshape(raw_d_probs, [len(c_range), len(d), 1]), len(b_range), axis=-1)
    d_probs = np.transpose(d_probs, [0, 2, 1])

    if model == 3:
        a_t_cartesian = cartesian_product(a_range, c_range)
        mult_1 = st.binom.pmf(a_t_cartesian[:, 1], a_t_cartesian[:, 0], params["p1"])
        mult_1 = np.reshape(mult_1, [len(c_range), len(a_range)])
        mult_1 = np.sum(mult_1, axis=1)

        b_t_cartesian = cartesian_product(b_range, np.arange(c_max, -1, -1))
        mult_2 = st.binom.pmf(b_t_cartesian[:, 1], b_t_cartesian[:, 0], params["p2"])
        mult_2 = np.reshape(mult_2, [len(c_range), len(b_range)])

        mult_1 = np.repeat(np.reshape(mult_1, [-1, 1]), len(b_range), axis=1)
        mult_2 = np.cumsum(mult_1 * mult_2, axis=0)

    elif model == 4:
        a_c_b_cartesian = cartesian_product(a_range, c_range, b_range)
        a = a_c_b_cartesian[:, 0]
        c = a_c_b_cartesian[:, 1]
        b = a_c_b_cartesian[:, 2]
        pmf = st.poisson.pmf(c, a * params["p1"] + b * params["p2"])
        pmf = np.reshape(pmf, [len(c_range), len(a_range), len(b_range)])

        mult_2 = np.sum(pmf, axis=1)
    else:
        raise ValueError("Unsupported model {}".format(model))

    mult_2 = np.repeat(np.reshape(mult_2, np.shape(mult_2) + (1,)), len(d), axis=-1)
    b_probs = np.sum(mult_2 * d_probs, axis=0)
    c_probs, c_vals = pc(params, model)
    c_probs = np.repeat(np.reshape(c_probs, [-1, 1]), len(d), axis=1)
    denominator = np.sum(c_probs * raw_d_probs, axis=0)

    b_probs = b_probs / denominator

    b_probs = np.reshape(b_probs, np.shape(b_probs)[:-1] + start_d_shape)
    b_probs = np.prod(b_probs, axis=-1)
    return b_probs, b_range


def pb_ad(a, d, params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_max = params["bmax"] + params["amax"]
    c_range = np.arange(0, c_max + 1)
    if model == 3:
        a_b_c_cartesian = cartesian_product(a_range, b_range, c_range)
        a = a_b_c_cartesian[:, 0]
        c = a_b_c_cartesian[:, 1]
        b = a_b_c_cartesian[:, 2]


def main():
    params = {
        'amin': 5, 'amax': 9, 'bmin': 50, 'bmax': 60,
        'p1': 0.1, 'p2': 0.01, 'p3': 0.3
    }
    model = 3
    print(pc(params, model)[0])


main()
