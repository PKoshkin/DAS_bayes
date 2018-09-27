import numpy as np
import scipy.stats as st


def pa(params, model):
    k_a = params["amax"] - params["amin"] + 1
    return np.full([k_a], 1.0 / k_a), np.arange(params["amin"], params["amax"] + 1)


def pb(params, model):
    k_b = params["bmax"] - params["bmin"] + 1
    return np.full([k_b], 1.0 / k_b), np.arange(params["bmin"], params["bmax"] + 1)


def cartesian_product(x, y, z):
    a, b, c = np.meshgrid(x, y, z)
    a = np.reshape(a, [-1])
    b = np.reshape(b, [-1])
    c = np.reshape(c, [-1])
    return np.dstack((a, b, c)).reshape(-1, 3)


def pc(params, model):
    a_range = np.arange(params["amin"], params["amax"] + 1)
    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    c_range = np.arange(0, params["bmax"] + params["amax"] + 1)

    result = []
    for c in c_range:
        abt = cartesian_product(a_range, b_range, np.arange(c))
        a = abt[:, 0]
        b = abt[:, 1]
        t = abt[:, 2]
        mult_1 = st.binom.pmf(t, a, params["p1"])
        mult_2 = st.binom.pmf(c - t, b, params["p2"])
        result.append(np.sum(mult_1 * mult_2))
    return np.array(result), c_range


def main():
    params = {
        'amin': 5, 'amax': 9, 'bmin': 50, 'bmax': 60,
        'p1': 0.1, 'p2': 0.01, 'p3': 0.3
    }
    model = 3
    pc(params, model)
