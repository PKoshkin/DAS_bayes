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
    c_range = np.arange(0, params["bmax"] + params["amax"] + 1)
    cartesian = cartesian_product(a_range, b_range, c_range)

    def is_good_row(row):
        return row[-1] <= row[0]

    bools = np.apply_along_axis(is_good_row, 1, cartesian)
    cartesian = cartesian[bools]
    bools = np.apply_along_axis(is_good_row, 1, cartesian)
    c = np.reshape(
        np.repeat(np.reshape(c_range, [-1, 1]), np.size(cartesian[:, 0:1]), axis=1),
        (len(c_range),) + np.shape(cartesian[:, 0:1])
    )
    cartesian = np.repeat(np.reshape(cartesian, (1,) + np.shape(cartesian)), len(c_range), axis=0)
    pmf_params = np.concatenate([cartesian, c], axis=-1)
    a = pmf_params[:, :, 0]
    b = pmf_params[:, :, 1]
    t = pmf_params[:, :, 2]
    c = pmf_params[:, :, 3]
    mult_1 = st.binom.pmf(t, a, params["p1"])
    mult_2 = st.binom.pmf(c - t, b, params["p2"])
    return np.sum(mult_1 * mult_2, axis=1), c_range


def main():
    params = {
        'amin': 5, 'amax': 9, 'bmin': 50, 'bmax': 60,
        'p1': 0.1, 'p2': 0.01, 'p3': 0.3
    }
    model = 3
    pc(params, model)
