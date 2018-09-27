import numpy as np
import scipy.stats as st


def pa(params, model):
    k_a = params["amax"] - params["amin"] + 1
    return np.full([k_a], 1.0 / k_a), np.arange(params["amin"], params["amax"] + 1)


def pb(params, model):
    k_b = params["bmax"] - params["bmin"] + 1
    return np.full([k_b], 1.0 / k_b), np.arange(params["bmin"], params["bmax"] + 1)


def get_pc_ab(c, a, b, params, model):
    return np.sum([
        st.binom.pmf(t, a, params["p1"]) * st.binom.pmf(c - t, b, params["p2"])
        for t in range(a)
    ])


def pc(params, model):
    c_range = np.arange(params["bmax"] + params["amax"])
    return np.array([np.sum([
        a_prob * b_prob * get_pc_ab(c, a_value, b_value, params, model)
        for a_prob, a_value in zip(*pa(params, model))
        for b_prob, b_value in zip(*pb(params, model))
    ]) for c in c_range]), c_range


def generate(N, a, b, params, model):
    c = [[
            np.random.binomial(cur_a, params["p1"], size=N) + np.random.binomial(cur_b, params["p2"], size=N)
            for cur_a in a
        ] for cur_b in b
    ]
    d = [[
        c[i][j] + np.random.binomial(c[i][j], params["p3"])
            for i, cur_a in enumerate(a)
        ] for j, cur_b in enumerate(b)
    ]
    return d
