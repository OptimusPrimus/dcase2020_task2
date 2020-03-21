import numpy as np
from sklearn.metrics import mutual_info_score


def mig(data_set,
        representation_function,
        random_state=np.random.RandomState(0),
        num_train=10_000,
        batch_size=64):
    mus_train, ys_train = generate_batch_factor_code(
        data_set,
        representation_function,
        num_train,
        random_state,
        batch_size
    )
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)


def _compute_mig(codes, factors):
    score_dict = {}
    discretized_mus = _histogram_discretize(codes)
    m = discrete_mutual_info(discretized_mus, factors)
    assert m.shape[0] == codes.shape[0]
    assert m.shape[1] == factors.shape[0]

    entropy = discrete_entropy(factors)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    score_dict["score"] = score_dict["discrete_mig"]
    return score_dict


def generate_batch_factor_code(
        data_set,
        representation_function,
        num_points,
        random_state,
        batch_size
):
    codes = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations, current_factors = data_set.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            codes = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            codes = np.vstack((codes, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(codes), np.transpose(factors)


def _histogram_discretize(target, num_bins=20):
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(codes, factors):
    num_codes = codes.shape[0]
    num_factors = factors.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(factors[j, :], codes[i, :])
    return m


def discrete_entropy(factors):
    num_factors = factors.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(factors[j, :], factors[j, :])
    return h
