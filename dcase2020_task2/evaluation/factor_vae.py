import numpy as np


def factor_vae(
        data_set,
        representation_function,
        random_state=np.random.RandomState(0),
        batch_size=64,
        num_train=10_000,
        num_eval=5_000,
        num_variance_estimate=10_000,
        valid_dimension_threshold: float = 0.05
):
    global_variances = _compute_variances(
        data_set,
        representation_function,
        num_variance_estimate,
        random_state
    )

    active_dims = _prune_dims(global_variances, threshold=valid_dimension_threshold)
    scores_dict = {}

    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0.
        scores_dict["score"] = 0.
        return scores_dict

    training_votes = _generate_training_batch(
        data_set,
        representation_function,
        batch_size,
        num_train,
        random_state,
        global_variances,
        active_dims
    )
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

    eval_votes = _generate_training_batch(
        data_set,
        representation_function, batch_size,
        num_eval, random_state,
        global_variances, active_dims
    )

    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    scores_dict["score"] = eval_accuracy
    return scores_dict


def _prune_dims(variances, threshold=0.):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(
        data_set,
        representation_function,
        sample_size,
        random_state,
        batch_size=64
):
    observations, _ = data_set.sample(sample_size, random_state)
    representations = obtain_representation(observations, representation_function, batch_size)
    return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(
        data_set,
        representation_function,
        batch_size,
        random_state,
        global_variances,
        active_dims
):
    # Select random coordinate to keep fixed.
    factor_index = random_state.randint(data_set.num_factors)
    # Sample two mini batches of latent variables.
    factors = data_set.sample_factors(batch_size, random_state)
    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]
    # Obtain the observations.
    observations = data_set.sample_observations(factors)
    representations = representation_function(observations)
    local_variances = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
    return factor_index, argmin


def _generate_training_batch(
        data_set,
        representation_function,
        batch_size,
        num_points,
        random_state,
        global_variances,
        active_dims
):
    votes = np.zeros((data_set.num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        factor_index, argmin = _generate_training_sample(
            data_set,
            representation_function,
            batch_size,
            random_state,
            global_variances,
            active_dims
        )
        votes[factor_index, argmin] += 1
    return votes


def obtain_representation(
        observations,
        representation_function,
        batch_size
):

    representations = []
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        representations.append(representation_function(current_observations))
        i += num_points_iter

    representations = np.concatenate(representations)

    return representations