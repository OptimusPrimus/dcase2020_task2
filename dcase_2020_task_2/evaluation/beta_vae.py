import numpy as np
from sklearn import linear_model


def beta_vae(
        data_set,
        representation_function,
        random_state=np.random.RandomState(0),
        batch_size=64,
        num_train=10_000,
        num_eval=10_000
):
    train_points, train_labels = _generate_training_batch(
        data_set,
        representation_function,
        batch_size,
        num_train,
        random_state
    )

    model = linear_model.LogisticRegression(random_state=random_state)
    model.fit(train_points, train_labels)

    train_accuracy = np.mean(model.predict(train_points) == train_labels)

    eval_points, eval_labels = _generate_training_batch(
        data_set,
        representation_function,
        batch_size,
        num_eval,
        random_state
    )

    eval_accuracy = model.score(eval_points, eval_labels)
    scores_dict = {}
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    scores_dict["score"] = eval_accuracy
    return scores_dict


def _generate_training_batch(
        data_set,
        representation_function,
        batch_size,
        num_points,
        random_state
):
    points = None
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = _generate_training_sample(data_set, representation_function, batch_size, random_state)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def _generate_training_sample(
        data_set,
        representation_function,
        batch_size,
        random_state
):
    # Select random coordinate to keep fixed.
    index = random_state.randint(data_set.num_factors)
    # Sample two mini batches of latent variables.
    factors1 = data_set.sample_factors(batch_size, random_state)
    factors2 = data_set.sample_factors(batch_size, random_state)
    # Ensure sampled coordinate is the same across pairs of samples.
    factors2[:, index] = factors1[:, index]
    # Transform latent variables to observation space.
    observation1 = data_set.sample_observations(factors1)
    observation2 = data_set.sample_observations(factors2)
    # Compute representations based on the observations.
    representation1 = representation_function(observation1)
    representation2 = representation_function(observation2)
    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
    return index, feature_vector
