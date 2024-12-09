from collections import Counter

import numpy as np


def dirichlet(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: set,
    client_num: int,
    alpha: float,
    least_samples: int,
    partition: dict,
    stats: dict,
):
    """Partition dataset according to Dirichlet with concentration parameter
    `alpha`.

    Args:
        targets (np.ndarray): Data label array.
        target_indices (np.ndarray): Indices of targets. If you haven't set `--iid`, then it will be np.arange(len(targets))
        Otherwise, it will be the absolute indices of the full targets.
        label_set (set): Label set.
        client_num (int): Number of clients.
        alpha (float): Concentration parameter. Smaller alpha indicates strong data heterogeneity.
        least_samples (int): Lease number of data samples each client should have.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.
    """
    partition["data_indices"][:client_num] = [[] for _ in range(client_num)]
    n_data_per_client = int((len(targets)) / client_num)
    # Draw from lognormal distribution
    client_data_list = np.random.lognormal(mean=np.log(n_data_per_client), sigma=0, size=client_num)
    client_data_list = (client_data_list / np.sum(client_data_list) * len(targets)).astype(int)
    # Add/Subtract the excess number starting from first client
    diff = np.sum(client_data_list) - len(targets)
    if diff != 0:
        for client_i in range(client_num):
            if client_data_list[client_i] > diff:
                client_data_list[client_i] -= diff
                break
    num_classes = len(label_set)
    class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=client_num)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]
    while np.sum(client_data_list) != 0:
        i = np.random.randint(client_num)
        # If current node is full resample a client
        if client_data_list[i] <= 0:
            continue
        client_data_list[i] -= 1
        curr_prior = prior_cumsum[i]
        while True:
            class_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if train_y is out of that class
            if class_amount[class_label] <= 0:
                continue
            class_amount[class_label] -= 1
            partition["data_indices"][i].append(idx_list[class_label][class_amount[class_label]])
            break

    for i in range(client_num):
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))
        partition["data_indices"][i] = target_indices[partition["data_indices"][i]]

    sample_num = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": sample_num.mean().item(),
        "stddev": sample_num.std().item(),
    }