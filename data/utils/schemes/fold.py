from collections import Counter

import numpy as np


def fold(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: set,
    client_num: int,
    alpha: float,
    least_samples: int,
    partition: dict,
    stats: dict,
    folds: int = 2,
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
    folds = np.array_split(ary=np.arange(len(label_set)), indices_or_sections=folds)

    noniid_s = 30
    s = noniid_s / 100
    num_per_user = 500
    num_classes = len(np.unique(targets))

    # -------------------------------------------------------
    # divide the first dataset that all clients share (includes all classes)
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(client_num)}
    num_samples = len(targets)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(targets)
    idxs1 = np.arange(len(targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [2000 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1) * iid_per_label

    for client_id in range(client_num):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            partition["data_indices"][client_id] = np.concatenate((partition["data_indices"][client_id], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num

        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = folds[client_id%len(folds)]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            partition["data_indices"][client_id] = np.concatenate((partition["data_indices"][client_id], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        partition["data_indices"][client_id] = partition["data_indices"][client_id].astype(int)

    for i in range(client_num):
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))
        partition["data_indices"][i] = target_indices[partition["data_indices"][i]]

    sample_num = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": sample_num.mean().item(),
        "stddev": sample_num.std().item(),
    }
