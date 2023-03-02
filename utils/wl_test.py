import numpy as np
from six import itervalues

from grakel.kernels.vertex_histogram import VertexHistogram
from grakel.graph import Graph

from scipy.special import rel_entr


def generate_graphs(label_count, WL_labels_inverse, n_iter, nx, L, Gs_ed, extras, _inv_labels, in_edges_mapping=False,
                    max_diameter=5):
    count_colors_before = 0
    count_colors_now = 0
    count_colors = True
    count_colors_iteration_stop = -1

    check_diameter = True
    diameter_iteration_stop = -1
    new_graphs = list()
    for j in range(nx):
        new_labels = dict()
        for k in L[j].keys():
            new_labels[k] = WL_labels_inverse[L[j][k]]
        L[j] = new_labels
        # add new labels
        new_graphs.append((Gs_ed[j], new_labels) + extras[j])
    yield new_graphs, count_colors_iteration_stop, diameter_iteration_stop

    for i in range(1, n_iter):
        if i == max_diameter + 1 and check_diameter:  # max_diameter+1 because the loop starts from 1
            # print("DIAMETER ARRIVED")
            diameter_iteration_stop = i
            check_diameter = False
        label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
        for j in range(nx):
            # Find unique labels and sort
            # them for both graphs
            # Keep for each node the temporary
            L_temp[j] = dict()

            for v in Gs_ed[j].keys():
                credential = str(L[j][v]) + "," + str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))

                if in_edges_mapping:
                    in_edges = []
                    for n in Gs_ed[j].keys():
                        if v in Gs_ed[j][n].keys():
                            in_edges.append(L[j][n])
                    credential += "," + str(sorted(in_edges))

                L_temp[j][v] = credential
                label_set.add(credential)

        label_list = sorted(list(label_set))
        for dv in label_list:
            WL_labels_inverse[dv] = label_count
            label_count += 1

        count_colors_now = len(WL_labels_inverse.keys())
        if count_colors_now == count_colors_before and count_colors:
            # print("SAME COLOUR MAPPINGS")
            count_colors_iteration_stop = i
            count_colors = False

        count_colors_before = count_colors_now

        # Recalculate labels
        new_graphs = list()
        for j in range(nx):
            new_labels = dict()
            for k in L_temp[j].keys():
                new_labels[k] = WL_labels_inverse[L_temp[j][k]]
            L[j] = new_labels
            # relabel
            new_graphs.append((Gs_ed[j], new_labels) + extras[j])
        _inv_labels[i] = WL_labels_inverse
        yield new_graphs, count_colors_iteration_stop, diameter_iteration_stop


def compute_metric(kernels, similarity_metric="cosine", steps=50):
    K = np.sum(np.array(kernels), axis=0)
    similarities = np.zeros(K.shape)

    # normalizer = Normalizer()

    if similarity_metric == "cosine":
        _X_diag = np.diagonal(K)
        similarities = K / np.sqrt(np.outer(_X_diag, _X_diag))
        similarities = np.around(similarities, decimals=4)

    elif similarity_metric == "kl":
        for i in range(len(K)):
            for j in range(len(K)):
                kl_1 = sum(rel_entr(K[i], K[j]))
                kl_2 = sum(rel_entr(K[j], K[i]))
                similarities[i][j] = np.around((kl_1 + kl_2) / 2, decimals=4)
    #         similarities = normalizer.fit_transform(similarities)

    elif similarity_metric == "l2":
        for i in range(len(K)):
            for j in range(len(K)):
                similarities[i][j] = np.around(np.linalg.norm(K[i] - K[j]), decimals=4)
    #         similarities = normalizer.fit_transform(similarities)

    return similarities


def compute_similarity(X, n_iter=10, graph_format="dictionary", similarity_metric="cosine", max_diameter=5,
                       in_edges_mapping=False):
    nx = 0
    Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
    Xs = []

    for (idx, x) in enumerate(iter(X)):

        if len(x) > 2:
            extra = tuple()
            if len(x) > 3:
                extra = tuple(x[3:])
            x = Graph(x[0], x[1], x[2], graph_format=graph_format)
            extra = (x.get_labels(purpose=graph_format,
                                  label_type="edge", return_none=True),) + extra
        else:
            x = Graph(x[0], x[1], {}, graph_format=graph_format)
            extra = tuple()

        Xs.append(x)
        Gs_ed[nx] = x.get_edge_dictionary()
        L[nx] = x.get_labels(purpose="dictionary")
        extras[nx] = extra
        distinct_values |= set(itervalues(L[nx]))
        nx += 1

    # Save the number of "fitted" graphs.
    _nx = nx

    # get all the distinct values of current labels
    WL_labels_inverse = dict()

    # assign a number to each label
    label_count = 0
    for dv in sorted(list(distinct_values)):
        WL_labels_inverse[dv] = label_count
        label_count += 1

    # Initalize an inverse dictionary of labels for all iterations
    _inv_labels = dict()
    _inv_labels[0] = WL_labels_inverse

    base_kernels = {i: VertexHistogram() for i in range(n_iter)}
    generated_graphs_kernels = []

    same_coloring_step = 1
    max_diameter_step = 1


    for (i, values) in enumerate(generate_graphs(label_count, WL_labels_inverse, n_iter, nx, L, Gs_ed, extras, _inv_labels,
                                                 in_edges_mapping=in_edges_mapping, max_diameter=max_diameter)):

        g, same_coloring_step_temp, max_diameter_step_temp = values

        if same_coloring_step_temp != -1:
            same_coloring_step = same_coloring_step_temp
        if max_diameter_step_temp != -1:
            max_diameter_step = max_diameter_step_temp

        k = base_kernels[i].fit_transform(g)
        generated_graphs_kernels.append(k)

    all_steps = compute_metric(kernels=generated_graphs_kernels, similarity_metric=similarity_metric, steps=n_iter)
    same_coloring = compute_metric(kernels=generated_graphs_kernels[:same_coloring_step], similarity_metric=similarity_metric, steps=n_iter)
    max_diameter = compute_metric(kernels=generated_graphs_kernels[:max_diameter_step], similarity_metric=similarity_metric, steps=n_iter)

    return all_steps, same_coloring, max_diameter