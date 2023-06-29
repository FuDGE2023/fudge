import matplotlib.pyplot as plt
import numpy as np


def power_law(degrees):
    from collections import Counter

    c = Counter(degrees)

    plt.figure()
    plt.plot(*zip(*sorted(c.items())), '+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    plt.show()


def plot_degree_dist(G):
    print('In degree')
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    print('Mean: ', np.mean(in_degrees))
    print('Std: ', np.std(in_degrees))
    plt.hist(in_degrees)
    plt.show()

    print('*' * 40)
    print('Out degree')

    out_degrees = [G.out_degree(n) for n in G.nodes()]
    print('Mean: ', np.mean(out_degrees))
    print('Std: ', np.std(out_degrees))
    plt.hist(out_degrees)
    plt.show()

    print('Power-law (?)')
    power_law(in_degrees)
    power_law(out_degrees)


def cumulative_distribution(G):
    print('In degrees')
    degrees = [G.in_degree(n) for n in G.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf = np.cumsum(c[:, 1][::-1])[::-1]

    plt.figure()
    plt.scatter(c[:, 0], cdf, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('CDF')
    plt.show()

    print('Out degrees')
    degrees = [G.out_degree(n) for n in G.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf = np.cumsum(c[:, 1][::-1])[::-1]

    plt.figure()
    plt.scatter(c[:, 0], cdf, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('CDF')
    plt.show()


def cumulative_distribution_all(G_pred, G_true, dataset, title_in=None, title_out=None):
    import matplotlib

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    print('In degrees')

    degrees = [G_true.in_degree(n) for n in G_true.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c_true = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf_true = np.cumsum(c_true[:, 1][::-1])[::-1]

    degrees = [G_pred.in_degree(n) for n in G_pred.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c_pred = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf_pred = np.cumsum(c_pred[:, 1][::-1])[::-1]

    plt.figure(figsize=(10, 7))
    plt.scatter(c_true[:, 0], cdf_true, s=150, label='True test set')
    plt.scatter(c_pred[:, 0], cdf_pred, s=150, alpha=0.5, label='Generated test set')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree', fontsize=25)
    plt.ylabel('CDF', fontsize=25)
    plt.title(dataset + ': In-Degree Distribution', fontsize=25)
    plt.legend()

    if title_in != None:
        plt.savefig(title_in, bbox_inches='tight')

    plt.show()

    print('Out degrees')
    degrees = [G_true.out_degree(n) for n in G_true.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c_true = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf_true = np.cumsum(c_true[:, 1][::-1])[::-1]

    degrees = [G_pred.out_degree(n) for n in G_pred.nodes()]

    degrees, counts = np.unique(degrees, return_counts=True)
    c_pred = np.array([(d, o) for d, o in sorted(zip(degrees, counts), key=lambda x: x[0])])
    cdf_pred = np.cumsum(c_pred[:, 1][::-1])[::-1]

    plt.figure(figsize=(10, 7))
    plt.scatter(c_true[:, 0], cdf_true, s=150, label='True test set')
    plt.scatter(c_pred[:, 0], cdf_pred, s=150, alpha=0.5, label='Generated test set')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree', fontsize=25)
    plt.ylabel('CDF', fontsize=25)
    plt.title(dataset + ': Out-Degree Distribution', fontsize=25)
    plt.legend()

    if title_out != None:
        plt.savefig(title_out, bbox_inches='tight')

    plt.show()


def power_law_plot(G, title_in=None, title_out=None):
    from collections import Counter
    print('In degrees')
    degrees = [G.in_degree(n) for n in G.nodes()]

    c = Counter(degrees)

    plt.figure()
    plt.plot(*zip(*sorted(c.items())), '+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    if title_in != None:
        plt.savefig(title_in, bbox_inches='tight')
    plt.show()

    print('Out degrees')
    degrees = [G.out_degree(n) for n in G.nodes()]

    c = Counter(degrees)

    plt.figure()
    plt.plot(*zip(*sorted(c.items())), '+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    if title_out != None:
        plt.savefig(title_out, bbox_inches='tight')
    plt.show()
