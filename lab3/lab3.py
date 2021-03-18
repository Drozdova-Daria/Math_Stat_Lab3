import numpy as np
from scipy.stats import cauchy, laplace, poisson, uniform, norm
import matplotlib.pyplot as plt
from distribution import *
import csv


def selection(mu, sigma, size, distribution):
    if distribution == Distribution.NORMAL:
        return norm.rvs(mu, sigma, size)
    elif distribution == Distribution.CAUCHY:
        return cauchy.rvs(mu, sigma, size)
    elif distribution == Distribution.LAPLACE:
        return laplace.rvs(mu, sigma, size)
    elif distribution == Distribution.POISSON:
        return poisson.rvs(mu, size=size)
    elif distribution == Distribution.UNIFORM:
        return uniform.rvs(mu, sigma, size)
    else:
        return None


def distribution_function(x, mu, sigma, distribution):
    if distribution == Distribution.NORMAL:
        return norm.cdf(x, mu, sigma)
    elif distribution == Distribution.CAUCHY:
        return cauchy.cdf(x, mu, sigma)
    elif distribution == Distribution.LAPLACE:
        return laplace.cdf(x, mu, sigma)
    elif distribution == Distribution.POISSON:
        return poisson.cdf(x, mu, sigma)
    elif distribution == Distribution.UNIFORM:
        return uniform.cdf(x, mu, sigma)
    else:
        return None


def probability_function(x, mu, sigma, distribution):
    if distribution == Distribution.POISSON:
        return poisson.pmf(x, mu, sigma)
    else:
        return None


def quartiles_of_selection(sel):
    return np.percentile(sel, 25), np.percentile(sel, 75)


def moustache_borders(q_1, q_3):
    x_1 = q_1 - 1.5 * (q_3 - q_1)
    x_2 = q_3 + 1.5 * (q_3 - q_1)
    return x_1, x_2


def find_emissions(sel: list, x_1, x_2):
    emissions = []
    for x in sel:
        if x < x_1 or x > x_2:
            emissions.append(x)
    return emissions


def share_of_emissions(sel: list, x_1, x_2):
    emissions = find_emissions(sel, x_1, x_2)
    return len(emissions) / len(sel)


def boxplot_building(a, b, size: list, distribution):
    selection_one = selection(a, b, size[0], distribution)
    selection_two = selection(a, b, size[1], distribution)
    plt.boxplot((selection_one, selection_two), sym='o', labels=[str(size[0]), str(size[1])])
    plt.xlabel('n')
    plt.ylabel('x')
    plt.title(Distribution.in_str(distribution))
    plt.savefig(Distribution.in_str(distribution), format='jpg', dpi=100)
    plt.show()


def research_of_actual_emissions(a, b, size, distribution, count):
    average_share = 0
    for _ in range(count):
        sel = selection(a, b, size, distribution)
        q_1, q_3 = quartiles_of_selection(sel)
        x_1, x_2 = moustache_borders(q_1, q_3)
        share = share_of_emissions(sel.tolist(), x_1, x_2)
        average_share += share
    return average_share / count


def research_of_theory_emissions(a, b, size, distribution):
    sel = selection(a, b, size, distribution)
    q_1, q_3 = quartiles_of_selection(sel)
    x_1, x_2 = moustache_borders(q_1, q_3)
    probability = 0
    if Distribution.distribution_type(distribution) == DistributionType.CONTINUOUS:
        probability = distribution_function(x_1, a, b, distribution) + (1 - distribution_function(x_2, a, b, distribution))
    elif Distribution.distribution_type(distribution) == DistributionType.DISCRETE:
        probability = (distribution_function(x_1, a, b, distribution) - probability_function(x_1, a, b, distribution)) + \
               (1 - distribution_function(x_2, a, b, distribution))
    return q_1, q_3, x_1, x_2, probability


count = 1000
size = [20, 100]
a_parameters = [0, 0, 0, 10, -np.sqrt(3)]
b_parameters = [1, 1, np.sqrt(2), 0, np.sqrt(3)]
distributions = [Distribution.NORMAL,
                 Distribution.CAUCHY,
                 Distribution.LAPLACE,
                 Distribution.POISSON,
                 Distribution.UNIFORM]
for a, b, distribution in zip(a_parameters, b_parameters, distributions):
    boxplot_building(a, b, size, distribution)

with open('results1.csv', mode='w', encoding='utf-8') as file:
    file_writer = csv.writer(file)
    for a, b, distribution in zip(a_parameters, b_parameters, distributions):
        for n in size:
            average_share = research_of_actual_emissions(a, b, n, distribution, count)
            file_writer.writerow((Distribution.in_str(distribution), str(n), str(average_share)))
    file.close()

with open('results2.csv', mode='w', encoding='utf-8') as file:
    file_writer = csv.writer(file)
    for a, b, distribution in zip(a_parameters, b_parameters, distributions):
        for n in size:
            q1, q3, x1, x2, p = research_of_theory_emissions(a, b, n, distribution)
            file_writer.writerow((Distribution.in_str(distribution) + str(n), round(q1, 4), round(q3, 4), round(x1, 4), round(x2, 4), round(p, 4)))
    file.close()
