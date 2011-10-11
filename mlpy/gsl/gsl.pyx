from cgsl cimport *


def sf_gamma (double x):
    return gsl_sf_gamma (x)

def sf_fact (unsigned int n):
    return gsl_sf_fact (n)
