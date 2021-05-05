# -*- coding: utf-8 -*-
"""
Use this file as an template for your project 2
"""
import numpy as np
import cvxpy as cp
import math
from random import shuffle


def generator(n, prob_inf, T):
    ppl = np.random.binomial(size=n, n=1, p= prob_inf)    # ppl is the population


    col_weight = math.ceil(math.log(2)*T/(n*prob_inf))
    X = np.zeros((T,n))
    X[0:col_weight,:] = 1
    idx = np.random.rand(*X.shape).argsort(0)
    X = X[idx, np.arange(X.shape[1])]
    y_temp = X@ppl #result vector
    y = np.ones_like(y_temp)*(y_temp>=1) #test results

    return X,ppl, y #return population and test results

def generator_nonoverlapping(n, q, p, m, T):

    ppl = np.zeros(n)    # ppl is the population
    A = np.zeros((m,n)) #family structure matrix
    A[0:1,:] = 1
    idx = np.random.rand(*A.shape).argsort(0)
    A = A[idx, np.arange(A.shape[1])]

    inf_families = np.random.binomial(size=m, n=1, p= q)

    for i in range(m):
        if inf_families[i] == 1:     #check if family is infected
            indices = A[i,:] == 1    #find the family members
            binom = np.random.binomial(size=np.sum(indices),n=1, p=p)
            ppl[indices] = (ppl[indices] + binom)>0


    col_weight = math.ceil(math.log(2)*T/(n*q*p))
    X = np.zeros((T,n))
    X[0:col_weight,:] = 1
    idx = np.random.rand(*X.shape).argsort(0)
    X = X[idx, np.arange(X.shape[1])]
    y_temp = X@ppl
    y = np.ones_like(y_temp)*(y_temp>=1) #test results

    return X, ppl, y, A   #return family structured population, family assignment vector, test results

def add_noise_zchannel(y, p_noisy):

    y_noisy = np.zeros_like(y)
    indices = y==1
    noise_mask = np.random.binomial(size=np.sum(indices),n=1, p=1-p_noisy)
    y_noisy[indices] = y[indices]*noise_mask

    return y_noisy

def add_noise_bsc(y, p_noisy):

    y_noisy = np.zeros_like(y)
    noise_mask = np.random.binomial(size=y.shape[0],n=1, p=p_noisy)
    y_noisy = (y+noise_mask)%2

    return y_noisy


def lp(X,y):
    """
    This solves the SSS problem for test matrix X
    and test results vector y. Returns the solution
    to the relaxation.
    """
    n,m = X.shape
    z = cp.Variable(m)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)

    constraints = []
    if num_pos > 0:
        constraints.append(X[y == 1] @ z >= np.ones(num_pos))
    if num_neg > 0:
        constraints.append(X[y == 0] @ z == np.zeros(num_neg))
    
    constraints.append(z >= np.zeros(m))
    constraints.append(z <= np.ones(m))

    prob = cp.Problem(cp.Minimize(np.ones(m).T@z),
                      constraints)
    prob.solve(solver='ECOS')

    return rounding_scheme(X, y, z.value)

def lp_nonoverlapping(X,y,A):
    """
    Extension of the original LP with addition to the objective to minimize
    the number of families with an infected person.
    """
    num_families = A.shape[0]
    num_test, num_people = X.shape
    # Indicator variable for if person is infected
    z = cp.Variable(num_people)
    # Indicator if any person for a family is infected
    w = cp.Variable(num_families)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)

    R = 20 # Family positive weight

    # contraints to make sure at least one person for a positive test in infected, and no one from negative tests
    constraints = []
    if num_pos > 0:
        constraints.append(X[y == 1] @ z >= np.ones(num_pos))
    if num_neg > 0:
        constraints.append(X[y == 0] @ z == np.zeros(num_neg))
    # Make sure that z is between 0 and 1
    constraints.append(z >= np.zeros(num_people))
    constraints.append(z <= np.ones(num_people))

    # family contraints
    # make sure that w is between 0 and 1
    constraints.append(w >= np.zeros(num_families))
    constraints.append(w <= np.ones(num_families))
    # contraints to ensure that if a family is not infected a person from that family cant be infected either
    for family in range(num_families):
        for person in range(num_people):
            if A[family, person] == 1:
                constraints.append(z[person] <= w[family])

    # Objective function to minimize infected people as well as infected families
    prob = cp.Problem(cp.Minimize(np.ones(num_people).T @ z + R * np.ones(num_families).T @ w),
                      constraints)
    prob.solve(solver='ECOS')
    ppl_pred = rounding_scheme(X, y, z.value)

    return ppl_pred

def lp_noisy_z(X,y):
    """
    Solve the SSS problem where false negatives are possible.
    L is the penalty for false negatives.
    """
    L = 1

    n, m = X.shape
    z = cp.Variable(m)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)
    e_neg = cp.Variable(num_neg)

    j = 0
    constraints = []
    constraints.append(X[y == 1] @ z >= np.ones(num_pos))
    constraints.append(X[y == 0] @ z >= e_neg)

    constraints.append(z >= np.zeros(m))
    constraints.append(z <= np.ones(m))
    constraints.append(e_neg >= np.zeros(num_neg))
    constraints.append(e_neg <= np.ones(num_neg))

    prob = cp.Problem(cp.Minimize(cp.sum(z) + L * cp.sum(e_neg)),
                      constraints)
    prob.solve(solver='ECOS')

    return threshold(z.value, 0.5)


def lp_noisy_bsc(X,y):
    """
    Solve the SSS problem where false negatives and false
    positives are possible. L is the penalty for false negatives
    and M is the penalty for false positives.
    """
    L = 1
    M = 1
    n,m = X.shape
    z = cp.Variable(m)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)
    e_neg = cp.Variable(num_neg)
    e_pos = cp.Variable(num_pos)

    j = 0
    k = 0
    constraints = []
    constraints.append(X[y == 1] @ z >= np.ones(num_pos) - e_pos)
    constraints.append(X[y == 0] @ z >= e_neg)
    constraints.append(z >= np.zeros(m))
    constraints.append(z <= np.ones(m))
    constraints.append(e_neg >= np.zeros(num_neg))
    constraints.append(e_neg <= np.ones(num_neg))
    constraints.append(e_pos >= np.zeros(num_pos))
    constraints.append(e_pos <= np.ones(num_pos))

    prob = cp.Problem(cp.Minimize(cp.sum(z) + L*cp.sum(e_neg) + M*cp.sum(e_pos)),
                      constraints)
    prob.solve(solver='ECOS');

    return threshold(z.value, 0.5)

def lp_noisy_z_nonoverlapping(X,y,A):
    """
    Extension of the alternate LP with addition to the objective to minimize
    the number of families with an infected person and z noise slacks.
    """
    R = 40
    L = 1
    num_families=A.shape[0]
    num_test,num_people = X.shape
    z = cp.Variable(num_people)
    w = cp.Variable(num_families)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)
    e_neg = cp.Variable(num_neg)
    constraints = []
    if num_pos > 0:
        constraints.append(X[y == 1] @ z >= np.ones(num_pos))
    if num_neg > 0:
        # constraints.append(X[y == 0] @ z == np.zeros(num_neg))
        constraints.append(X[y == 0] @ z >= e_neg)
    constraints.append(z >= np.zeros(num_people))
    constraints.append(z <= np.ones(num_people))
    constraints.append(e_neg >= np.zeros(num_neg))
    constraints.append(e_neg <= np.ones(num_neg))

    # family contraints
    constraints.append(w >= np.zeros(num_families))
    constraints.append(w <= np.ones(num_families))
    for family in range(num_families):
        for person in range(num_people):
            if A[family,person]==1:
                constraints.append(z[person]<=w[family])

    prob = cp.Problem(cp.Minimize(np.ones(num_people).T@z+R*np.ones(num_families).T@w + L*cp.sum(e_neg)),
                      constraints)
    prob.solve(solver='ECOS')

    return threshold(z.value, 0.5)

def lp_noisy_bsc_nonoverlapping(X,y,A):
    """
    Extension of the alternative LP with addition to the objective to minimize
    the number of families with an infected person and bidirectional noise slacks.
    """

    num_families=A.shape[0]
    num_test,num_people = X.shape
    z = cp.Variable(num_people)
    w = cp.Variable(num_families)
    num_neg = sum(y == 0)
    num_pos = sum(y == 1)
    e_neg = cp.Variable(num_neg)
    e_pos = cp.Variable(num_pos)
    constraints = []
    if num_pos > 0:
        # constraints.append(X[y == 1] @ z >= np.ones(num_pos))
        constraints.append(X[y == 1] @ z >= np.ones(num_pos) - e_pos)
    if num_neg > 0:
        # constraints.append(X[y == 0] @ z == np.zeros(num_neg))
        constraints.append(X[y == 0] @ z >= e_neg)
    constraints.append(z >= np.zeros(num_people))
    constraints.append(z <= np.ones(num_people))
    constraints.append(e_neg >= np.zeros(num_neg))
    constraints.append(e_neg <= np.ones(num_neg))
    constraints.append(e_pos >= np.zeros(num_pos))
    constraints.append(e_pos <= np.ones(num_pos))
    
    K = X.sum(axis=0)[0]
    T = num_test
    n = num_people
    pred_inf = n*T/K * (1 - (1 - num_pos/T)**(1/n))
    constraints.append(cp.sum(z) >= pred_inf)

    R = 30
    L = 1
    M = 100

    # family contraints
    constraints.append(w >= np.zeros(num_families))
    constraints.append(w <= np.ones(num_families))
    for family in range(num_families):
        for person in range(num_people):
            if A[family,person]==1:
                constraints.append(z[person]<=w[family])

    prob = cp.Problem(cp.Minimize(np.ones(num_people).T@z+R*np.ones(num_families).T@w + L*cp.sum(e_neg)+M*cp.sum(e_pos)),
                      constraints)
    prob.solve(solver='ECOS')

    bound = np.partition(z.value,-int(round(pred_inf)))[-int(round(pred_inf))]
    return threshold(z.value, bound)


def threshold(z, tau):
    pred = z.copy()
    pred[z >= tau] = 1
    pred[z < tau] = 0
    return pred


def rounding_scheme(X, y, rel):
    """
    This function implements the rounding scheme discussed in section
    Dynamic Rounding II of the report.

    Args:
        X: The test matrix
        y: The test results
        rel: Solution to the relaxed LP
    """

    # First we determine which constraints are already satisfied.
    satisfied_rows = y == 0
    for i in range(len(y)):
        if y[i] == 1:
            tested_ppl = X[i].nonzero()
            if np.any(rel[tested_ppl] == 1):
                satisfied_rows[i] = True
    
    # Then we take the subset of constraints which are not yet satisfied
    # We try to satisfy each constraint by setting a person in the test
    # to be positive
    unsatisfied_rows = (~satisfied_rows)
    unset_people = ((rel != 0) & (rel != 1))
    mask = (X[unsatisfied_rows].sum(axis=0) > 0) & unset_people
    while np.any(unsatisfied_rows):
        # Until each constraint is satisfied, we choose more people to be infected
        # We do this by selecting the person with highest relaxation value
        mask = (X[unsatisfied_rows].sum(axis=0) > 0) & unset_people
        person = mask.nonzero()[0][rel[mask].argmax()]
        rel[person] = 1
        unset_people[person] = False
        satisfied_rows[X[:,person] == 1] = True
        unsatisfied_rows = ~satisfied_rows
    rel[rel < 1] = 0
    return rel

if __name__ == '__main__':

    #Change these values according to your needs, you can also define new variables.
    n = 1000                      # size of population
    m = 300                          #number of families
    p = 0.8                         #probability of infection
    q = 0.1                         #probability of a family to be chosen as infected
    T = 300                         #number of tests
    p_noisy = 0.1                   #test noise
    X, ppl, y, A = generator_nonoverlapping(n,q,p,m,T)
    ppl_pred=alternate_lp(X,y,A)
