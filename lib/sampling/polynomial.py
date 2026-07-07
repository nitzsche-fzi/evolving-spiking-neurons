"""Random sparse-polynomial coefficient sampler used to seed neuron genomes."""

import torch
from math import factorial
import random
import itertools

def combination(n, k):
    """Returns the binomial coefficient ``n choose k``."""
    return factorial(n) // (factorial(k) * factorial(n - k))

def count_monomials(polynomial_dimension, degree):
    """Returns the number of monomials in ``polynomial_dimension`` variables up to ``degree``."""
    return combination(polynomial_dimension + 1 + degree, degree)

class PolynomialSampler:
    """Samples sparse coefficient tensors for a set of multivariate polynomials.

    Each polynomial gets a random number of non-zero terms (between the min/max
    term counts). Optional constraints can guarantee that one polynomial
    includes a highest-degree monomial (``enforce_degree``) and that one
    polynomial depends on the input variable x0 (``enforce_x0_dependence``), so
    sampled neurons are not trivially degenerate.
    """

    def __init__(
            self, 
            n_polynomials, 
            degree, 
            std, 
            min_terms_per_polynomial, 
            max_terms_per_polynomial, 
            enforce_degree,
            enforce_x0_dependence):
        self.n_polynomials = n_polynomials
        self.degree = degree
        self.std = std
        self.min_terms_per_polynomial = min_terms_per_polynomial
        self.max_terms_per_polynomial = max_terms_per_polynomial
        self.enforce_degree = enforce_degree
        self.enforce_x0_dependence = enforce_x0_dependence

        # Count total monomials for polynomials of up to given degree
        self.n_monomials = count_monomials(n_polynomials, degree)
        self.max_terms_per_polynomial = min(self.max_terms_per_polynomial, self.n_monomials)
        self.min_terms_per_polynomial = min(self.min_terms_per_polynomial, self.max_terms_per_polynomial)

        # Identify the highest-degree monomial indices, if needed
        if self.enforce_degree:
            start_of_highest_degree = count_monomials(n_polynomials, degree - 1)
            self.highest_degree_indices = list(range(start_of_highest_degree, self.n_monomials))

        if self.enforce_x0_dependence:
            combinations = list(itertools.chain.from_iterable(
                itertools.combinations_with_replacement(range(n_polynomials + 1), d)
                for d in range(degree + 1)
            ))
            self.indices_with_x0 = [i for i, comb in enumerate(combinations) if (0 in comb)]

    def choose_num_terms(self):
        return random.randint(self.min_terms_per_polynomial, self.max_terms_per_polynomial)

    def sample_coeffs(self):
        """Samples one sparse coefficient set.

        Returns:
            Tuple ``(coeffs, dense_coeffs, keep_mask)``: the sparsified
            coefficients, the dense (pre-masking) coefficients, and the boolean
            mask of kept terms.
        """
        keep_mask = torch.zeros(self.n_polynomials, self.n_monomials, dtype=torch.bool)

        if self.enforce_x0_dependence:
            x0_dependent_polynomial = random.randint(0,self.n_polynomials - 1)
        if self.enforce_degree:
            max_degree_polynomial = random.randint(0,self.n_polynomials - 1)

        for i in range(self.n_polynomials):
            T_i = self.choose_num_terms()  # Number of non-zero terms for polynomial i

            chosen = set()
            if self.enforce_degree:
                if i == max_degree_polynomial:
                    chosen.add(random.choice(self.highest_degree_indices))
            if self.enforce_x0_dependence:
                if i == x0_dependent_polynomial:
                    chosen.add(random.choice(self.indices_with_x0))
            #remove all chosen ones from all indices:
            all_indices = set(range(self.n_monomials)) - chosen
            n_extra = T_i - len(chosen)
            if n_extra > 0:
                chosen.update(random.sample(list(all_indices), k = T_i - len(chosen)))
            for idx in chosen:
                keep_mask[i, idx] = True

        coeffs = torch.randn(self.n_polynomials, self.n_monomials) * self.std
        dense_coeffs = coeffs.clone()
        coeffs[~keep_mask] = 0.0
        return coeffs, dense_coeffs, keep_mask
