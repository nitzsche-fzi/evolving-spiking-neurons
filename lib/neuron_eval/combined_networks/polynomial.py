import torch
import itertools

class Polynomial(torch.nn.Module):
    """Evaluates a population of sparse multivariate polynomials in parallel.

    Each individual in the population is described by a coefficient tensor of
    shape ``[population_size, n_polynomials, n_monomials]``. The module is used
    as the state-evolution and reset function of the PMSN-CLR neuron, so its
    input/output shapes follow the ``[batch, population, ..., feature]``
    convention used throughout the network.

    Because most coefficients are zero, a dense evaluation would waste a large
    amount of compute. Instead, the constructor analyses the coefficient tensor
    once and builds a "program" of pairwise multiplications: every monomial
    (e.g. ``x**3`` computed as ``x * x**2``) is evaluated exactly once and only
    for the individuals whose coefficients actually require it. Constant, linear
    and higher-degree terms are handled in three separate index-computation
    stages so the forward pass reduces to a sequence of gather/scatter-add
    operations.
    """

    def __init__(
            self,
            coeffs,
            n_variables,
            degree,
            batch_size,
            feature_size,
            device):
        """
        Args:
            coeffs: Coefficient tensor of shape
                ``[population_size, n_polynomials, n_monomials]``.
            n_variables: Number of input variables each polynomial takes
                (the neuron input plus its state variables).
            degree: Maximum total degree of the polynomials.
            batch_size: Number of samples processed per forward pass.
            feature_size: Feature dimensionality of the input and state.
            device: Torch device the buffers are allocated on.
        """
        super(Polynomial, self).__init__()
        self.sparse_coeffs = coeffs
        self.register_buffer('sparse_coeffs_buffer', self.sparse_coeffs)

        self.population_size, self.n_polynomials, self.n_monomials = coeffs.shape
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.n_variables = n_variables
        self.degree = degree
        self.device = device

        # The three stages below turn the dense coefficient tensor into the
        # sparse index structure that forward() uses. Stage one builds the
        # multiplication program for degree >= 2 monomials; stages two and
        # three gather the weights/scatter targets for those monomials and for
        # the linear terms respectively.
        self.combinations, self.monomial_map = self._compute_first_stage_indices()
        self.indices_individuals, self.indices_polynomials, self.weights, \
            self.scatter_indices = self._compute_second_stage_indices()
        self.linear_indices_individuals, self.linear_indices_polynomials, \
            self.linear_weights, self.linear_scatter_indices = self._compute_third_stage_indices()

        # Register one buffer per non-empty weight vector so they move with the
        # module across devices / state_dict save-load.
        for i, weight in enumerate(self.weights):
            if len(self.scatter_indices[i]) > 0:
                self.register_buffer(f'weight_{i}', weight)
        
        for i, weight in enumerate(self.linear_weights):
            if len(self.linear_scatter_indices[i]) > 0:
                self.register_buffer(f'linear_weight_{i}', weight)


    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """Evaluates the population of polynomials for one time step.

        The interface is shaped for use as the differential-equation / reset
        function inside a population of recurrent neurons.

        Args:
            x: Input tensor of shape ``(batch_size, population_size, feature_size)``.
            state: State tensor of shape
                ``(batch_size, population_size, n_variables - 1, feature_size)``.

        Returns:
            Output tensor of shape
            ``(batch_size, population_size, n_polynomials, feature_size)``.
        """

        output = self.sparse_coeffs[:,:,0][None,:,:,None].expand(
                self.batch_size, -1, -1, self.feature_size
            ).view(self.batch_size,-1,self.feature_size).clone()

        #terms is a list of monomials, e.g. x^2, y^2, x*y, y^3
        #but only computed for the individuals in the population which need it
        #so the tensors in this list all have different dim 2,
        #but the general shape of the entries is (batch_size,individuals_who_need_it,feature_size)
        terms = []
        for i,c in enumerate(self.combinations):
            #c[0] is the first term's index. it is always either the input (index 0)
            #or one of the states (indices 1 to 1 + state_size)
            #c[1] is the second term's index, which can range higher, into the other terms (e.g. x² is needed to compute x³ as x*x²)
            #c[2] is an index tensor which tells us which individuals this value is to be computed for
            #c[3] serves the same purpose, but operates on the sparsified terms tensors, which have different indices for individuals
            #e.g. if two individuals need x², but only one of them also needs x³, 
            #then we need c[3] to index only for the x³ individual in the list of terms for which x² was computed
            first_term = x[:, c[2]] if c[0] == 0 else state[:, c[2], c[0] - 1]
            if c[1] < self.n_variables:
                second_term = x[:, c[2]] if c[1] == 0 else state[:, c[2], c[1] - 1]
            else:
                second_term = terms[c[1] - self.n_variables][:, c[3]]

            product = first_term * second_term
            # Every intermediate product is kept so later monomials can reuse it;
            # torch.compile is expected to prune any that are never referenced.
            terms.append(product)
            if len(self.indices_individuals[i]) == 0:
                continue #this triggers if the product is purely intermediate
            product = product.unsqueeze(2).expand(-1,-1,self.n_polynomials,-1)
            weighted_monomials = product[:,self.indices_individuals[i],self.indices_polynomials[i]] * self.weights[i]
            # weighted_monomials now has shape [batch_size,a,feature_size]
            # and we're going to scatter it into output, which has shape [batch_size,b,feature_size]
            output.index_add_(1,self.scatter_indices[i],weighted_monomials)

        # Add the linear terms (constant terms are already in the initial output).
        # x is [batch_size, population_size, feature_size] and output is
        # [batch_size, population_size, n_polynomials, feature_size]; the same
        # gather/scatter setup as above is used, without the extra factor.
        if len(self.linear_indices_individuals[0]) > 0:
            expanded_x = x.unsqueeze(2).expand(-1,-1,self.n_polynomials,-1)
            product = expanded_x[:,self.linear_indices_individuals[0],self.linear_indices_polynomials[0]] * self.linear_weights[0]
            output.index_add_(1,self.linear_scatter_indices[0],product)

        for i in range(1,self.n_variables):
            if len(self.linear_indices_individuals[i]) == 0:
                continue
            expanded_state = state[:,:,i-1:i].expand(-1,-1,self.n_polynomials,-1)
            product = expanded_state[:, self.linear_indices_individuals[i], self.linear_indices_polynomials[i]] * self.linear_weights[i]
            output.index_add_(1, self.linear_scatter_indices[i], product)

        return output.view(self.batch_size,self.population_size,self.n_polynomials,self.feature_size)
    
    def _compute_third_stage_indices(self):
        """Gathers the sparse index/weight structure for the linear terms.

        Returns one entry per variable (the input followed by each state
        variable), matching the layout consumed by forward().
        """
        x_sparse_weights = self.sparse_coeffs[:,:,1]
        x_ip, x_ii, x_si = self._index_computation_helper(x_sparse_weights)
        linear_indices_individuals = [x_ii]
        linear_indices_polynomials = [x_ip]
        linear_weights = []
        linear_scatter_indices = [torch.LongTensor(x_si).to(self.device)]
        if len(x_ii) > 0:
            linear_weights.append(x_sparse_weights[x_ii,x_ip][None,:,None].detach())
        else:
            linear_weights.append([])

        for i in range(1,self.n_variables):
            current_weights = self.sparse_coeffs[:,:,i+1]
            current_ip, current_ii, current_si = self._index_computation_helper(current_weights)
            if len(current_ii) > 0:
                linear_weights.append(current_weights[current_ii,current_ip][None,:,None].detach())
            else:
                linear_weights.append([])
            linear_indices_individuals.append(current_ii)
            linear_indices_polynomials.append(current_ip)
            linear_scatter_indices.append(torch.LongTensor(current_si).to(self.device))

        return linear_indices_individuals, linear_indices_polynomials, linear_weights, linear_scatter_indices

    def _index_computation_helper(self,current_weights):
        """Finds the (individual, polynomial) positions of nonzero weights.

        Returns parallel lists of polynomial indices, individual indices, and
        flattened scatter targets (``individual * n_polynomials + polynomial``).
        """
        current_ip = []
        true_ii = []
        current_si = []
        for ii in range(current_weights.shape[0]): # over population size
            for ip in range(current_weights.shape[1]): # over n_polynomials
                if current_weights[ii,ip] != 0.0:
                    true_ii.append(ii)
                    current_ip.append(ip)
                    current_si.append(ii * self.n_polynomials + ip)
        return current_ip, true_ii, current_si
    
    def _compute_second_stage_indices(self):
        """Gathers the sparse index/weight structure for degree >= 2 monomials.

        Produces, one entry per monomial in ``self.combinations``:
            weights: tensors of shape ``(1, len(indices_individuals[i]), 1)``
            indices_individuals / indices_polynomials: 1D index tensors
            scatter_indices: flattened targets for ``index_add_`` in forward().
        """
        indices_individuals = []
        indices_polynomials = []
        weights = []
        scatter_indices = []
        for i,comb in enumerate(self.combinations):
            weight_index = self.monomial_map[i] + self.n_variables + 1 #offset from constant and linear terms
            #self.sparse_coeffs is shape [population_size,n_polynomials,n_monomials]
            #and the weight_index is into the last dim
            current_weights = self.sparse_coeffs[:,:,weight_index]
            current_ip, true_ii, current_si = self._index_computation_helper(current_weights)
            current_ii = [comb[2].index(ii) for ii in true_ii]
            if len(true_ii) > 0:
                cw = current_weights[true_ii,current_ip][None,:,None].detach()
                cw.to(self.device)
                weights.append(cw)
            else: #intermediate products, so the lists are empty.
                weights.append([])
            indices_individuals.append(current_ii)
            indices_polynomials.append(current_ip)
            scatter_indices.append(torch.LongTensor(current_si).to(self.device))
        return indices_individuals, indices_polynomials, weights, scatter_indices

    def _compute_first_stage_indices(self):
        """
        Calculates the combinations of indices for the polynomial terms.
        """
        combinations = self._compute_polynomial_programs()
        combinations, monomial_map = self._compute_c2(combinations)
        combinations = self._compute_c3(combinations)
        return combinations, monomial_map
    
    def _compute_c3(self, combinations):
        """Adds the ``c3`` index to each combination.

        Because the intermediate ``terms`` computed in forward() are themselves
        sparse (only computed for the individuals that need them), ``c3`` is a
        reverse lookup from an individual index into the position it occupies in
        the second factor's own (sparsified) term list.
        """
        ret = []
        for i,(c0,c1,c2) in enumerate(combinations):
            if c1 >= self.n_variables:
                # Backtrack a single step to find where this monomial's second
                # factor lives in the (sparsified) terms list.
                c_ret = []
                relevant_index = c1 - self.n_variables
                for ind in c2:
                    c_ret.append(combinations[relevant_index][2].index(ind))
                ret.append(c_ret)
            else:
                ret.append([])
        return [[c0,c1,c2,c3] for (c0,c1,c2),c3 in zip(combinations,ret)]
    
    def _compute_c2(self, combinations):
        """Adds the ``c2`` index to each combination and drops unused monomials.

        For every individual, records which monomials it actually needs, so
        each monomial ends up annotated with the list of individuals that use
        it. Monomials that no individual needs are removed entirely.
        """
        term_index_list = [[] for _ in range(self.n_monomials - self.n_variables - 1)]
        for i in range(self.population_size):
            relevant_terms = self._compute_relevant_terms(self.sparse_coeffs[i],combinations)
            # relevant_terms lists the monomials this individual needs; append
            # this individual's index to each of those monomials.
            for j in relevant_terms:
                term_index_list[j].append(i)
        # term_index_list only covers monomials of degree >= 2, so it has the
        # same length as combinations.
        ret = []
        to_remove = []

        for i,il in enumerate(term_index_list):
            if len(il) == 0:
                to_remove.append(i)
            else:
                ret.append(il)
        combinations = self._remove_monomials(combinations,to_remove)
        monomial_map = {a : b for a,b in enumerate([c for c in list(range(self.n_monomials)) if c not in to_remove])}

        return [[c0,c1,c2] for (c0,c1),c2 in zip(combinations,ret)], monomial_map
    
    def _remove_monomials(self,combinations,to_remove):
        for i in to_remove:
            full_index = i + self.n_variables
            #full index = how would the combination number i appear in c0 and c1? -> offset by self.n_variables
            for j in range(i + 1,len(combinations)):
                for k in [0,1]:
                    if combinations[j][k] > full_index: #if we find that the index that we access is greater than the one we will remove
                        combinations[j][k] -= 1 #we reduce the relevant index, since the comb will move 1 step forward due to this deletion
        ret_comb = [x for i,x in enumerate(combinations) if not (i in to_remove)]
        return ret_comb
    
    def _compute_relevant_terms(self,sparse_coeffs,combinations):
        #sparse_coeffs is of shape [n_polynomials, n_monomials]
        #combinations is a list of length n_monomials - n_variables - 1, of 2-int-tuples
        #we return here a list of integers, which tells us which indices in combinations are 
        #to be computed for this individual
        nonzero_coeffs = (sparse_coeffs != 0).any(dim=0).nonzero(as_tuple=True)[0].cpu().tolist()
        # For every nonzero coefficient, backtrack through the multiplication
        # program to collect all intermediate monomials it depends on.
        ret = []
        for coeff_index in nonzero_coeffs:
            # Constant and linear terms have no intermediate dependencies.
            if coeff_index <= self.n_variables:
                continue
            search_index = coeff_index - self.n_variables - 1
            while search_index >= 0: # index 0 is also relevant: it is input * input
                ret.append(search_index)
                # Note: only the second factor is followed here, so products of
                # two composite monomials (e.g. (x*x)*(x*x)) are not expanded.
                _, search_index = combinations[search_index]
                search_index = search_index - self.n_variables
        ret = sorted(list(set(ret)))
        return ret
    
    def _compute_polynomial_programs(self):
        """Builds the pairwise-multiplication program for degree >= 2 monomials.

        Each monomial of total degree d is expressed as ``variable * (monomial
        of degree d-1)``, so higher-degree terms reuse the lower-degree ones
        already computed. The program is a list of 2-tuples of indices into the
        variables/previously-computed monomials.

        Note: the current factorisation always peels off a single variable
        (``x**4`` is built as ``x * (x * (x * x))``); a balanced factorisation
        such as ``(x*x) * (x*x)`` would use fewer multiplications but is not
        implemented.
        """
        pre_combinations = list(itertools.chain.from_iterable(
            itertools.combinations_with_replacement(range(self.n_variables), d)
            for d in range(self.degree + 1)
        ))
        if not len(pre_combinations) == self.n_monomials:
            expected_shape = (self.population_size,self.n_polynomials,len(pre_combinations))
            gotten_shape = self.sparse_coeffs.shape
            raise ValueError(f"expected coefficients to have shape {expected_shape} but got {gotten_shape} instead")
        combinations = []
        monomial_dict = {}
        for i in range(self.n_variables):
            monomial_dict[(i,)] = i

        for comb in pre_combinations:
            if len(comb) < 2: # skip the constant and linear terms
                continue
            combinations.append((comb[0],monomial_dict[tuple(comb[1:])]))
            monomial_dict[tuple(comb)] = (len(combinations) - 1) + self.n_variables
        combinations = [[c0,c1] for c0,c1 in combinations]
        return combinations
    