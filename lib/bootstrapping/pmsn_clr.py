"""Probes used to characterise a freshly sampled PMSN-CLR neuron.

These helpers simulate a candidate neuron in isolation to decide whether it is
usable: finding its resting state, measuring its spike rates under zero and
random input, and analysing the dependency graph of its state polynomials.
"""

import torch
from esn.polynomial_compiler import create_polynomials

def compute_resting_state(neuron, max_steps, rs_delta_eps, max_state_abs):
    """Finds a neuron's resting state by simulating it until the state settles.

    Args:
        neuron: The neuron to simulate.
        max_steps: Maximum number of steps to simulate.
        rs_delta_eps: State-change threshold below which the neuron is
            considered to be at rest.
        max_state_abs: Absolute state threshold above which the neuron is
            considered dead (exploding).

    Returns:
        Tuple ``(state, dead)``: the resting state tensor and a bool that is
        True if the neuron exploded or failed to settle.
    """
    dead = True
    with torch.no_grad():
        # Start slightly above the resting level, and push the first state
        # variable above threshold to induce a spike; this activates neurons
        # that would otherwise stay silent from a zero start.
        state = torch.ones(neuron.params.n_states) * 0.1
        x = torch.tensor(0.0)
        state[0] = neuron.params.threshold + 0.1
        for i in range(max_steps):
            _, new_state = neuron(x, state)
            if torch.max(torch.abs(state)) > max_state_abs:
                dead = True
                break
            if torch.max(torch.abs(new_state - state)) < rs_delta_eps:
                dead = False
                break
            state = new_state
        return state, dead #return the resting state and whether the neuron is dead

def input_trials(neuron,n_rand_trials,n_steps,max_state_abs):
    """Measures a neuron's spike rate under zero and random input.

    A constant-zero input (which ideally produces no spikes) is run alongside
    several standard-normal input trials whose average spike rate is reported.

    Args:
        neuron: The neuron to simulate.
        n_rand_trials: Number of random-input trials to run.
        n_steps: Number of steps to simulate.
        max_state_abs: Absolute state threshold above which the neuron is
            considered dead (exploding).

    Returns:
        Tuple ``(spike_rate_zero, spike_rate_rand, exploded)``.
    """
    with torch.no_grad():
        const_input = torch.zeros((1,1))
        state = neuron.init_state(1,1+n_rand_trials)
        spike_rate_0 = 0
        spike_rate_rand = 0
        for i in range(n_steps):
            rand_input = torch.randn((1,n_rand_trials)) + 1.0 # mean 1 and stddev 1
            combined_input = torch.cat((const_input,rand_input), dim=1)
            spikes, state = neuron(combined_input, state)
            spike_rate_0 += spikes[0,0]
            spike_rate_rand += torch.sum(spikes[0,1:])/n_rand_trials
            if torch.max(torch.abs(state)) > max_state_abs:
                return 0.0, 1.0, True
        return spike_rate_0/n_steps, spike_rate_rand/n_steps, False

def find_influencers_of_state1(neuron):
    """Returns the set of variable indices that (in)directly influence x1.

    Walks the state-polynomial dependency graph upwards from the first state
    variable, returning every input/state index ``j`` that x1 transitively
    depends on.
    """
    state_polynomials = create_polynomials(neuron.params.polynomial_coeffs, neuron.params.degree)
    reset_polynomials = create_polynomials(neuron.params.reset_coeffs, 1)

    n_states = len(state_polynomials)
    direct_deps = [set() for _ in range(n_states)]
    all_vars = [f"x{i}" for i in range(n_states + 1)]
    
    # Build direct dependencies
    for i in range(n_states):
        poly = state_polynomials[i] + "+" + reset_polynomials[i]
        for j, var in enumerate(all_vars):
            if var in poly:
                direct_deps[i].add(j)
    
    # We want to do a BFS/DFS *up* the graph from x1
    # x1 is "i=0" in direct_deps, so let's call that node_i = 0
    node_i = 0  # i=0 => x1
    
    to_visit = list(direct_deps[node_i])  # direct parents of x1
    visited = set()
    
    while to_visit:
        curr = to_visit.pop()
        if curr not in visited:
            visited.add(curr)
            # If curr > 0, that means it's another state variable x_curr
            # so we can add *its* parents
            if curr > 0:
                # x_{curr} is stored in direct_deps[curr-1]
                to_visit.extend(direct_deps[curr - 1])
            # If curr == 0 => x0 => no further parents since x0 is the input
            # and not updated by anything else.
    
    # visited now holds all j where j is a direct or indirect parent of x1
    return visited  # set of indices j

def check_influence_of_input(neuron):
    """
    Return a list of booleans indicating for each state variable x_{i+1},
    does x0 (the input) influence it?
    """
    state_polynomials = create_polynomials(neuron.params.polynomial_coeffs, neuron.params.degree)
    reset_polynomials = create_polynomials(neuron.params.reset_coeffs, 1)

    n_states = len(state_polynomials)
    direct_deps = [set() for _ in range(n_states)]
    all_vars = [f"x{i}" for i in range(n_states + 1)]
    
    # Build direct dependencies
    for i in range(n_states):
        poly = state_polynomials[i] + "+" + reset_polynomials[i]
        for j, var in enumerate(all_vars):
            if var in poly:
                direct_deps[i].add(j)
    
    # Build forward adjacency: from each variable j in [0..n_states], 
    # which states does it feed into? i.e. find all i s.t. j in direct_deps[i].
    forward_adj = [[] for _ in range(n_states + 1)]  
    # forward_adj[j] will be a list of state indices i (0-based for x_{i+1})
    # for which there's an edge j->(i+1).
    
    for i in range(n_states):
        for j in direct_deps[i]:
            # j -> (i+1)
            forward_adj[j].append(i)  # store the state index i (meaning x_{i+1})
    
    # BFS from x0 (which is index 0) in forward_adj
    influenced_states = set()
    queue = forward_adj[0][:]  # the states x_{i+1} directly fed by x0
    while queue:
        s = queue.pop()
        if s not in influenced_states:
            influenced_states.add(s)
            # from x_{s+1}, see which states it feeds
            # "Which states does x_{s+1} feed?"
            # That would be forward_adj[s+1] because s+1 is the index for x_{s+1} 
            queue.extend(forward_adj[s+1])
    
    # Now influenced_states is the set of state indices i 
    # for which there's a path x0 -> x_{i+1}.
    # We want a boolean list of length n_states: does x0 influence x_{i+1}?
    return [(i in influenced_states) for i in range(n_states)]
