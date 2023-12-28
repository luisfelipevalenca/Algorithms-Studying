#!/usr/bin/env python
# coding: utf-8

# # Problem 1
# 
# We saw how to solve TSPs in this module and in particular presented two approaches to encode a TSP as an integer linear program. In this problem, we will ask you to adapt the TSP solution to the related problem of $k$ Travelling Salespeople Problem ($k$-TSP).
# 
# Let $G$ be a complete graph with $n$ vertices that we will label $0, \ldots, n-1$ (keeping Python array indexing in mind). Our costs are specified using a matrix $C$ wherein $C_{i,j}$ is the cost of the edge from vertex $i$ to $j$ for $i \not= j$.
# 
# In this problem, we have $k \geq 1$ salespeople who must  start from vertex $0$ of the graph (presumably the location of the sales office) and together visit every location in the graph,  each returning back to vertex $0$. Each location must be visited exactly once by some salesperson in the team. Therefore, other than vertex $0$ (the start/end vertex of every salesperson's tour), no two salesperson tours have a vertex in common. Notice that for $k=1$, this is just the regular TSP problem we have studied. 
# 
# Also, all $k$ salespeople must be employed in the tour. In other words, if we have $k=3$ then each salesperson must start at $0$ and visit a sequence of one or more vertices and come back to $0$. No salesperson can be "idle".
# 
# ## Example-1
# 
# Consider a graph with $5$ nodes and the following cost matrix:
# 
# $$ \begin{array}{c|ccccc}
#   \text{Vertices} & 0 & 1 & 2 & 3 & 4 \\ 
#    \hline
#  0 & - & 3 & 4 & 3 & 5 \\ 
#  1 & 1 & - & 2 & 4 & 1 \\ 
#  2 & 2 & 1 & - & 5 & 4 \\ 
#  3 & 1 & 1 & 5 & - & 4 \\ 
#  4 & 2 & 1 & 3 & 5 & - \\ 
#  \end{array}$$
#  
#  For instance $C_{2,3}$ the cost of edge from vertex $2$ to $3$ is $5$. The $-$ in the diagonal entries simply tells us that we do not care what goes in there since we do not have self-loops in the graph.
#  
# The optimal $2$-TSP tour for $k=2$ salespeople is shown below.
#   - Salesperson # 1: $0 \rightarrow 2 \rightarrow 1 \rightarrow 4 \rightarrow 0$.
#   - Salesperson # 2: $0 \rightarrow 3 \rightarrow 0$.
#   
# The total cost of the edges traversed by the two salespeople equals $12$.
# 
# For $k=3$, the optimal $3-$ TSP tour is as shown below.
#   - Salesperson # 1: $0 \rightarrow 1 \rightarrow 4$, 
#   - Salesperson # 2: $0 \rightarrow 2$, 
#   - Salesperson # 3: $0 \rightarrow 3$.
# 
# The total cost is $16$.
# 
# The objective of this problem is to formulate an ILP using the MTZ approach.

# ### Problem 1A (MTZ approach)
# 
# We will use the same ILP setup as in our notes (see the notes on Exact Approaches to TSP that includes the ILP encodings we will use in this problem).
#   - Decision variables $x_{i,j}$ for $i \not= j$ denoting that the tour traverses the edge from $i$ to $j$.
#   - Time stamps $t_1, \ldots, t_{n-1}$. The start/end vertex $0$ does not get a time stamp.
#   
# Modify the MTZ approach to incorporate the fact that $k$ salespeople are going to traverse the graph.
# 
# #### (A) Degree Constraints
# 
# What do the new degree constraints look like? Think about how many edges in the tour will need to enter/leave each vertex? Note that you may have to treat vertex $0$ differently from the other vertices of the graph.
# 
# Your answer below is not graded. However you are encouraged to write it down and check with the answers to select problems provided at the end.

# YOUR ANSWER HERE

# ### (B) Time Stamp Constraints 
# 
# Formulate the time stamp constraints for the $k$-TSP problem. Think about how you would need to change them to eliminate subtour.
# 
# Your answer below is not graded. However you are encouraged to write it down and check with the answers to select problems provided at the end.
# 

# YOUR ANSWER HERE

# ### (C) Implement
# 
# Complete the implementation of the function `k_tsp_mtz_encoding(n, k, cost_matrix)` below. It follows the same input convention as the code supplied in the notes. The input `n` denotes the size of the graph with vertices labeled `0`,.., `n-1`, `k` is the number of salespeople, and `cost_matrix` is a list of lists wherein `cost_matrix[i][j]` is the edge cost to go from `i` to `j` for `i != j`. Your code must avoid accessing `cost_matrix[i][i]` to avoid bugs. These entries will be supplied as `None` in the test cases.
# 
# Your code must return a list `lst` that has exactly $k$ lists in it, wherein `lst[j]` represents the locations visited by the $j^{th}$ salesperson. 
# 
# For the example above, for $k=2$, your code must return 
# ~~~
# [ [0, 2, 1, 4], [0, 3] ]
# ~~~
# For the example above, for $k=3$, your code must return
# ~~~
# [ [0, 1, 4], [0, 2], [0, 3] ]
# ~~~

# In[177]:


from pulp import *

def k_tsp_mtz_encoding(n, k, cost_matrix):
    assert len(cost_matrix) == n, "Cost matrix is not nxn."
    assert all(len(row) == n for row in cost_matrix), "Cost matrix rows are not of length n."

    # Create the LP problem
    prob = LpProblem('k-TSP', LpMinimize)

    # Decision variables for paths (x) and active tours (active)
    x = [[[LpVariable(f"x_{i}_{j}_{l}", cat="Binary") if i != j else None for j in range(n)] for i in range(n)] for l in range(k)]
    active = [LpVariable(f"active_{l}", cat="Binary") for l in range(k)]

    # MTZ constraints to avoid subtours
    u = [[LpVariable(f"u_{i}_{l}", lowBound=0, upBound=n, cat="Continuous") for i in range(n)] for l in range(k)]

    # Objective function: Minimize the total travel cost
    prob += lpSum(cost_matrix[i][j] * x[l][i][j] for l in range(k) for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None)

    # Constraints
    # Ensure exactly k tours are active
    prob += lpSum(active[l] for l in range(k)) == k

    for l in range(k):
        # Degree constraints
        # Vertex 0
        prob += lpSum(x[l][0][j] for j in range(1, n) if x[l][0][j] is not None) == active[l]
        prob += lpSum(x[l][j][0] for j in range(1, n) if x[l][j][0] is not None) == active[l]
        # Other vertices
        for i in range(1, n):
            prob += lpSum(x[l][i][j] for j in range(n) if j != i and x[l][i][j] is not None) == lpSum(x[l][j][i] for j in range(n) if j != i and x[l][j][i] is not None)

        # Subtour elimination constraints for each salesperson
        for i in range(1, n):
            for j in range(1, n):
                if i != j and x[l][i][j] is not None:
                    prob += u[l][i] - u[l][j] + n * x[l][i][j] <= n - 1

    # Constraint to ensure each vertex is visited by exactly one salesperson
    for i in range(1, n):
        prob += lpSum(x[l][j][i] for l in range(k) for j in range(n) if j != i and x[l][j][i] is not None) == 1

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False))

    # Extract the tours
    all_tours = []
    for l in range(k):
        tour = [0]
        current_city = 0
        while True:
            next_city = None
            for j in range(n):
                if current_city != j and x[l][current_city][j] is not None and value(x[l][current_city][j]) == 1:
                    next_city = j
                    break
            if next_city is None or next_city == 0:
                break
            tour.append(next_city)
            current_city = next_city
        if len(tour) > 1:
            all_tours.append(tour)

    return all_tours


# In[178]:


cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=2
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 12) <= 0.001, f'Expected tour cost is 12, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')


# In[179]:


cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=3
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 16) <= 0.001, f'Expected tour cost is 16, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 2 points')


# In[180]:


cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 2

all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 4) <= 0.001, f'Expected tour cost is 4, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')


# In[181]:


cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 4

all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 2 points')


# In[182]:


from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(5):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]

    print(f'Tour cost obtained by your code: {tour_cost}')
    #assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
    for i in range(1, n):
        is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
        assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'
    print('------')
print('test passed: 15 points')


# ## Problem 1 B
# 
# Notice that in previous part, it happens that with $k=4$ salespeople, we actually get a worse cost than using $k=3$ people. You can try out a few examples to convince yourself as to why this happens. 
# 
# We wish to modify the problem to allow salespeople to idle. In other words, although we input $k$ salespeople, the tour we construct may involve $1 \leq l \leq k$ salespeople. 
# 
# Modify the ILP formulation from the previous problem to solve the problem of up to $k$ people rather than exactly $k$ salespeople. Note that we still require that every vertex be visited exactly once by some salesperson. 
# 
# Complete the implementation of the function `upto_k_tsp_mtz_encoding(n, k, cost_matrix)` below. It follows the same input convention as previous problem but note that we are now computing a tour with at most $k$ salespeople. In other words, not all salespeople need be employed in the tour.
# 
# Your code must return a list `lst` that has less than or equal to  $k$ lists, wherein `lst[j]` represents the locations visited by the $j^{th}$ salesperson. 
# 
# For Example-1 from the previous part above, for $k=2$ or $k=3$, your code must return 
# ~~~
# [ [0, 3, 1, 4, 2] ]
# ~~~
# As it turns out, in this example a single salesperson suffices to yield optimal cost.

# In[183]:


from pulp import *

def upto_k_tsp_mtz_encoding_with_constraints(n, k, cost_matrix, constraints):
    assert 1 <= k < n
    assert len(cost_matrix) == n, "Cost matrix is not nxn."
    assert all(len(row) == n for row in cost_matrix), "Cost matrix rows are not of length n."
    assert all(0 <= il < n and 0 <= jl < n for il, jl in constraints), "Invalid constraints."

    # Initialize the problem
    prob = LpProblem("upto_kTSP_with_Constraints", LpMinimize)

    # Decision variables
    x = [[[LpVariable(f"x_{i}_{j}_{l}", cat="Binary") if i != j else None for j in range(n)] for i in range(n)] for l in range(k)]
    active = [LpVariable(f"active_{l}", cat="Binary") for l in range(k)]
    u = [[LpVariable(f"u_{i}_{l}", lowBound=1, cat="Continuous") for i in range(n)] for l in range(k)]

    # Objective function
    prob += lpSum(cost_matrix[i][j] * x[l][i][j] for l in range(k) for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None)

    # Constraints
    for l in range(k):
        for i in range(n):
            prob += lpSum(x[l][j][i] for j in range(n) if j != i and x[l][j][i] is not None) == active[l]
            prob += lpSum(x[l][i][j] for j in range(n) if j != i and x[l][i][j] is not None) == active[l]

        # Subtour elimination (MTZ constraints)
        for i in range(1, n):
            for j in range(1, n):
                if i != j and x[l][i][j] is not None:
                    prob += u[l][i] - u[l][j] + n * x[l][i][j] <= n - 1

    # Each vertex must be visited exactly once by any salesperson
    for i in range(1, n):
        prob += lpSum(x[l][j][i] for l in range(k) for j in range(n) if j != i and x[l][j][i] is not None) == 1

    # Additional constraints for specified order
    for il, jl in constraints:
        for l in range(k):
            prob += u[l][il] + 1 <= u[l][jl]

    # Solve the problem
    prob.solve()

    # Extract tours
    all_tours = []
    for l in range(k):
        if value(active[l]) == 0:
            continue
        tour = [0]
        current_city = 0
        while True:
            next_city = None
            for j in range(n):
                if current_city != j and x[l][current_city][j] is not None and value(x[l][current_city][j]) == 1:
                    next_city = j
                    break
            if next_city is None or next_city == 0:
                break
            tour.append(next_city)
            current_city = next_city
        all_tours.append(tour)

    return all_tours

# Example usage
n = 5
k = 2
cost_matrix = [[None, 3, 4, 3, 5], [1, None, 2, 4, 1], [2, 1, None, 5, 4], [1, 1, 5, None, 4], [2, 1, 3, 5, None]]
constraints = [(3, 4), (1, 2)]
all_tours = upto_k_tsp_mtz_encoding_with_constraints(n, k, cost_matrix, constraints)
print(f'Tours: {all_tours}')


# In[184]:


cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=3
all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) <= k, f'<= {k} tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

assert len(all_tours) == 1, f'In this example, just one salesperson is needed to optimally visit all vertices. Your code returns {len(all_tours)}'
print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 10) <= 0.001, f'Expected tour cost is 10, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')


# In[185]:


cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 5

all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) <= k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 4) <= 0.001, f'Expected tour cost is 4, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')


# In[186]:


from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) <= k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]

    print(f'Tour cost obtained by your code: {tour_cost}')
    #assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
    for i in range(1, n):
        is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
        assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'
    print('------')
print('test passed: 4 points')


# ## Problem 2 (10 points)
# 
# We noted the use of Christofides algorithm for metric TSP. We noted that for non-metric TSPs it does not work. 
# In fact, the shortcutting used in Christofides algorithm can be _arbitrarily_ bad for a TSP that is symmetric but fails to be a metric TSP.
# 
# In this example, we would like you to frame a symmetric TSP instance ($C_{ij} = C_{ji}$) with $5$ vertices wherein the algorithm obtained by "shortcutting" the minimum spanning tree (MST), that would be a 2-factor approximation for metric TSP, yields an answer that can be quite "far off" from the optimal solution.
# 
# Enter a __symmetric__ cost-matrix for the TSP below as a 5x5 matrix as a list of lists following convention in our notes. such that the optimal answer is at least $10^6$ times smaller than that obtained by the TSP-based approximation. We will test your answer by running the TSP with shortcutting algorithm.
# 
# __Hint:__ Force the edges $(0,1), (1,2), (2,3)$ and $(3,4)$ to be the minimum spanning tree. But make the weight of the edge form $4$ back to $0$ very high.
# 
# 
# __Note:__ this problem is tricky and requires you to be very familiar with how Christofides algorithm works. It may be wise to attempt the remaining problems first before this one. Do not worry about the diagonal entry of your matrices.
# 

# In[187]:


import networkx as nx
from pulp import *

# Defining the symmetric cost matrix
cost_matrix = [
    [0, 1, 10, 10, 10**9],  # Increased weight from 4 to 0
    [1, 0, 1, 10, 10],
    [10, 1, 0, 1, 10],
    [10, 10, 1, 0, 1],
    [10**9, 10, 10, 1, 0]   # Symmetric update
]

# MST-based TSP approximation
def minimum_spanning_tree_tsp(cost_matrix):
    n = len(cost_matrix)
    G = nx.Graph()
    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=cost_matrix[i][j])
    T = nx.minimum_spanning_tree(G)

    # Traverse the MST to form a tour
    def traverse_mst(tour_so_far, cur_node, mst_dict):
        next_nodes = mst_dict[cur_node]
        for j in next_nodes:
            if j not in tour_so_far:
                tour_so_far.append(j)
                traverse_mst(tour_so_far, j, mst_dict)
    
    mst_dict = {i: [] for i in range(n)}
    for (i, j) in T.edges:
        mst_dict[i].append(j)
        mst_dict[j].append(i)

    tour = [0]
    traverse_mst(tour, 0, mst_dict)
    
    tour_cost = sum(cost_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + cost_matrix[tour[-1]][0]
    return tour, tour_cost

# MTZ encoding for optimal TSP tour
def mtz_encoding_tsp(cost_matrix):
    n = len(cost_matrix)
    prob = LpProblem('TSP-MTZ', LpMinimize)
    x = [[LpVariable(f'x_{i}_{j}', cat='Binary') if i != j else None for j in range(n)] for i in range(n)]
    u = [LpVariable(f'u_{i}', lowBound=1, upBound=n, cat='Continuous') for i in range(n)]

    # Objective function
    prob += lpSum(cost_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

    # Constraints
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(n) if j != i) == 1  # Exactly one edge out
        prob += lpSum(x[j][i] for j in range(n) if j != i) == 1  # Exactly one edge in

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    prob.solve()
    tour = []
    for v in prob.variables():
        if v.varValue == 1 and v.name.startswith('x_'):
            i, j = map(int, v.name.split('_')[1:])
            tour.append((i, j))

    # Constructing the tour path
    next_node = 0
    opt_tour = [0]
    while len(opt_tour) < n:
        for i, j in tour:
            if i == next_node:
                next_node = j
                opt_tour.append(j)
                break

    opt_tour_cost = value(prob.objective)
    return opt_tour, opt_tour_cost

# Run the MST-based TSP approximation
mst_tour, mst_tour_cost = minimum_spanning_tree_tsp(cost_matrix)
print(f'MST-based TSP Approximation: Tour = {mst_tour}, Cost = {mst_tour_cost}')

# Run the MTZ encoding for optimal TSP
opt_tour, opt_tour_cost = mtz_encoding_tsp(cost_matrix)
print(f'Optimal TSP: Tour = {opt_tour}, Cost = {opt_tour_cost}')

# Check if the MST-based approximation is at least 10^6 times costlier
assert mst_tour_cost / opt_tour_cost >= 1e6, f'The MST-based approximation is not at least 10^6 times costlier. Ratio = {mst_tour_cost / opt_tour_cost}'
print('The MST-based approximation is at least 10^6 times costlier than the optimal solution.')


# In[188]:


# check that the cost matrix is symmetric.
assert len(cost_matrix) == 5, f'Cost matrix must have 5 rows. Yours has {len(cost_matrix)} rows'
assert all(len(cj) == 5 for cj in cost_matrix), f'Each row of the cost matrix must have 5 entries.'
for i in range(5):
    for j in range(i):
        assert cost_matrix[i][j] == cost_matrix[j][i], f'Cost matrix fails to be symmetric at entries {(i,j)} and {(j,i)}'
print('Structure of your cost matrix looks OK (3 points).')


# Please ensure that you run the two cells below or else, your tests will fail.

# In[189]:


# MST based tsp approximation
import networkx as nx

# This code implements the simple MST based shortcutting approach that would yield factor of 2
# approximation for metric TSPs.
def minimum_spanning_tree_tsp(n, cost_matrix):
    G = nx.Graph()
    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=cost_matrix[i][j])
    T = nx.minimum_spanning_tree(G)
    print(f'MST for your graph has the edges {T.edges}')
    mst_cost = 0
    mst_dict = {} # store mst as a dictionary
    for (i,j) in T.edges:
        mst_cost += cost_matrix[i][j]
        if i in mst_dict:
            mst_dict[i].append(j)
        else:
            mst_dict[i] = [j]
        if j in mst_dict:
            mst_dict[j].append(i)
        else:
            mst_dict[j] = [i]
    print(f'MST cost: {mst_cost}')
    print(mst_dict)
    # Let's form a tour with short cutting
    def traverse_mst(tour_so_far, cur_node):
        assert cur_node in mst_dict
        next_nodes = mst_dict[cur_node]
        for j in next_nodes:
            if j in tour_so_far:
                continue
            tour_so_far.append(j)
            traverse_mst(tour_so_far, j)
        return
    tour = [0]
    traverse_mst(tour, 0)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    return tour, tour_cost


# In[190]:


# optimal TSP tour taken from our notes using MTZ encoding
from pulp import *

def mtz_encoding_tsp(n, cost_matrix):
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'
    # create our encoding variables
    binary_vars = [ # add a binary variable x_{ij} if i not = j else simply add None
        [ LpVariable(f'x_{i}_{j}', cat='Binary') if i != j else None for j in range(n)] 
        for i in range(n) ]
    # add time stamps for ranges 1 .. n (skip vertex 0 for timestamps)
    time_stamps = [LpVariable(f't_{j}', lowBound=0, upBound=n, cat='Continuous') for j in range(1, n)]
    # create the problem
    prob = LpProblem('TSP-MTZ', LpMinimize)
    # create add the objective function 
    objective_function = lpSum( [ lpSum([xij*cj if xij != None else 0 for (xij, cj) in zip(brow, crow) ])
                           for (brow, crow) in zip(binary_vars, cost_matrix)] )
    
    prob += objective_function 
    
    # add the degree constraints
    for i in range(n):
        # Exactly one leaving variable
        prob += lpSum([xj for xj in binary_vars[i] if xj != None]) == 1
        # Exactly one entering
        prob += lpSum([binary_vars[j][i] for j in range(n) if j != i]) == 1
    # add time stamp constraints
    for i in range(1,n):
        for j in range(1, n):
            if i == j: 
                continue
            xij = binary_vars[i][j]
            ti = time_stamps[i-1]
            tj = time_stamps[j -1]
            prob += tj >= ti + xij - (1-xij)*(n+1) # add the constraint
    # Done: solve the problem
    status = prob.solve(PULP_CBC_CMD(msg=False)) # turn off messages
    assert status == constants.LpStatusOptimal, f'Unexpected non-optimal status {status}'
    # Extract the tour
    tour = [0]
    tour_cost = 0
    while len(tour) < n:
        i = tour[-1]
        # find all indices j such that x_ij >= 0.999 
        sols = [j for (j, xij) in enumerate(binary_vars[i]) if xij != None and xij.varValue >= 0.999]
        assert len(sols) == 1, f'{sols}' # there better be just one such vertex or something has gone quite wrong
        j = sols[0] # extract the lone solutio 
        tour_cost = tour_cost + cost_matrix[i][j] # add to the tour cost
        tour.append(j) # append to the tour
        assert j != 0
    i = tour[-1]
    tour_cost = tour_cost + cost_matrix[i][0]
    return tour, tour_cost
        


# In[191]:


#test that exact answer is 10^6 times smaller than approximate answer.
# compute MST based approximation
tour, tour_cost = minimum_spanning_tree_tsp(5, cost_matrix)
print(f'MST approximation yields tour is {tour} with cost {tour_cost}')
# compute exact answer
opt_tour, opt_tour_cost = mtz_encoding_tsp(5, cost_matrix)
print(f'Optimal tour is {opt_tour} with cost {opt_tour_cost}')
# check that the fraction is 1million times apart.
assert tour_cost/opt_tour_cost >= 1E+06, 'The TSP + shortcutting tour must be at least 10^6 times costlier than optimum. In your case, the ratio is {tour_cost/opt_tour_cost}'
print('Test passed: 7 points')


# ## Problem 3
# 
# In this problem, we wish to solve TSP with additional constraints. Suppose we are given a TSP instance in the form of a $n\times n$ matrix $C$ representing a complete graph. 
# 
# We wish to solve a TSP but with additional constraints specified as a list $[(i_0, j_0), \ldots, (i_k, j_k)]$ wherein each pair $(i_l, j_l)$ in the list specifies that vertex $i_l$ must be visited in the tour before vertex $j_l$. Assume that the tour starts/ends at vertex $0$ and none of the vertices in the constraint list is $0$. I.e, $i_l\not= 0, j_l \not= 0$ for all $0 \leq l \leq k$.
# 
# Modify one of the ILP encodings we have presented to solve TSP with extra constraints. Implement your solution in the function `tsp_with_extra_constraints(n, cost_matrix, constr_list)` where the extra argument `constr_list` is a list of pairs `[(i0,j0),...., (ik, jk)]` that specify for each pair `(il,jl)` that vertex `il` must be visited before `jl`. Assume that the problem is feasible (no need to handle infeasible instances). 
# Your code should output the optimal tour as a list.
# 
# ## Example
# 
# Consider again the graph with $5$ nodes and the following cost matrix from problem 1:
# 
# $$ \begin{array}{c|ccccc}
#   \text{Vertices} & 0 & 1 & 2 & 3 & 4 \\ 
#    \hline
#  0 & - & 3 & 4 & 3 & 5 \\ 
#  1 & 1 & - & 2 & 4 & 1 \\ 
#  2 & 2 & 1 & - & 5 & 4 \\ 
#  3 & 1 & 1 & 5 & - & 4 \\ 
#  4 & 2 & 1 & 3 & 5 & - \\ 
#  \end{array}$$
#  
# The optimal TSP tour will be $[0, 3, 1, 4, 2]$ with total cost $10$.
# 
# Suppose we added the constraints $[(4, 3), (1, 2)]$ we note that the tour satisfies the constraint $(1, 2)$ since it visits vertex $1$ before vertex $2$ but it unfortunately, $(4,3)$ is violated since vetex $3$ is visited before $4$ in the tour.
# 

# In[204]:


from pulp import *
from random import uniform, randint

def tsp_with_extra_constraints(n, cost_matrix, constr_list):
    # Initialize the problem
    prob = LpProblem("TSP_with_Constraints", LpMinimize)

    # Decision variables: x[i][j] is 1 if the path from i to j is taken, 0 otherwise
    x = [[LpVariable(f"x_{i}_{j}", cat='Binary') if i != j else None for j in range(n)] for i in range(n)]

    # Order variables to prevent subtours: u[i] is the order of visiting node i
    u = [LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat='Continuous') for i in range(n)]

    # Objective function: Minimize the total travel cost
    prob += lpSum(cost_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None)

    # Constraints
    # Each node must be entered and left exactly once
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(n) if i != j and x[i][j] is not None) == 1
        prob += lpSum(x[j][i] for j in range(n) if i != j and x[j][i] is not None) == 1

    # Subtour elimination (MTZ constraints)
    for i in range(1, n):
        for j in range(1, n):
            if i != j and x[i][j] is not None:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    # Additional constraints for specified order
    for il, jl in constr_list:
        prob += u[il] + 1 <= u[jl]

    # Solve the problem
    prob.solve()

    # Extract tour
   
    tour = [0]  # Start from vertex 0
    current_city = 0
    visited = set([0])  # Set to track visited vertices

    while len(visited) < n:
        for j in range(n):
            if current_city != j and x[current_city][j].varValue > 0.99 and j not in visited:
                tour.append(j)
                current_city = j
                visited.add(j)
                break

    return tour


def calculate_tour_cost(tour, cost_matrix):
    tour_cost = 0
    for i in range(len(tour) - 1):
        if cost_matrix[tour[i]][tour[i + 1]] is not None:
            tour_cost += cost_matrix[tour[i]][tour[i + 1]]
    # Handle the return to the start, avoiding None
    if cost_matrix[tour[-1]][tour[0]] is not None:
        tour_cost += cost_matrix[tour[-1]][tour[0]]
    return tour_cost

def create_cost(n):
    return [[uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(6, 11)
    cost_matrix = create_cost(n)
    constraints = [(1, 3), (4, 2), (n-1, 1), (n-2, 2)]
    tour = tsp_with_extra_constraints(n, cost_matrix, constraints)

    if tour is not None:
        tour_cost = calculate_tour_cost(tour, cost_matrix)
        print(f'Tour: {tour}')
        print(f'Cost of your tour: {tour_cost}')
        for i in range(n):
            num = sum([1 if j == i else 0 for j in tour])
            assert num == 1, f'Vertex {i} repeats {num} times in tour'
        for (i, j) in constraints:
            assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i, j)}'
    else:
        print("No feasible tour found.")



# In[205]:


cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
constraints = [(3,4),(1,2)]
tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
i = 0
tour_cost = 0
for j in tour[1:]:
    tour_cost += cost_matrix[i][j]
    i = j
tour_cost += cost_matrix[i][0]
print(f'Tour:{tour}')
print(f'Cost of your tour: {tour_cost}')
assert abs(tour_cost-10) <= 0.001, 'Expected cost was 10'
for i in range(n):
    num = sum([1 if j == i else 0 for j in tour])
    assert  num == 1, f'Vertex {i} repeats {num} times in tour'
for (i, j) in constraints:
    assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (3 points)')


# In[206]:


cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
constraints = [(4,3),(1,2)]
tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
i = 0
tour_cost = 0
for j in tour[1:]:
    tour_cost += cost_matrix[i][j]
    i = j
tour_cost += cost_matrix[i][0]
print(f'Tour:{tour}')
print(f'Cost of your tour: {tour_cost}')
assert abs(tour_cost-13) <= 0.001, 'Expected cost was 13'
for i in range(n):
    num = sum([1 if j == i else 0 for j in tour])
    assert  num == 1, f'Vertex {i} repeats {num} times in tour'
for (i, j) in constraints:
    assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (3 points)')


# In[207]:


from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(6, 11)
    cost_matrix = create_cost(n)
    constraints = [(1, 3), (4, 2), (n-1, 1), (n-2, 2)]
    tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    print(f'Tour:{tour}')
    print(f'Cost of your tour: {tour_cost}')
    for i in range(n):
        num = sum([1 if j == i else 0 for j in tour])
        assert  num == 1, f'Vertex {i} repeats {num} times in tour'
    for (i, j) in constraints:
        assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (10 points)')


# ## Answers to Select Problems
# 
# ### 1A part A
# 
# - Vertex 0: $k$ edges leave and $k$ edges enter.
# - Vertex 1, ..., n-1: 1 edge leaves and 1 edge enters (same as TSP).
# 
# ### 1A part B
# 
# This is a trick question. There is no need to change any of the time stamp related constraints.
# 

# ## That's All Folks!
