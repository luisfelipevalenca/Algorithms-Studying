#!/usr/bin/env python
# coding: utf-8

# # Problem Set \# 2 (Basic Datastructures and Heaps)
# 
# Topics covered:
#   - Basic data-structures
#   - Heap data-structures
#   - Using heaps and arrays to realize interesting functionality.

# ## Problem 1 (Least-k Elements Datastructure)
# 
# 
# We saw how min-heaps can efficiently allow us to query the least element in a heap (array). We would like to modify minheaps in this exercise to design a data structure to maintain the __least k__ elements for a  given $k \geq 1$ with $$k = 1$$ being the minheap data-structure.
# 
# Our design is to hold two arrays: 
#   - (a) a sorted array `A` of $k$ elements that forms our least k elements; and 
#   - (b) a minheap `H` with the remaining $n-k$ elements. 
# 
# Our data structure will itself be a pair of arrays `(A,H)` with the following property:
#  - `H` must be a minheap
#  - `A` must be sorted of size $k$.
#  - Every element of `A` must be smaller than every element of `H`.
# 
# The key operations to implement in this assignment include: 
#   - insert a new element into the data-structure
#   - delete an existing element from the data-structure.
# 
# 
# We will first ask you to design the data structure and them implement it.
# 
# ### (A) Design Insertion  Algorithm
# 
# Suppose we wish to insert a new element with key $j$ into this data structure. Describe the pseudocode. Your pseudocode must deal with two cases: when the inserted element $j$ would be one of the `least k` elements i.e, it belongs to the array `A`; or when the inserted element belongs to the heap `H`. How would you distinguish between the two cases? 
# 
# - You can assume that heap operations such as `insert(H, key)` and `delete(H, index)` are defined.
# - Assume that the heap is indexed as  `H[1]`,...,`H[n -k]` with `H[0]` being unused.
# - Assume $ n > k$, i.e, there are already more than $k$ elements in the data structure.
# 
# 
# What is the complexity of the insertion operation in the worst case in terms of $k, n$.
# 
# __Unfortunately, we cannot grade your answer. We hope you will use this to design your datastructure on paper before attempting to code it up__

# YOUR ANSWER HERE

# ### (B) Design Deletion Algorithm
# 
# Suppose we wish to delete an index $j$ from the top-k array $A$. Design an algorithm to perform this deletion. Assume that the heap is not empty, in which case you can assume that the deletion fails. 
# 
# 
# 
# - You can assume that heap operations such as `insert(H, key)` and `delete(H, index)` are defined.
# - Assume that the heap is indexed as  `H[1]`,...,`H[n -k]` with `H[0]` being unused.
# - Assume $ n > k$, i.e, there are already more than $k$ elements in the data structure.
# 
# What is the complexity of the insertion operation in the worst case in terms of $k, n$.
# 
# __Unfortunately, we cannot grade your answer. We hope you will use this to design your datastructure on paper before attempting to code it up__

# YOUR ANSWER HERE

# ## (C) Program your solution by completing the code below
# 
# Note that although your algorithm design above assume that your are inserting and deleting from cases where $n \geq k$, the data structure implementation below must handle $n < k$ as well. We have provided implementations for that portion to help you out.

# In[94]:


class MinHeap:
    def __init__(self):
        self.H = [None]

    def size(self):
        return len(self.H) - 1

    def __repr__(self):
        return str(self.H[1:])

    def satisfies_assertions(self):
        for i in range(2, len(self.H)):
            assert self.H[i] >= self.H[i // 2], f'Min heap property fails at position {i // 2}, parent elt: {self.H[i // 2]}, child elt: {self.H[i]}'

    def min_element(self):
        return self.H[1] if self.size() > 0 else None

    def bubble_up(self, index):
        assert index >= 1
        if index == 1:
            return
        parent_index = index // 2
        if self.H[parent_index] > self.H[index]:
            self.H[parent_index], self.H[index] = self.H[index], self.H[parent_index]
            self.bubble_up(parent_index)

    def bubble_down(self, index):
        assert index >= 1 and index < len(self.H)
        lchild_index = 2 * index
        rchild_index = 2 * index + 1
        lchild_value = self.H[lchild_index] if lchild_index < len(self.H) else float('inf')
        rchild_value = self.H[rchild_index] if rchild_index < len(self.H) else float('inf')
        if self.H[index] <= min(lchild_value, rchild_value):
            return
        min_child_value, min_child_index = min((lchild_value, lchild_index), (rchild_value, rchild_index))
        self.H[index], self.H[min_child_index] = self.H[min_child_index], self.H[index]
        self.bubble_down(min_child_index)

    def insert(self, elt):
        self.H.append(elt)
        index = len(self.H) - 1
        self.bubble_up(index)

    def delete_min(self):
        if len(self.H) <= 1:
            return None  # Heap is empty
        min_elt = self.H[1]
        last_elt = self.H.pop()
        if len(self.H) > 1:
            self.H[1] = last_elt
            self.bubble_down(1)
        return min_elt


# In[95]:


h = MinHeap()
print('Inserting: 5, 2, 4, -1 and 7 in that order.')
h.insert(5)
print(f'\t Heap = {h}')
assert(h.min_element() == 5)
h.insert(2)
print(f'\t Heap = {h}')
assert(h.min_element() == 2)
h.insert(4)
print(f'\t Heap = {h}')
assert(h.min_element() == 2)
h.insert(-1)
print(f'\t Heap = {h}')
assert(h.min_element() == -1)
h.insert(7)
print(f'\t Heap = {h}')
assert(h.min_element() == -1)
h.satisfies_assertions()

print('Deleting minimum element')
h.delete_min()
print(f'\t Heap = {h}')
assert(h.min_element() == 2)
h.delete_min()
print(f'\t Heap = {h}')
assert(h.min_element() == 4)
h.delete_min()
print(f'\t Heap = {h}')
assert(h.min_element() == 5)
h.delete_min()
print(f'\t Heap = {h}')
assert(h.min_element() == 7)
# Test delete_max on heap of size 1, should result in empty heap.
h.delete_min()
print(f'\t Heap = {h}')
print('All tests passed: 10 points!')


# In[108]:


class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.A = []
        self.H = MinHeap()
        self.size = 0

    def insert_into_A(self, elt):
        self.A.append(elt)
        self.A.sort()
        self.size += 1
        # If A becomes larger than k, move the largest element to H
        if len(self.A) > self.k:
            self.H.insert(self.A.pop())
            self.size -= 1

    def insert(self, elt):
        if self.size < self.k:
            self.insert_into_A(elt)
        else:
            if elt < self.A[-1]:
                self.H.insert(self.A[-1])
                self.A[-1] = elt
                self.A.sort()
            else:
                self.H.insert(elt)

    def delete_top_k(self, j):
        assert 0 <= j < self.k
        # Remove the jth element from A
        del self.A[j]
        self.size -= 1
        # Refill A from H if necessary
        if self.H.size() > 0:
            self.insert_into_A(self.H.delete_min())

    def size(self):
        return len(self.H) - 1

    def __repr__(self):
        return str(self.H[1:])

    def satisfies_assertions(self):
        # Assertions for the array A
        for i in range(len(self.A) - 1):
            assert self.A[i] <= self.A[i + 1], f'Array A fails to be sorted at position {i}, {self.A[i]}, {self.A[i + 1]}'
        
        # Assertions for the heap H
        self.H.satisfies_assertions()

        # Additional checks
        if len(self.A) > 0 and self.H.size() > 0:
            assert self.A[-1] <= self.H.min_element(), f'Largest element in A ({self.A[-1]}) is greater than the smallest element in H ({self.H.min_element()})'

    def max_element(self):
        return self.H[1] if self.size() > 0 else None

    def bubble_up(self, index):
        assert index >= 1
        if index == 1:
            return
        parent_index = index // 2
        if self.H[parent_index] < self.H[index]:
            self.H[parent_index], self.H[index] = self.H[index], self.H[parent_index]
            self.bubble_up(parent_index)

    def bubble_down(self, index):
        assert index >= 1 and index < len(self.H)
        lchild_index = 2 * index
        rchild_index = 2 * index + 1
        lchild_value = self.H[lchild_index] if lchild_index < len(self.H) else float('-inf')
        rchild_value = self.H[rchild_index] if rchild_index < len(self.H) else float('-inf')
        if self.H[index] >= max(lchild_value, rchild_value):
            return
        max_child_value, max_child_index = max((lchild_value, lchild_index), (rchild_value, rchild_index))
        self.H[index], self.H[max_child_index] = self.H[max_child_index], self.H[index]
        self.bubble_down(max_child_index)

    def delete_max(self):
        if len(self.H) <= 1:
            return None  # Heap is empty
        max_elt = self.H[1]
        last_elt = self.H.pop()
        if len(self.H) > 1:
            self.H[1] = last_elt
            self.bubble_down(1)
        return max_elt


# In[109]:


h = TopKHeap(5)
# Force the array A
h.A = [-10, -9, -8, -4, 0]
# Force the heap to this heap
[h.H.insert(elt) for elt in  [1, 4, 5, 6, 15, 22, 31, 7]]

print('Initial data structure: ')
print('\t A = ', h.A)
print('\t H = ', h.H)

# Insert an element -2
print('Test 1: Inserting element -2')
h.insert(-2)
print('\t A = ', h.A)
print('\t H = ', h.H)
# After insertion h.A should be [-10, -9, -8, -4, -2]
# After insertion h.H should be [None, 0, 1, 5, 4, 15, 22, 31, 7, 6]
assert h.A == [-10,-9,-8,-4,-2]
assert h.H.min_element() == 0 , 'Minimum element of the heap is no longer 0'
h.satisfies_assertions()

print('Test2: Inserting element -11')
h.insert(-11)
print('\t A = ', h.A)
print('\t H = ', h.H)
assert h.A == [-11, -10, -9, -8, -4]
assert h.H.min_element() == -2
h.satisfies_assertions()

print('Test 3 delete_top_k(3)')
h.delete_top_k(3)
print('\t A = ', h.A)
print('\t H = ', h.H)
h.satisfies_assertions()
assert h.A == [-11,-10,-9,-4,-2]
assert h.H.min_element() == 0
h.satisfies_assertions()

print('Test 4 delete_top_k(4)')
h.delete_top_k(4)
print('\t A = ', h.A)
print('\t H = ', h.H)
assert h.A == [-11, -10, -9, -4, 0]
h.satisfies_assertions()

print('Test 5 delete_top_k(0)')
h.delete_top_k(0)
print('\t A = ', h.A)
print('\t H = ', h.H)
assert h.A == [-10, -9, -4, 0, 1]
h.satisfies_assertions()

print('Test 6 delete_top_k(1)')
h.delete_top_k(1)
print('\t A = ', h.A)
print('\t H = ', h.H)
assert h.A == [-10, -4, 0, 1, 4]
h.satisfies_assertions()
print('All tests passed - 15 points!')


# ## Problem 2: Heap data structure to mantain/extract median (instead of minimum/maximum key)
# 
# We have seen how min-heaps can efficiently extract the smallest element efficiently and maintain the least element as we insert/delete elements. Similarly, max-heaps can maintain the largest element. In this exercise, we combine both to maintain the "median" element.
# 
# The median is the middle element of a list of numbers. 
# - If the list has size $n$ where $n$ is odd, the median is the $(n-1)/2^{th}$ element where $0^{th}$ is least and $(n-1)^{th}$ is the maximum. 
# - If $n$ is even, then we designate the median the average of the $(n/2-1)^{th}$ and $(n/2)^{th}$ elements.
# 
# 
# #### Example 
# 
# - List is $[-1, 5, 4, 2, 3]$ has size $5$, the median is the $2^{nd}$ element (remember again least element is designated as $0^{th}$) which is $3$.
# - List is $[-1, 3, 2, 1 ]$ has size $4$. The median element is the average of  $1^{st}$ element (1) and $2^{nd}$ element (2) which is  $1.5$.
# 
# ## Maintaining median using two heaps.
# 
# The data will be maintained as the union of the elements in two heaps $H_{\min}$ and $H_{\max}$, wherein $H_{\min}$ is a min-heap and $H_{\max}$ is a max-heap.  We will maintain the following invariant:
#   - The max element of  $H_{\max}$ will be less than or equal to the min element of  $H_{\min}$. 
#   - The sizes of $H_{max}$ and $H_{min}$ are equal (if number of elements in the data structure is even) or $H_{max}$ may have one less element than $H_{min}$ (if the number of elements in the data structure is odd).
#   
# 
# 
# ## (A)  Design algorithm for insertion.
# 
# Suppose, we have the current data split between $H_{max}$ and $H_{min}$ and we wish to insert an element $e$ into the data structure, describe the algorithm you will use to insert. Your algorithm must decide which of the two heaps will $e$ be inserted into and how to maintain the size balance condition.
# 
# Describe the algorithm below and the overall complexity of an insert operation. This part will not be graded.

# YOUR ANSWER HERE

# ## (B) Design algorithm for finding the median.
# 
# Implement an algorithm for finding the median given the heaps $H_{\min}$ and $H_{\max}$. What is its complexity?

# YOUR ANSWER HERE

# ## (C) Implement the algorithm
# 
# Complete the implementation for maxheap data structure.
# First complete the implementation of MaxHeap.  You can cut and paste relevant parts from previous problems although we do not really recommend doing that. A better solution would have been to write a single implementation that could have served as min/max heap based on a flag. 

# In[114]:


class MaxHeap:
    def __init__(self):
        self.H = [None]

    def size(self):
        return len(self.H) - 1

    def max_element(self):
        return self.H[1] if self.size() > 0 else None

    def bubble_up(self, index):
        while index // 2 > 0:
            if self.H[index] > self.H[index // 2]:
                self.H[index], self.H[index // 2] = self.H[index // 2], self.H[index]
            index = index // 2

    def bubble_down(self, index):
        while (index * 2) <= self.size():
            mc = self.max_child(index)
            if self.H[index] < self.H[mc]:
                self.H[index], self.H[mc] = self.H[mc], self.H[index]
            index = mc

    def max_child(self, index):
        if index * 2 + 1 > self.size():
            return index * 2
        else:
            return index * 2 if self.H[index * 2] > self.H[index * 2 + 1] else index * 2 + 1

    def insert(self, k):
        self.H.append(k)
        self.bubble_up(self.size())

    def delete_max(self):
        retval = self.H[1]
        self.H[1] = self.H[self.size()]
        self.H.pop()
        self.bubble_down(1)
        return retval

    def satisfies_assertions(self):
        for i in range(2, len(self.H)):
            assert self.H[i] <= self.H[i // 2], f'Max heap property fails at position {i // 2}, parent: {self.H[i // 2]}, child: {self.H[i]}'

    def __repr__(self):
        return str(self.H[1:])


# In[115]:


h = MaxHeap()
print('Inserting: 5, 2, 4, -1 and 7 in that order.')
h.insert(5)
print(f'\t Heap = {h}')
assert(h.max_element() == 5)
h.insert(2)
print(f'\t Heap = {h}')
assert(h.max_element() == 5)
h.insert(4)
print(f'\t Heap = {h}')
assert(h.max_element() == 5)
h.insert(-1)
print(f'\t Heap = {h}')
assert(h.max_element() == 5)
h.insert(7)
print(f'\t Heap = {h}')
assert(h.max_element() == 7)
h.satisfies_assertions()

print('Deleting maximum element')
h.delete_max()
print(f'\t Heap = {h}')
assert(h.max_element() == 5)
h.delete_max()
print(f'\t Heap = {h}')
assert(h.max_element() == 4)
h.delete_max()
print(f'\t Heap = {h}')
assert(h.max_element() == 2)
h.delete_max()
print(f'\t Heap = {h}')
assert(h.max_element() == -1)
# Test delete_max on heap of size 1, should result in empty heap.
h.delete_max()
print(f'\t Heap = {h}')
print('All tests passed: 5 points!')


# In[126]:


class MedianMaintainingHeap:
    def __init__(self):
        self.hmin = MinHeap()
        self.hmax = MaxHeap()

    def satisfies_assertions(self):
        # Check if both heaps are empty
        if self.hmin.size() == 0 and self.hmax.size() == 0:
            return

        # Check if only one of the heaps is empty
        if self.hmin.size() == 0 or self.hmax.size() == 0:
            assert abs(self.hmin.size() - self.hmax.size()) <= 1, "Heap sizes are unbalanced when one heap is empty"
            return

        # Check the max element of hmax is less than or equal to the min element of hmin
        assert self.hmax.max_element() <= self.hmin.min_element(),             f'Failed: Max element of max heap = {self.hmax.max_element()} > min element of min heap {self.hmin.min_element()}'

        # Check that the heap sizes are approximately equal
        s_min = self.hmin.size()
        s_max = self.hmax.size()
        assert s_min == s_max or s_min == s_max + 1 or s_max == s_min + 1,             f'Heap sizes are unbalanced. Min heap size = {s_min} and Max heap size = {s_max}'

    def __repr__(self):
        return f'Maxheap: {self.hmax} Minheap: {self.hmin}'

    def get_median(self):
        if self.hmin.size() == 0 and self.hmax.size() == 0:
            raise ValueError("Cannot ask for median from empty heaps")

        if self.hmax.size() == self.hmin.size():
            return (self.hmin.min_element() + self.hmax.max_element()) / 2.0
        elif self.hmax.size() > self.hmin.size():
            return self.hmax.max_element()
        else:
            return self.hmin.min_element()

    def balance_heap_sizes(self):
        while abs(self.hmin.size() - self.hmax.size()) > 1:
            if self.hmin.size() > self.hmax.size():
                elt = self.hmin.delete_min()
                self.hmax.insert(elt)
            else:
                elt = self.hmax.delete_max()
                self.hmin.insert(elt)

    def insert(self, elt):
        if self.hmax.size() == 0 or elt <= self.hmax.max_element():
            self.hmax.insert(elt)
        else:
            self.hmin.insert(elt)
        self.balance_heap_sizes()

    def delete_median(self):
        if self.hmin.size() == 0 and self.hmax.size() == 0:
            raise ValueError("Cannot delete median from empty heaps")

        if self.hmin.size() == self.hmax.size():
            return self.hmax.delete_max()
        else:
            return self.hmin.delete_min() if self.hmin.size() > self.hmax.size() else self.hmax.delete_max()



# In[127]:


m = MedianMaintainingHeap()
print('Inserting 1, 5, 2, 4, 18, -4, 7, 9')

m.insert(1)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 1,  f'expected median = 1, your code returned {m.get_median()}'

m.insert(5)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 3,  f'expected median = 3.0, your code returned {m.get_median()}'

m.insert(2)
print(m)
print(m.get_median())
m.satisfies_assertions()

assert m.get_median() == 2,  f'expected median = 2, your code returned {m.get_median()}'
m.insert(4)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 3,  f'expected median = 3, your code returned {m.get_median()}'

m.insert(18)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 4,  f'expected median = 4, your code returned {m.get_median()}'

m.insert(-4)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 3,  f'expected median = 3, your code returned {m.get_median()}'

m.insert(7)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median() == 4, f'expected median = 4, your code returned {m.get_median()}'

m.insert(9)
print(m)
print(m.get_median())
m.satisfies_assertions()
assert m.get_median()== 4.5, f'expected median = 4.5, your code returned {m.get_median()}'

print('All tests passed: 15 points')


# ## Solutions to Manually Graded Portions
# 
# ### Problem 1 A
# In order to insert a new element `j`, we will first  distinguish between two cases:
# - $j < A[k-1]$ : In this case $j$ belongs to the array $A$.
#   - First, let $j' = A[k-1]$. 
#   - Replace $A[k-1]$ by $j$.
#   - Perform an insertion to move $j$ into its correct place in the sorted array $A$.
#   - Insert $j'$ into the heap using heap insert.
# - $j \geq A[k-1]$: In this case, $j$ belongs to the heap $H$.
#   - Insert $j$ into the heap using heap-insert.
#   
# In terms of $k, n$, the worst case complexity is $\Theta(k + \log(n))$ for each insertion operation.
# 
# ### Problem 1B 
# 
# - First, in order to delete the index j from array, move elements from j+1 .. k-1 left one position.
# - Insert the minimum heap element at position $k-1$ of the array A.
# - Delete the element at index 1 of the heap.
# 
# Overall complexity = $\Theta(k + \log(n))$ in the worst case.
# 
# ### Problem 2 A 
# 
# Let $a$ be the largest element in $H_{\max}$ and $b$ be the least element in $H_{\min}$. 
#  - If $elt < a$, then we insert the new element into $H_{\max}$.
#  - If $elt >= a$, then we insert the new element into $H_{\min}$. 
#  
#  If the size of $H_{\max}$ and $H_{\min}$ differ by 2, then 
#  - If $H_{\max}$ is larger then, extract the largest element from $H_{\max}$ andd insert into $H_{\min}$.
#  - If $H_{\min}$ is larger then,  extract the least element from $H_{\min}$ andd insert into $H_{\max}$.
#  
#  The overall complexity is $\Theta(\log(n))$.
#  
# ### Problem 2 B 
#  
#  If sizes of heaps are the same, then median is the average of maximum element of the max heap and minimum element of the minheap.
# 
# Otherwise, the median is simply the minimum elemment of the min-heap. 
# 
# Overall complexity is $\Theta(1)$.
# 
# 

# ## That's all folks

# In[ ]:




