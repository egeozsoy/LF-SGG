# distutils: language = c++
from branched_ssg_matcher cimport BranchedSSGMatcher

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
cdef class PyBranchedSSGMatcher:
    cdef BranchedSSGMatcher c_BranchedSSGMatcher  # Hold a C++ instance which we're wrapping
    # Getz
    def __init__(self):
        self.c_BranchedSSGMatcher = BranchedSSGMatcher()  # Create the C++ instance

    def branched_matching(self, gts_vector, preds_vector, int N=3, int depth_limit=15, allow_no_matching=False):
        return self.c_BranchedSSGMatcher.branched_matching(gts_vector, preds_vector, N, depth_limit, allow_no_matching)
