from libcpp cimport

bool
from libcpp.map cimport

map
from libcpp.pair cimport

pair
from libcpp.vector cimport

vector

# Declare the class with cdef
cdef extern from "branched_ssg_matcher.h":
    cdef cppclass BranchedSSGMatcher:
        BranchedSSGMatcher() except +
        map[pair[int, int], int] branched_matching(vector[vector[int]] & gts_vector, vector[vector[int]] & preds_vector, int N, int depth_limit,
                                                   bool allow_no_matching) except +
