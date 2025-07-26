# ---------- #
# lowerbound #
# ---------- #
lowerbound(weights::AbstractVector, ::WidthOrAlgorithm) = error()

# ----------- #
# permutation #
# ----------- #

permutation(weights::AbstractVector, alg::PermutationOrAlgorithm) = error()

# --------------- #
# eliminationtree #
# --------------- #

eliminationtree(weights::AbstractVector, alg::PermutationOrAlgorithm) = error()

# --------- #
# treewidth #
# --------- #

treewidth(weights::AbstractVector, alg::PermutationOrAlgorithm) = error()

# -------- #
# treefill #
# -------- #

treefill(weights::AbstractVector, alg::PermutationOrAlgorithm) = error()
