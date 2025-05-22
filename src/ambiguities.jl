# ---------- #
# lowerbound #
# ---------- #
lowerbound(weights::AbstractVector, ::Number) = error()
lowerbound(weights::AbstractVector, ::MMW) = error()

# ----------- #
# permutation #
# ----------- #

permutation(weights::AbstractVector, alg::EliminationAlgorithm) = error()
permutation(weights::AbstractVector, alg::AbstractVector) = error()
permutation(weights::AbstractVector, alg::Tuple{AbstractVector, AbstractVector}) = error()
permutation(weights::AbstractVector, alg::BFS) = error()
permutation(weights::AbstractVector, alg::MCS) = error()
permutation(weights::AbstractVector, alg::LexBFS) = error()
permutation(weights::AbstractVector, alg::RCMMD) = error()
permutation(weights::AbstractVector, alg::RCMGL) = error()
permutation(weights::AbstractVector, alg::LexM) = error()
permutation(weights::AbstractVector, alg::MCSM) = error()
permutation(weights::AbstractVector, alg::AMF) = error()
permutation(weights::AbstractVector, alg::MF) = error()
permutation(weights::AbstractVector, alg::MMD) = error()
permutation(weights::AbstractVector, alg::AMD) = error()
permutation(weights::AbstractVector, alg::SymAMD) = error()
permutation(weights::AbstractVector, alg::METIS) = error()
permutation(weights::AbstractVector, alg::ND) = error()
permutation(weights::AbstractVector, alg::Spectral) = error()
permutation(weights::AbstractVector, alg::BT) = error()
permutation(weights::AbstractVector, alg::SAT) = error()
permutation(weights::AbstractVector, alg::MinimalChordal) = error()
permutation(weights::AbstractVector, alg::CompositeRotations) = error()
permutation(weights::AbstractVector, alg::SafeRules) = error()
permutation(weights::AbstractVector, alg::SafeSeparators) = error()
permutation(weights::AbstractVector, alg::ConnectedComponents) = error()
permutation(weights::AbstractVector, alg::BestWidth) = error()
permutation(weights::AbstractVector, alg::BestFill) = error()

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
