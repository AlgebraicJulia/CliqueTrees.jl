const SCORE = Float64

#   Node-selection scores for the minimum local fill algorithm.
#   For a supernode K with
#       |Fill(K)| = defncy[j]                (deficiency)
#       |K|       = qsize[j]                 (supernode size)
#       |Adj(K)|  = degree[j] - qsize[j]     (external degree)
#   the scores are
#       MinFill      |Fill(K)|                                      (MF)
#       MinAvgFill   |Fill(K)| / (|Adj(K)| + |K|)
#       MinMeanFill  |Fill(K)| / (|K| + qmin)                       (MMF)
#       MinIncDeg    |Fill(K)| - |Adj(K)| * |K|                     (MIND)
#       MinSqrtFill  |Fill(K)| / sqrt(|K|)
#   where qmin is the smallest positive vertex weight (one for unit
#   weights). Adding it to |K| gives lower fill than |Fill(K)| / |K| on
#   most test matrices, and the score does not depend on the unit of
#   the weights. The last three are from Rothberg and Eisenstat, "Node
#   selection strategies for bottom-up sparse matrix ordering" (1998),
#   Section 3.1.
@enum MLFScore MinFill MinAvgFill MinMeanFill MinIncDeg MinSqrtFill

function mlfscore(score::MLFScore, defncy::AbstractVector, degree::AbstractVector, qsize::AbstractVector, qmin, j::Integer)
    @inbounds def = convert(SCORE, defncy[j])

    if score == MinFill
        key = def
    elseif score == MinAvgFill
        @inbounds deg = convert(SCORE, degree[j])
        key = def / deg
    elseif score == MinMeanFill
        @inbounds q = convert(SCORE, qsize[j])
        key = def / (q + convert(SCORE, qmin))
    elseif score == MinIncDeg
        @inbounds q = convert(SCORE, qsize[j])
        @inbounds deg = convert(SCORE, degree[j])
        key = def - (deg - q) * q
    else
        @inbounds q = convert(SCORE, qsize[j])
        key = def / sqrt(q)
    end

    return key
end
