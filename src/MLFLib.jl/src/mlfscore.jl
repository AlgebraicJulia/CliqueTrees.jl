const SCORE = Float64

#   Node-selection scores for the minimum local fill algorithm.
#   For a supernode K with
#       |Fill(K)| = defncy[j]                (deficiency)
#       |K|       = qsize[j]                 (supernode size)
#       |Adj(K)|  = degree[j] - qsize[j]     (external degree)
#   the scores are
#       MinFill      |Fill(K)|                                      (MF)
#       MinAvgFill   |Fill(K)| / (|Adj(K)| + |K|)
#       MinMeanFill  |Fill(K)| / |K|                                (MMF)
#       MinIncDeg    |Fill(K)| - |Adj(K)| * |K|                     (MIND)
#       MinSqrtFill  |Fill(K)| / sqrt(|K|)
#   The last three are from Rothberg and Eisenstat, "Node selection
#   strategies for bottom-up sparse matrix ordering" (1998), Section 3.1.
@enum MLFScore MinFill MinAvgFill MinMeanFill MinIncDeg MinSqrtFill

function mlfscore(score::MLFScore, defncy::AbstractVector, degree::AbstractVector, qsize::AbstractVector, j::Integer)
    @inbounds def = convert(SCORE, defncy[j])

    if score == MinFill
        key = def
    elseif score == MinAvgFill
        @inbounds deg = convert(SCORE, degree[j])
        key = def / deg
    elseif score == MinMeanFill
        @inbounds q = convert(SCORE, qsize[j])
        key = def / q
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
