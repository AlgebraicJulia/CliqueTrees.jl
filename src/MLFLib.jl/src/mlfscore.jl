const SCORE = Float64

@enum MLFScore MinFill MinAvgFill

function mlfscore(score::MLFScore, defncy::AbstractVector, degree::AbstractVector, j::Integer)
    @inbounds def = defncy[j]

    if score == MinFill
        key = convert(SCORE, def)
    else
        @inbounds deg = degree[j]
        key = convert(SCORE, def) / convert(SCORE, deg)
    end

    return key
end
