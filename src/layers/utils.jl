# Power iteration
function power_iteration!(u::AbstractVector, v::AbstractVector, weight::AbstractMatrix; eps=1f-12)
    mul!(v, transpose(weight), u)
    v ./= norm(v) + eps
    mul!(u, weight, v)
    u ./= norm(u) + eps
    return nothing
end

# Functions for warmstart
iswarmstart(::Val{warmstart}) where {warmstart} = warmstart
iswarmstart(st::NamedTuple) = hasproperty(st, :warmstart) && iswarmstart(st.warmstart)

# SILU activation function
silu(x) = @. x/(1+exp(-x))

# Cosine similarity
function cosine_similarity(x1,x2;eps=1e-8)
    ∑x1 = sqrt(dot(x1,x1))
    ∑x2 = sqrt(dot(x2,x2))
    return dot(x1,x2)/(max(∑x1,eps*one(∑x1))*max(∑x2,eps*one(∑x2)))
end

function onehot_encoder(value, allowed_list)
    return Float32.(value .== allowed_list)
end

function _validate_features(features::AbstractVector{Symbol}, allowed::Tuple, label::String)
    for feature in features
        feature ∈ allowed || error("Unsupported $(label) feature: $(feature)")
    end
end