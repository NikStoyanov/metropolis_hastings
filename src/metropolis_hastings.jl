module metropolis_hastings

using UnPack
using Random
using Distributions

# https://python4mpia.github.io/fitting_data/Metropolis-Hastings.html
# Salpeter likelihood function
# \int_{n=1}^N dMcM^{-α} = c \frac{Mmax ^{1-α} - Mmin ^{1-α}}{1 - α} = 1
# Log-likelihood L = N * log(c) - α∑_{n=1}{N} log(Mn/M)
mutable struct Salpeter{T1, T2, T3}
    α::T1
    D::T2
    Mmin::T2
    Mmax::T2
    N::T3
end

function sample_salpeter(p::Salpeter)
    @unpack α, D, Mmin, Mmax, N = p
    log_Mmin = log(Mmin)
    log_Mmax = log(Mmax)

    max_likelihood = Mmin^(1.0 - α)

    masses = []

    for i = 1:N
        log_M = rand(Uniform(log_Mmin, log_Mmax))
        M = exp(log_M)
        likelihood = M^(1.0 - α)
        accept = rand(Uniform(0.0, max_likelihood))

        if accept < likelihood
            append!(masses, M)
        end
    end

    return masses
end

function evaluate_logLikelihood(p::Salpeter)
    @unpack α, D, Mmin, Mmax, N = p
    c = (1.0 - α) / (Mmax^(1.0 - α) - Mmin^(1.0 - α))
    return N * log(c) - α*D
end

function train(p::Salpeter, α₀, stepsize, iter)
    accepted = 0
    αₛ = [α₀]

    αᵢ = α₀
    p.α = αᵢ
    D = p.D

    for n = 1:iter
        αᵢ₋₁ = αᵢ
        loglikeᵢ₋₁ = evaluate_logLikelihood(p)
        αᵢ = rand(Normal(αᵢ₋₁, stepsize))
        p.α = αᵢ
        loglikeᵢ = evaluate_logLikelihood(p)

        if loglikeᵢ > loglikeᵢ₋₁
            append!(αₛ, αᵢ)
            accepted = accepted + 1
        else
            u = rand(Uniform(0.0, 1.0))
            if u < exp(loglikeᵢ - loglikeᵢ₋₁)
                append!(αₛ, αᵢ)
                accepted = accepted + 1
            else
                append!(αₛ, αᵢ₋₁)
            end
        end
    end

    return accepted, αₛ
end

export Salpeter, sample_salpeter, train

end
