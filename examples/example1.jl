using metropolis_hastings, Plots, Statistics

const α      = 2.35
const Mmin   = 1.0
const Mmax   = 100.0
const N      = 10^6
p = Salpeter(α, 0.0, Mmin, Mmax, N)

const Masses = sample_salpeter(p)
const LogM   = log.(Masses)
const D      = sum(LogM)

const α₀ = 3.0
const stepsize = 0.01

iter = 10000
accepted, αₛ = train(p, α₀, stepsize, 10000)

# Remove first half of chain and thin
clean = zeros(ceil(Int, iter/20))

global j = 1
for i = range(ceil(Int, iter/2), stop=floor(Int, iter) - 1, step=1)
    if i % 10 == 0
        clean[j] = αₛ[i]
        global j = j + 1
    end
end

@show (accepted / iter)
@show clean
@show mean(clean)
@show std(clean)

histogram(clean)
savefig("histogram.svg")
