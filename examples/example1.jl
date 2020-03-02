using metropolis_hastings, Plots

const α      = 2.35
const Mmin   = 1.0
const Mmax   = 100.0
const N      = 10^6
p = Salpeter(α, 0.0, Mmin, Mmax, N)

const Masses = sample_salpeter(p)
const LogM   = log.(Masses)
const D      = sum(LogM)

const α₀ = 3.0
const stepsize = 0.005

accepted, αₛ = train(p, α₀, stepsize, 10000)

histogram(αₛ)
