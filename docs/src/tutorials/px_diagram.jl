# # Computing a p-x Diagram with the HANNA model
#
# This tutorial shows how to compute and visualize an isothermal p-x-y diagram for a binary mixture.
# We use the **HANNA model** [specht_hanna_2024](@cite) (`ogHANNA`) to predict the activity coefficients and couple it to the **Peng-Robinson EOS** (`PR`) for pure-component saturation pressures.
#
# As an example system we use the system **ethanol (1) + benzene (2)** at T = 333.15 K, a classic mixture with a low-boiling azeotrope.
#
# ## Setup
#
# Load the required packages.

using MLThermoProperties
using Clapeyron

# ## Building the model
#
# [`ogHANNA`](@ref) accepts component names that resolve to canonical SMILES and molecular weights through the Clapeyron database. 
# The `puremodel` keyword selects the EOS used for pure-component saturation properties -- here `PR` (Peng-Robinson).

model = ogHANNA(["ethanol", "benzene"]; puremodel = PR)

# ## Checking activity coefficients
#
# Before running phase-equilibrium calculations we verify that the model returns physically
# reasonable activity coefficients at an equimolar composition.  Both γᵢ > 1 is consistent with the low-boiling (minimum-pressure) azeotrope behaviour of this system.

T = 333.15  ## K  (60 °C)
γ = activity_coefficient(model, 1e5, T, [0.5, 0.5])

# `γ[1]` is γ(ethanol) and `γ[2]` is γ(benzene).

# ## Computing the p-x-y diagram
#
# We scan liquid compositions ``x_2 \in [0, 1]\; \rm{mol\,mol^{-1}}`` and use Clapeyron's `bubble_pressure` solver at each point (see also its documentation [here](https://clapeyronthermo.github.io/Clapeyron.jl/stable/properties/multi/#Clapeyron.bubble_pressure)).
# The function returns `(p, v_l, v_v, y)` where `y` is the equilibrium vapour composition.

N  = 100
x2s = range(0, 1.0, N)

p_bub = zeros(N)
y2s = zeros(N)

for (i, x2) in enumerate(x2s)
    p_bub[i], _, _, (_, y2s[i]) = bubble_pressure(model, T, [1 - x2, x2])
end

# ## Visualisation
#
# The two curves share the same pressure axis.  They meet at the pure-component endpoints and form a lens shape with the azeotrope at the pressure maximum.

using CairoMakie

fig = Figure(size = (600, 400))
ax  = Axis(fig[1, 1]; 
    xlabel = "x₂, y₂ / mol mol⁻¹", 
    ylabel = "p / kPa", 
    title = "ethanol (1) + benzene (2) at T = 333.15 K",
    limits = ((0,1),nothing)
)
lines!(ax, collect(x2s), p_bub./1e3; label = "bubble curve", color=:red)
lines!(ax, y2s, p_bub./1e3; label = "dew curve", color=:blue)
Legend(fig[1,2], ax, framevisible=false)
fig

# The azeotrope appears at ``x_2 \approx 0.56 \; \rm{mol\,mol^{-1}}`` and ``p \approx 78.7 \; \rm{kPa}``. 
# The `ogHANNA` model recovers this non-ideal behaviour from SMILES alone, without system-specific fitted parameters.
