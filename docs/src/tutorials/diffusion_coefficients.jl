# # Diffusion Coefficients at Infinite Dilution with ESE
#
# This tutorial shows how to compute diffusion coefficients at infinite dilution for binary liquid mixtures with the **ESE model** [wagner_hybrid_2026](@cite) (`ESE`) from `MLThermoProperties.jl`, coupled to viscosity models from [`EntropyScaling.jl`](https://github.com/se-schmitt/EntropyScaling.jl).
#
# Internally, ESE evaluates a Stokes--Einstein expression that requires the *solvent* viscosity ``\eta_j`` at the conditions of interest.  The viscosity is therefore supplied through the `vismodel` keyword of [`ESE`](@ref).  Three options are demonstrated below:
#
# 1. the default `RefpropRES` (fluid-specific reference correlation),
# 2. `GCES` (group-contribution model, for systems where `RefpropRES` parameters are unavailable), and
# 3. `ConstantModel` (when only a single experimental viscosity value is known).
#
# ## Setup
#
# Load the required packages.  `CoolProp` is needed so that `RefpropRES` can access the underlying multi-fluid Helmholtz EOS via `Clapeyron`.

using MLThermoProperties
using EntropyScaling
using CoolProp

# ## Default: `RefpropRES` for ethanol + water
#
# When no viscosity model is specified, [`ESE`](@ref) builds a `RefpropRES` model for the components.  This is the recommended choice whenever fluid-specific reference parameters are available -- both ethanol and water are covered.

model = ESE(["ethanol", "water"])

# `inf_diffusion_coefficient` (from `EntropyScaling.jl`) returns the full ``N \times N`` matrix of infinite-dilution diffusion coefficients when called positionally.  Diagonal entries are zero by convention; off-diagonals carry the physical values.

D_matrix = inf_diffusion_coefficient(model, 1e5, 300.0)

# Individual entries can be requested by component name (or index) via the `solute` / `solvent` keyword arguments.

D_eth = inf_diffusion_coefficient(model, 1e5, 300.0; solute="ethanol", solvent="water")

# ### Temperature dependence
#
# We scan a typical liquid-phase temperature range at ``p = 1 \; \rm{bar}`` and compute the diffusion coefficient of ethanol in water and of water in ethanol.

p   = 1e5
Ts  = range(280.0, 340.0; length=40)

D_eth_in_w = inf_diffusion_coefficient.(model, p, Ts; solute="ethanol", solvent="water")
D_w_in_eth = inf_diffusion_coefficient.(model, p, Ts; solute="water",   solvent="ethanol")

using CairoMakie

fig1 = Figure(size = (600, 400))
ax1  = Axis(fig1[1, 1];
    xlabel = "T / K",
    ylabel = "D∞ / 10⁻⁹ m² s⁻¹",
    title  = "ethanol (1) + water (2) - RefpropRES",
)
lines!(ax1, Ts, D_eth_in_w .* 1e9; label = "ethanol in water", color = :red)
lines!(ax1, Ts, D_w_in_eth .* 1e9; label = "water in ethanol", color = :blue)
Legend(fig1[1, 2], ax1, framevisible = false)
fig1

# Both curves rise monotonically with temperature, as expected from the ``T/\eta(T)`` scaling of the Stokes-Einstein relation.

# ## Alternative: `GCES` for aniline + ethanol
#
# `RefpropRES` has no parameters for aniline, so the default model cannot be built for this mixture.  The group-contribution `GCES` viscosity model fills the gap whenever a PCP-SAFT group decomposition is available -- aniline (`CH_arom`, `C_arom`, `NH2`) and ethanol (`CH3`, `CH2`, `OH`) are both covered.
#
# Passing `vismodel = GCES` (as a *type*) instructs the constructor to build the viscosity model itself.

model_gces = ESE(["aniline", "ethanol"]; vismodel = GCES)

D_an_eth = inf_diffusion_coefficient(model_gces, 1e5, 300.0; solute="aniline", solvent="ethanol")

# ### Temperature dependence

Ts_ae  = range(290.0, 340.0; length=20)
D_ae   = inf_diffusion_coefficient.(model_gces, p, Ts_ae; solute="aniline", solvent="ethanol")

fig2 = Figure(size = (600, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "T / K",
    ylabel = "D∞ / 10⁻⁹ m² s⁻¹",
    title  = "aniline (1) in ethanol (2) - GCES",
)
lines!(ax2, Ts_ae, D_ae .* 1e9; color = :red)
fig2

# ## `ConstantModel`: when only the solvent viscosity is known
#
# In practice one often has an experimental viscosity for the solvent but not for the solute.  Because the ESE Stokes-Einstein expression evaluates only the *solvent* viscosity ``\eta_j``, the solute entry is irrelevant for a solute-in-solvent calculation and can safely be set to `NaN`.
#
# As an example we consider methylal in dodecane -- neither `RefpropRES` nor `GCES` covers methylal, but an experimental viscosity for the solvent dodecane at 300 K is available.  We pass a per-component vector of `ConstantModel`s and supply SMILES via `userlocations` (methylal is not in the default Clapeyron database).  No plot is needed -- `ConstantModel` is independent of ``T``.

η_dodecane = 0.0013153   ## Pa·s, experimental value at 300 K

model_const = ESE(
    ["methylal", "dodecane"];
    userlocations = (; SMILES = ["COCOC", "CCCCCCCCCCCC"]),
    vismodel = [ConstantModel(Viscosity(), NaN), ConstantModel(Viscosity(), η_dodecane)],
)

D_const = inf_diffusion_coefficient(model_const, 1e5, 300.0; solute="methylal", solvent="dodecane")

# The result is ``\approx 1.68 \times 10^{-9} \; \rm{m^2\,s^{-1}}``.
#
# ## Summary
#
# - `RefpropRES` (the default) is preferred whenever fluid-specific reference parameters exist.
# - `GCES` extends ESE to systems lacking those parameters, at the cost of group-contribution accuracy.
# - `ConstantModel` is the right escape hatch when an experimental viscosity is available.
#
# Any `EntropyScaling.jl` viscosity model can be passed via the `vismodel` keyword -- as a type, as an instance, or as a per-component vector of instances.
