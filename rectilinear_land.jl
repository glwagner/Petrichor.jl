using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ∂²zᵃᵃᶜ, Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using KernelAbstractions: @kernel, @index

@kernel function _step_temperature!(T, grid, κ, Q, Δt)
    i, j, k = @index(Global, NTuple)

    # Interior heat flux
    dTdt = κ * ∂²zᵃᵃᶜ(i, j, k, grid, T)

    # add surface contribution
    Qij = @inbounds Q[i, j, 1]
    Δz = Δzᵃᵃᶜ(i, j, k, grid)
    dTdt += ifelse(k == grid.Nz, - Qij / Δz, zero(grid))
    
    @inbounds T[i, j, k] += Δt * dTdt
end

function step_temperature!(T, κ, Q, Δt)
    fill_halo_regions!(T)
    grid = T.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz, _step_temperature!, T, grid, κ, Q, Δt)
    return nothing
end

arch = CPU()

# x will be the horizontal dimension
Nh = 100
Nz = 10
Lz = 10

grid = RectilinearGrid(arch, size=(Nh, Nz), x=(0, 1), z=(-Lz, 0), topology=(Periodic, Flat, Bounded))
Q = Field{Center, Center, Nothing}(grid) # surface heat flux
T = CenterField(grid) # temperature

set!(Q, 10) # modest cooling
set!(T, 20)

Δt = 0.1
κ = 1 # m² / s

for n = 1:100
    step_temperature!(T, κ, Q, Δt)
end
