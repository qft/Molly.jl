# Thermostats

export
    NoThermostat,
    apply_thermostat!,
    AndersenThermostat,
    RescaleThermostat,
    BerendsenThermostat,
    FrictionThermostat,
    velocity,
    maxwellboltzmann,
    temperature

"""
    NoThermostat()

Placeholder thermostat that does nothing.
"""
struct NoThermostat <: Thermostat end

"""
    apply_thermostat!(simulation, thermostat)

Apply a thermostat to modify a simulation.
Custom thermostats should implement this function.
"""
function apply_thermostat!(velocities, s::Simulation, ::NoThermostat)
    return velocities
end

"""
    AndersenThermostat(coupling_const)

Rescale random velocities according to the Andersen thermostat.
"""
struct AndersenThermostat{T} <: Thermostat
    coupling_const::T
end

function apply_thermostat!(s::Simulation, thermostat::AndersenThermostat)
    dims = length(first(s.velocities))
    for i in 1:length(s.velocities)
        if rand() < s.timestep / thermostat.coupling_const
            mass = s.atoms[i].mass
            s.velocities[i] = velocity(mass, s.temperature; dims=dims)
        end
    end
    return s
end

struct RescaleThermostat{T} <: Thermostat
    target_temp::T
end

function apply_thermostat!(velocities, s, thermostat::RescaleThermostat)
    velocities *= sqrt(thermostat.target_temp / temperature(velocities, s))
    return velocities
end

struct BerendsenThermostat{T} <: Thermostat
    target_temp::T
    coupling_const::T
end

function apply_thermostat!(velocities, s, thermostat::BerendsenThermostat)
    λ2 = 1 + (s.timestep / thermostat.coupling_const) * (
                (thermostat.target_temp / temperature(velocities, s)) - 1)
    velocities *= sqrt(λ2)
    return velocities
end

struct FrictionThermostat{T} <: Thermostat
    friction_const::T
end

function apply_thermostat!(velocities, s, thermostat::FrictionThermostat)
    velocities *= thermostat.friction_const
    return velocities
end

"""
    velocity(mass, temperature; dims=3)
    velocity(T, mass, temperature; dims=3)

Generate a random velocity from the Maxwell-Boltzmann distribution.
"""
function velocity(T::Type, mass::Real, temp::Real; dims::Integer=3)
    return SVector([maxwellboltzmann(T, mass, temp) for i in 1:dims]...)
end

function velocity(mass::Real, temp::Real; dims::Integer=3)
    return velocity(DefaultFloat, mass, temp, dims=dims)
end

"""
    maxwellboltzmann(mass, temperature)
    maxwellboltzmann(T, mass, temperature)

Draw from the Maxwell-Boltzmann distribution.
"""
function maxwellboltzmann(T::Type, mass::Real, temp::Real)
    return rand(Normal(zero(T), sqrt(temp / mass)))
end

function maxwellboltzmann(mass::Real, temp::Real)
    return maxwellboltzmann(DefaultFloat, mass, temp)
end

svdot(v::SVector) = sum(v .* v)

"""
    temperature(simulation)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(s::Simulation)
    ke = sum([a.mass * dot(s.velocities[i], s.velocities[i]) for (i, a) in enumerate(s.atoms)]) / 2
    df = 3 * length(s.coords) - 3
    return 2 * ke / df
end

function temperature(velocities, s::Simulation)
    ke = sum(svdot.(velocities) .* getproperty.(s.atoms, :mass) / 2)
    df = 3 * length(velocities) - 3
    return 2 * ke / df
end
