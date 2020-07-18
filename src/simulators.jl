# Different ways to simulate molecules

export
    accelerations,
    VelocityVerlet,
    simulate!,
    VelocityFreeVerlet

"""
    accelerations(simulation, neighbours; parallel=true)

Calculate the accelerations of all atoms using the general and specific
interactions and Newton's second law.
"""
function accelerations(coords, s::Simulation, neighbours, is, js; parallel::Bool=true)
    n_atoms = length(coords)
    forces = zero(coords)

    for inter in values(s.general_inters)
        if inter.nl_only
            forces -= reshape(sum(
                        force.((coords,), is, js, (inter,), (s,), neighbours), dims=2), n_atoms)
        else
            forces -= reshape(sum(
                        force.((coords,), is, js, (inter,), (s,)), dims=2), n_atoms)
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_forces = force.((coords,), inter_list, (s,))
        sparse_vecs = SparseVector.(n_atoms, getindex.(sparse_forces, 1),
                                    getindex.(sparse_forces, 2))
        forces += Array(sum(sparse_vecs))
    end

    return forces ./ getproperty.(s.atoms, :mass)
end

"""
    VelocityVerlet()

The velocity Verlet integrator.
"""
struct VelocityVerlet <: Simulator end

"""
    simulate!(simulation; parallel=true)
    simulate!(simulation, n_steps; parallel=true)
    simulate!(simulation, simulator, n_steps; parallel=true)

Run a simulation according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(s::Simulation,
                    ::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    # See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
    #   integration algorithm - used shorter second version
    n_atoms = length(s.coords)
    is = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    js = permutedims(is, (2, 1))
    coords, velocities = s.coords, s.velocities
    neighbours = find_neighbours(coords, s.box_size, zeros(typeof(s.timestep), n_atoms, n_atoms),
                                    is, js, s.neighbour_finder, 0, parallel=parallel)
    accels_t = accelerations(coords, s, neighbours, is, js, parallel=parallel)
    accels_t_dt = zero(s.coords)

    for step_n in 1:n_steps
        Zygote.ignore() do
            s.coords[1:end] = coords[1:end]
            s.velocities[1:end] = velocities[1:end]
            for logger in values(s.loggers)
                log_property!(logger, s, step_n)
            end
        end

        coords += velocities * s.timestep + (accels_t * s.timestep ^ 2) / 2
        coords = adjust_bounds_vec.(coords, s.box_size)
        accels_t_dt = accelerations(coords, s, neighbours, is, js, parallel=parallel)
        velocities += (accels_t + accels_t_dt) * s.timestep / 2

        velocities = apply_thermostat!(velocities, s, s.thermostat)
        neighbours = find_neighbours(coords, s.box_size, neighbours,
                                        is, js, s.neighbour_finder, step_n,
                                        parallel=parallel)

        accels_t = accels_t_dt
        Zygote.ignore() do
            s.n_steps_made[1] += 1
        end
    end
    return coords
end

"""
    VelocityFreeVerlet()

The velocity-free Verlet integrator, also known as the Störmer method.
In this case the `velocities` given to the `Simulator` act as the previous step
coordinates for the first step.
"""
struct VelocityFreeVerlet <: Simulator end

function simulate!(s::Simulation,
                    ::VelocityFreeVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    n_atoms = length(s.coords)
    is = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    js = permutedims(is, (2, 1))
    coords, coords_last = s.coords, s.velocities
    neighbours = find_neighbours(coords, s.box_size, zeros(typeof(s.timestep), n_atoms, n_atoms),
                                    is, js, s.neighbour_finder, 0, parallel=parallel)

    for step_n in 1:n_steps
        Zygote.ignore() do
            s.coords[1:end] = coords[1:end]
            s.velocities[1:end] = velocities[1:end]
            for logger in values(s.loggers)
                log_property!(logger, s, step_n)
            end
        end

        accels_t = accelerations(coords, s, neighbours, is, js, parallel=parallel)
        coords_copy = coords
        coords += vector.(coords_last, coords, s.box_size) + accels_t * s.timestep ^ 2
        coords = adjust_bounds_vec.(coords, s.box_size)
        coords_last = coords_copy

        #velocities = apply_thermostat!(velocities, s, s.thermostat)
        neighbours = find_neighbours(coords, s.box_size, neighbours,
                                        is, js, s.neighbour_finder, step_n,
                                        parallel=parallel)

        Zygote.ignore() do
            s.n_steps_made[1] += 1
        end
    end
    return coords
end

function simulate!(s::Simulation, n_steps::Integer; parallel::Bool=true)
    simulate!(s, s.simulator, n_steps, parallel=parallel)
end

function simulate!(s::Simulation; parallel::Bool=true)
    simulate!(s, s.n_steps - first(s.n_steps_made), parallel=parallel)
end
