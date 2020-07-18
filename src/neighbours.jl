# Neighbour finders

export
    NoNeighbourFinder,
    find_neighbours,
    DistanceNeighbourFinder

"""
    NoNeighbourFinder()

Placeholder neighbour finder that returns no neighbours.
When using this neighbour finder, ensure that `nl_only` for the interactions is
set to `false`.
"""
struct NoNeighbourFinder <: NeighbourFinder end

"""
    find_neighbours(simulation, current_neighbours, neighbour_finder, step_n; parallel=true)

Obtain a list of close atoms in a system.
Custom neighbour finders should implement this function.
"""
function find_neighbours(coords,
                            box_size,
                            current_neighbours,
                            is,
                            js,
                            nf::NoNeighbourFinder,
                            step_n::Integer;
                            kwargs...)
    return Tuple{Int, Int}[]
end

"""
    DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff)
    DistanceNeighbourFinder(nb_matrix, n_steps)

Find close atoms by distance.
"""
struct DistanceNeighbourFinder{T} <: NeighbourFinder
    nb_matrix::Array{T, 2}
    n_steps::Int
    dist_cutoff::T
end

function DistanceNeighbourFinder(nb_matrix::BitArray{2},
                                n_steps::Integer)
    return DistanceNeighbourFinder(nb_matrix, n_steps, 1.2)
end

function find_neighbours(coords,
                            box_size,
                            current_neighbours,
                            is,
                            js,
                            nf::DistanceNeighbourFinder,
                            step_n::Integer;
                            parallel::Bool=true)
    if step_n % nf.n_steps == 0
        sqdist_cutoff = nf.dist_cutoff ^ 2
        n_atoms = length(coords)
        sqdists = square_distance.(is, js, (coords,), box_size)
        lt = typeof(nf.dist_cutoff).(sqdists .< sqdist_cutoff)
        return lt #.* nf.nb_matrix # Why does this error?
    else
        return current_neighbours
    end
end
