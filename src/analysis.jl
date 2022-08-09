# Analysis tools

export
    visualize,
    displacements,
    distances,
    rdf,
    velocity_autocorr,
    rmsd,
    radius_gyration

"""
    visualize(coord_logger, boundary, out_filepath; <keyword arguments>)

Visualize a simulation as an animation.
This function is only available when GLMakie is imported.
GLMakie v0.6 or later should be used.
It can take a while to run, depending on the length of the simulation and the
number of atoms.

# Arguments
- `connections=Tuple{Int, Int}[]`: pairs of atoms indices to link with bonds.
- `connection_frames`: the frames in which bonds are shown. Should be a list of
    the same length as the number of frames, where each item is a list of
    `Bool`s of the same length as `connections`. Defaults to always `true`.
- `trails::Integer=0`: the number of preceding frames to show as transparent
    trails.
- `framerate::Integer=30`: the frame rate of the animation.
- `color=:purple`: the color of the atoms. Can be a single color or a list of
    colors of the same length as the number of atoms.
- `connection_color=:orange`: the color of the bonds. Can be a single color or a
    list of colors of the same length as `connections`.
- `markersize=0.05`: the size of the atom markers, in the units of the data.
- `linewidth=2.0`: the width of the bond lines.
- `transparency=true`: whether transparency is active on the plot.
- `show_boundary::Bool=true`: whether to show the bounding box as lines.
- `boundary_linewidth=2.0`: the width of the boundary lines.
- `boundary_color=:black`: the color of the boundary lines.
- `kwargs...`: other keyword arguments are passed to the point plotting
    function.
"""
function visualize end

function axis_limits(boundary_conv, coord_logger, dim)
    lim = boundary_conv[dim]
    if isinf(lim)
        # Find coordinate limits in given dimension
        low  = ustrip(minimum(cs -> minimum(c -> c[dim], cs), values(coord_logger)))
        high = ustrip(maximum(cs -> maximum(c -> c[dim], cs), values(coord_logger)))
        return low, high
    else
        return 0.0, lim
    end
end

"""
    displacements(coords, boundary)

Calculate the pairwise vector displacements of a set of coordinates, accounting
for the periodic boundary conditions.
"""
function displacements(coords, boundary)
    n_atoms = length(coords)
    coords_rep = repeat(reshape(coords, n_atoms, 1), 1, n_atoms)
    diffs = vector.(coords_rep, permutedims(coords_rep, (2, 1)), (boundary,))
    return diffs
end

"""
    distances(coords, boundary)

Calculate the pairwise distances of a set of coordinates, accounting for the
periodic boundary conditions.
"""
distances(coords, boundary) = norm.(displacements(coords, boundary))

"""
    rdf(coords, boundary; npoints=200)

Calculate the radial distribution function of a set of coordinates.
This describes how density varies as a function of distance from each atom.
Returns a list of distance bin centers and a list of the corresponding
densities.
"""
function rdf(coords, boundary; npoints::Integer=200)
    n_atoms = length(coords)
    dims = length(first(coords))
    dists = distances(coords, boundary)
    dists_vec = [dists[i, j] for i in 1:n_atoms, j in 1:n_atoms if j > i]
    dist_unit = unit(first(dists_vec))
    kd = kde(ustrip.(dists_vec), npoints=npoints)
    ρ = n_atoms / box_volume(boundary)
    if dims == 3
        normalizing_factor = 4π .* ρ .* step(kd.x) .* kd.x .^ 2 .* dist_unit .^ 3
    elseif dims == 2
        normalizing_factor = 2π .* ρ .* step(kd.x) .* kd.x .* dist_unit .^ 2
    end
    bin_centers = collect(kd.x) .* dist_unit
    density_weighted = kd.density ./ normalizing_factor
    return bin_centers, density_weighted
end

"""
    velocity_autocorr(vl, first_ind, last_ind)

Calculate the autocorrelation function of velocity from the velocity logger. 
This characterizes the similarity between velocities observed at different
time instances.
"""
function velocity_autocorr(vl::GeneralObservableLogger{T, typeof(velocities_wrapper)}, first_ind::Integer=1,
                            last_ind::Integer=length(values(vl))) where T
    n_atoms = length(first(values(vl)))
    return dot(values(vl)[first_ind], values(vl)[last_ind]) / n_atoms
end

"""
    rmsd(coords_1, coords_2)

Calculate the root-mean-square deviation (RMSD) of two sets of
3D coordinates after superimposition by the Kabsch algorithm.
Assumes the coordinates do not cross the bounding box, i.e. all
coordinates in each set correspond to the same periodic image.
"""
function rmsd(coords_1::AbstractArray{SVector{D, T}},
                coords_2::AbstractArray{SVector{D, T}}) where {D, T}
    n_atoms = length(coords_1)
    trans_1 = mean(coords_1)
    trans_2 = mean(coords_2)
    p = Array(reshape(reinterpret(T, coords_1), D, n_atoms)) .- repeat(reinterpret(T, trans_1), 1, n_atoms)
    q = Array(reshape(reinterpret(T, coords_2), D, n_atoms)) .- repeat(reinterpret(T, trans_2), 1, n_atoms)
    cov = p * transpose(q)
    svd_res = svd(ustrip.(cov))
    Ut = transpose(svd_res.U)
    d = sign(det(svd_res.V * Ut))
    dmat = [1 0 0; 0 1 0; 0 0 d]
    rot = svd_res.V * dmat * Ut
    diffs = rot * p - q
    msd = sum(abs2, diffs) / n_atoms
    return sqrt(msd)
end

sum_abs2(x) = sum(abs2, x)

"""
    radius_gyration(coords, atoms)

Calculate the radius of gyration of a set of coordinates.
Assumes the coordinates do not cross the bounding box, i.e. all
coordinates correspond to the same periodic image.
"""
function radius_gyration(coords, atoms)
    center = mean(coords)
    vecs_to_center = coords .- (center,)
    atom_masses = mass.(atoms)
    I = sum(sum_abs2.(vecs_to_center) .* atom_masses)
    return sqrt(I / sum(atom_masses))
end
