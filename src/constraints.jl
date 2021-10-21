# Bond and angle constraints

export BondConstraint

#
struct BondConstraint{D, M}
    i::Int
    j::Int
    distance::D
    reduced_mass::M
end

# Constant Constraint Matrix Approximation method
# See Eastman and Pande 2010
function ccma_matrix(atoms, atoms_data, bonds, angles, coords, box_size)
    T = Float64
    elements = [at.element for at in atoms_data]
    masses = mass.(atoms)

    bond_constraints = BondConstraint[]
    angle_constraints = Tuple{Int, Int, Int}[]

    for bond in bonds
        if elements[bond.i] == "H" || elements[bond.j] == "H"
            i, j = sort([bond.i, bond.j])
            dist = norm(vector(coords[i], coords[j], box_size))
            reduced_mass = inv(2 * (inv(masses[i]) + inv(masses[j])))
            push!(bond_constraints, BondConstraint(i, j, dist, reduced_mass))
        end
    end

    for angle in angles
        n_hydrogen = 0
        if elements[angle.i] == "H"
            n_hydrogen += 1
        end
        if elements[angle.k] == "H"
            n_hydrogen += 1
        end
        if n_hydrogen == 2 || (n_hydrogen == 1 && elements[angle.j] == "O")
            i, k = sort([angle.i, angle.k])
            push!(angle_constraints, (i, angle.j, k))
        end
    end

    n_constraints = length(bond_constraints)
    K = spzeros(T, n_constraints, n_constraints)
    for ci in 1:n_constraints
        for cj in 1:n_constraints
            cj_inds = (bond_constraints[cj].i, bond_constraints[cj].j)
            if ci == cj
                K[ci, ci] = one(T)
            elseif bond_constraints[ci].i in cj_inds || bond_constraints[ci].j in cj_inds
                if bond_constraints[ci].i in cj_inds
                    shared_ind = bond_constraints[ci].i
                    other_ind_i = bond_constraints[ci].j
                else
                    shared_ind = bond_constraints[ci].j
                    other_ind_i = bond_constraints[ci].i
                end
                if shared_ind == bond_constraints[cj].i
                    other_ind_j = bond_constraints[cj].j
                else
                    other_ind_j = bond_constraints[cj].i
                end
                sort_other_ind_1, sort_other_ind_2 = sort([other_ind_i, other_ind_j])
                angle_constraint_ind = findfirst(isequal((sort_other_ind_1, shared_ind, sort_other_ind_2)), angle_constraints)
                if isnothing(angle_constraint_ind)
                    # If the angle is unconstrained, use the equilibrium angle of the harmonic force term
                    angle_force_ind = findfirst(angles) do a
                        (other_ind_i, shared_ind, other_ind_j) in ((a.i, a.j, a.k), (a.k, a.j, a.i))
                    end
                    isnothing(angle_force_ind) && error("No angle term found for atoms ", (other_ind_i, shared_ind, other_ind_j))
                    cos_θ = cos(angles[angle_force_ind].th0)
                else
                    # If the angle is constrained, use the actual constrained angle
                    ba = vector(coords[shared_ind], coords[other_ind_i], box_size)
                    bc = vector(coords[shared_ind], coords[other_ind_j], box_size)
                    cos_θ = dot(ba, bc) / (norm(ba) * norm(bc))
                end
                mass_term = inv(masses[shared_ind]) / (inv(masses[shared_ind]) + inv(masses[other_ind_i]))
                K[ci, cj] = mass_term * cos_θ
            end
        end
    end

    inv_K = qr(K).Q
    return [bond_constraints...], inv_K
end

function applyconstraints(inv_K, atoms, coords, box_size, bond_constraints,
                            tolerance=1e-5u"nm", max_n_iters=150)
    n_constraints = length(bond_constraints)
    vecs_start = [vector(coords[bc.i], coords[bc.j], box_size) for bc in bond_constraints]

    lower_tol = 1.0 - 2 * tolerance + tolerance ^ 2
    upper_tol = 1.0 + 2 * tolerance + tolerance ^ 2

    for iter_i in 1:max_n_iters
        n_converged = 0
        deltas = eltype(coords)[]
        for (bond_constraint, vec_start) in zip(bond_constraints, vecs_start)
            dr = vector(coords[i], coords[j], box_size)
            r2 = sum(abs2, dr)
            diff = r2 - bond_constraint.distance ^ 2
            rrpr = dot(dr, vec_start)
            push!(delatas, bond_constraint.reduced_mass * diff / rrpr)
            if r2 >= (lower_tol * bond_constraint.distance ^ 2) && r2 <= (upper_tol * bond_constraint.distance ^ 2)
                n_converged += 1
            end
        end

        if n_converged == n_constraints
            break
        end
    end
end