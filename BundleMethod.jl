module BundleMethod

using GLPK
using Ipopt
using LinearAlgebra
using JuMP
using Debugger
using ProgressBars
using Statistics
using DataStructures

using ..Utils

export bundle_method_cycle, StabProximal, StabTrust

mutable struct StabTrust
  step::Function
  δ::Float64
  SS_update!::Function
  NS_update!::Function

  function StabTrust(; δ=.01, δ_incr=1.1, δ_max=.01, δ_decr=.9, δ_min=1e-4)
    SS_update = (x) -> x.δ = min(x.δ * 1.1, .01)
    NS_update = (x) -> x.δ = max(x.δ * .9, .0001)
    new(StabTrustRegion_stabilizer_step, δ, SS_update, NS_update)
  end
end

function StabTrustRegion_stabilizer_step(x̄::Array{Float64}, B::Queue{Tuple{Array{Float64}, Float64}}, precision, stabilizer)
  m = size(x̄)[1]
  δ_vec = fill(stabilizer.δ, size(x̄))

  tol = if (precision <= 1e-7) 1e-15 else 1e-8 end
  opt = optimizer_with_attributes(GLPK.Optimizer,
                                  "tol_bnd" => tol, "tol_obj" => tol,
                                  "tol_int" => tol, "tol_piv" => tol,
                                  "tol_dj" => tol) 
  master_problem = Model(opt)

  @variable(master_problem, v)
  @variable(master_problem, d[1:m])
  @objective(master_problem, Min, v)
  # Bundle constraints
  for (z_b, α_b) in B
    @constraint(master_problem, v >= dot(z_b, x̄ .+ d) + α_b)
  end
  # L-infinity norm
  @constraint(master_problem, -δ_vec .<= d .<= δ_vec)

  status = optimize!(master_problem)
  
  x_ii = JuMP.value.(d) .+ x̄
  return x_ii, JuMP.value.(v), termination_status(master_problem)
end
  
  
mutable struct StabProximal 
  step::Function
  μ::Float64
  SS_update!::Function
  NS_update!::Function

  function StabProximal(; μ=1., μ_incr=1.2, μ_max=1e+2, μ_decr=.98, μ_min=1e-2)
    SS_update = (x) -> x.μ = max(x.μ * .98, 1e-2)
    NS_update = (x) ->  x.μ = min(x.μ * 1.2, 1.)
    new(StabProximal_stabilizer_step, μ, SS_update, NS_update)
  end
end

function StabProximal_stabilizer_step(x̄::Array{Float64}, B::Queue{Tuple{Array{Float64}, Float64}}, precision, stabilizer)
  m = size(x̄)[1]

  tol = if (precision > 1e-7) 1e-8 else 1e-15 end
  max_iter = if (precision > 1e-7) 500 else 5000 end
  opt = optimizer_with_attributes(Ipopt.Optimizer, 
                                  "max_iter" => max_iter,
                                  "tol" => tol, "bound_relax_factor" => tol, "constr_viol_tol" => tol,
                                  "print_level" => 0, "sb"=>"yes")
  master_problem = Model(opt)

  @variable(master_problem, v)
  @variable(master_problem, d[1:m])
  # JuMP doesn't support norm or vector operations ¯\_(ツ)_/¯
  # Min{ v + norm(x̄, x)^2: v >=〈z_b, x〉+ α_b ∀b∈B }
  @NLobjective(master_problem, Min, v + (stabilizer.μ * sum(d_i^2 for d_i in d)))
  # Bundle constraints ----- #
  for (z_b, α_b) in B
    @constraint(master_problem, v >= dot(z_b, (d .+ x̄)) + α_b)
  end
  status = optimize!(master_problem)

  x_ii = JuMP.value.(d) .+ x̄
  return x_ii, JuMP.value.(v), termination_status(master_problem)
end


# ----- ----- BUNDLE METHOD ----- ----- #
# ----------- ------------- ----------- #
function shrink!(B, n)
  while length(B) > n
    dequeue!(B)
  end
end

function bundle_method_cycle(x_zero::Array{Float64}, oracle::Function; 
                             m::Float64=1e-3, stabilization=nothing,
                             maxtime=-1, maxiter::Int=1000, ε::Float64=1e-3, 
                             f_star=nothing, ε_star=1e-7,
                             reset_bundle::Bool=false, max_bundle_size=50,
                             callback_array=[])
  stabilizer = if !isnothing(stabilization) stabilization else StabProximal() end
  
  # Initialization
  infostring = ""
  total_time = 0.
  z̄::Union{Array{Float64}, Nothing} = nothing
  B = Queue{Tuple{Array{Float64}, Float64}}()
  fB::Function = x -> maximum([ dot(z_b, x) - α_b for (z_b, α_b) in B])

  opt_reached = false
  if !isnothing(f_star) 
    ε = 1e-15
  end

  # Bundle initialization
  ∆ = -1
  x_i = x_zero
  x̄ = x_zero
  f_x̄, z_i = oracle(x_zero)
  α_i = f_x̄ - dot(z_i, x_i)
  z̄ = z_i
  enqueue!(B, (z_i, α_i))

  pb = ProgressBar(1:maxiter)
  for i in pb
    # Step && Bundle update ..... #
    (x_i, v, status), timed_res = @timed stabilizer.step(x̄, B, ε, stabilizer)

    f_xi, z_i = oracle(x_i)
    α_i = f_xi - dot(z_i, x_i)
    enqueue!(B, (z_i, α_i))

    # Armijo condition ..... #
    if v > f_x̄
      return x̄, f_x̄, "[PRECISION_ERROR: $status][v > f_x̄: $v > $f_x̄]"
    end
    SS::Bool = f_xi < f_x̄ + m * (v - f_x̄)

    # Callbacks ..... #
    if SS
      infostring = "" 
    end
    for c in callback_array
      cinfo = c(f_xi=f_xi, f_x̄=f_x̄, x_i=x_i, z_i=z_i, 
                x̄=x̄, z̄=z̄, α_i=α_i, SS=SS, B=B, it=i, v=v, time_elapsed=total_time)
      if cinfo isa String
        infostring = string(infostring, cinfo)
      end
    end

    # Everything else ..... #
    if (SS == true)
      # Stopping criteria
      ∆ = f_x̄ - v
  
      # Update 
      f_x̄ = f_xi 
      x̄ = x_i; z̄ = z_i;
      stabilizer.SS_update!(stabilizer)

      # Bundle update
      if max_bundle_size > 0
        shrink!(B, max_bundle_size)
      elseif reset_bundle
        shrink!(B, 1)
      end
      
      # Termination
      if !isnothing(f_star)
        rel_error = abs(f_x̄ - f_star) / f_star
        if rel_error <= ε_star
          return (x̄, f_x̄, "[OPTIMUM REACHED: $∆]")
        end
      end

      if ∆ < ε
        return (x̄, f_x̄, "[OPTIMUM REACHED: $∆][f_*: $f_x̄]")
      end
    else 
      stabilizer.NS_update!(stabilizer)
    end

    # Stats ..... #
    err::String = string(infostring, "[#B: $(length(B))]")
    if SS
      set_description(pb, "$err[SS]")
    else
      set_description(pb, "$err[NN]")
    end

    # Time update ..... #
    total_time += timed_res[1]
    if maxtime > 0 && total_time > maxtime
      return (x̄, f_x̄, "[MAX TIME REACHED][precision: $∆][f_*: $f_x̄]")
    end
  end 

  return (x̄, f_x̄, "[MAX ITERATION REACHED][precision: $∆][f_*: $f_x̄]")
end
end
