module Callbacks

using Dates
using LinearAlgebra
using Statistics
using Debugger
using Plots

import ..Utils

include("JbSVR.jl")
using .JbSVR

export plotCallbacks, DistanceFromOptimum, DistanceBetweenSS, 
       ConvergenceCallBack, SeriousStepCallback, newCallback


function SeriousStepCallback(; DictCallback, SS::Bool, B, it::Int, kwargs...)
  currentSS = if (it == 1) 0 else maximum(DictCallback)[2] end
  if SS
    currentSS += 1
  end

  push!(DictCallback, it => currentSS)
end


function StabilityCenterCallback(; DictCallback, f_x̄::Float64, it::Int, kwargs...)
  push!(DictCallback, it => f_x̄)
end

function Termination(; DictCallback, it::Int, f_x̄::Float64, v::Float64, SS::Bool, kwargs...)
  if SS
    ∆ = (f_x̄ .- v)
    push!(DictCallback, it => ∆)
    return "[∆: $∆]"
  end
end

function ConvergenceCallBack(; DictCallback, f_xi::Float64, f_x̄::Float64, it::Int, kwargs...)
    kwargs = Dict(kwargs)
    if haskey(kwargs, :optimum)
        optimum = kwargs[:optimum]
        f_xi = f_xi - optimum
        f_x̄ = f_x̄ - optimum
    end
    push!(DictCallback[it], "f_xi"=> f_xi, "f_x̄" => f_x̄)
end

function RelativeError(; DictCallback, SS::Bool, it::Int, f_x̄::Float64, time_elapsed::Float64, f_x_star::Float64, kwargs...)
    if SS
      rel_error = abs(f_x̄ - f_x_star) / f_x_star
      push!(DictCallback, it => rel_error)
      return "[rel_err: $rel_error]"
    end
end

function DistanceBetweenSS(; DictCallback, x_i::Array{Float64}, x̄::Array{Float64}, it::Int, SS::Bool, kwargs...)
    kwargs = Dict(kwargs)
    if SS
        if isempty(DictCallback)
            prev_xi = fill(0, size(x_i))
            prev_x̄ = fill(0, size(x̄))
        else
            last_idx = maximum(keys(DictCallback))
            prev_xi = DictCallback[last_idx]["x_i"]
            prev_x̄ = DictCallback[last_idx]["x̄"]
            #To avoid storing x_i and x̄ for every iteration, we keep them just for the last one
            DictCallback[last_idx] = Dict("l2_xi"=> DictCallback[last_idx]["l2_xi"], "l2_x̄" => DictCallback[last_idx]["l2_x̄"]) 
        end
        l2_xi = norm(x_i.-prev_xi).^2
        l2_x̄ = norm(x̄.-prev_x̄).^2
        push!(DictCallback[it], "x̄" => x̄, "x_i" => x_i, "l2_xi"=> l2_xi, "l2_x̄" => l2_x̄)
    end
end


function DistanceFromOptimum(; DictCallback, x_i::Array{Float64}, x̄::Array{Float64}, it::Int, SS::Bool, kwargs...)
    kwargs = Dict(kwargs)
    if SS
        if haskey(kwargs, :optimum) optimum = kwargs[:optimum] else optimum = 0 end
        l2_xi = norm(x_i.-optimum).^2
        l2_x̄ = norm(x̄.-optimum).^2
        push!(DictCallback[it], "l2_xi"=> l2_xi, "l2_x̄" => l2_x̄)
    end
end


function L2Callback(; gramian, Y, svr, DictCallback, it::Int, x̄::Array{Float64}, SS::Bool, label="", kwargs...)
  if !SS
    return
  end
  
  # Preparing svr 
  tmp_svr = deepcopy(svr)
  tmp_svr.β = x̄[1:svr.m]
  tmp_svr.b = x̄[svr.m+1]

  # Predicting 
  ŷ = JbSVR.predict(tmp_svr, gramian=gramian)
  l2_y = (Y .- ŷ).^2
  l2_y_m = mean(l2_y)

  # Updating status 
  push!(DictCallback, it => l2_y_m)
  return "[$label: $(string(l2_y_m)[1:7])]"
end

function newCallback(callback, history_entry; svr=nothing, X=nothing, callargs...)
  args = Dict{Symbol, Any}(callargs)

  if !isnothing(X) && !isnothing(svr)
    gramian_cache = svr.gramian(X)
    args[:gramian] = gramian_cache
    args[:svr] = svr
  end

  return (; kwargs...) -> callback(; DictCallback=history_entry, kwargs..., args...)
end


# ----- ----- PLOTS ------ ---- #
# ----------- ----- ----------- #
function multiplotCallbacks(histories, config)
  if !isdir("./plots")
    mkdir("plots")
  end
  datestr = string(Dates.DateTime(Dates.now()))

  colors = [:red, :green, :orange, :blue, :brown, :black] # range(colorant"green", stop=colorant"red", length=n_colors)

  for (idx, (key, cf)) in enumerate(histories)
    gr()
    pl = plot(size=(800, 500))

    for (idx, (cf_name, vl)) in enumerate(cf)
      x = collect(keys(vl))
      y = collect(values(vl))
      y[y.<=0] .= 1e-15

      markervery = Int(round(length(x) * .075))
      xlabel = if haskey(config[key], :xaxis) config[key][:xaxis] else "Iteration" end
      plot!(x, y; xaxis=xlabel, label="", color=colors[idx], config[key]...)

      scatter!(x[1:markervery:end], y[1:markervery:end], color=colors[idx], label=cf_name, 
               marker=:auto, markerstrokewidth=0., legend=:outertopright) 
    end
    display(pl)
    # savefig("./plots/$(datestr)__$(key).png")
  end   
end

function plotCallbacks(history, config)
  if !isdir("./plots")
    mkdir("plots")
  end
  datestr = string(Dates.DateTime(Dates.now()))

  for (key, dict) in history
    gr()
    x = collect(keys(dict))
    y = collect(values(dict))
    
    pl = plot(collect(x), collect(y), xaxis="Iteration", title=key; config[key]...)
    display(pl)
    # savefig("./plots/$(datestr)__$(key).png")
  end   
end
end
