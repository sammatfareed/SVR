using GLPK
using DataStructures: SortedDict
using DataStructures: DefaultDict
using Ipopt
using ProgressBars
using ArgParse
using JuMP
using LinearAlgebra
using Plots
using Debugger
using CSV
using SVR
using DataFrames
using Statistics
using Random

include("Utils.jl")
include("Callbacks.jl")
include("BundleMethod.jl")
include("JbSVR.jl")
include("QMax.jl")


seed = rand(1:10000)
println("SEED: $seed")
Random.seed!(seed);

function compare_rel_error(dataset, f_x_star; maxtime=10*60, ε_star=1e-6, svr_args...)
  X, Y = Utils.SVR_input_read_csv(dataset, '.')

  # Jbundle/SVR parameters ..... #
  maxit = 2000
  maxtime = 10 * 60

  # Configurations to test ..... #
  bundle_configurations = Dict(
    # "TR_FB" => Dict(:stabilization => JbSVR.StabTrust()),
    "TR_RS" => Dict(:stabilization => JbSVR.StabTrust(), :reset_bundle => true),
    "TR_MX_50" => Dict(:stabilization => JbSVR.StabTrust(), :max_bundle_size => 50),

    # "PR_FB" => Dict(:stabilization => JbSVR.StabProximal()),
    "PR_RS" => Dict(:stabilization => JbSVR.StabProximal(), :reset_bundle => true),
    "PR_MX_50" => Dict(:stabilization => JbSVR.StabProximal(), :max_bundle_size => 50))

  # Plot options ..... #
  histories = DefaultDict(Dict)
  plotconfig = DefaultDict(Dict)
  push!(plotconfig, "RelativeError" => Dict(:yaxis => :log, :xlabel => "Seconds", :title => "Relative Error"))
  push!(plotconfig, "Serious Steps" => Dict(:yaxis => :log, :xlabel => "Seconds", :title => "SS"))
  push!(plotconfig, "Convergence" => Dict(:yaxis => :log, :title => "∆ = f_x̄ .- v"))

  # Train & Prediction ..... #
  for (name, cf) in bundle_configurations
    svr_conf = JbSVR.SVR_conf(X, Y, JbSVR.RBF_kernel; svr_args...)

    # Callbacks
    println("# ----- ----- #\n$name")
    hist = DefaultDict(SortedDict)
    callbacks = Function[ Callbacks.newCallback(Callbacks.SeriousStepCallback, hist["Serious Steps"]),
                          Callbacks.newCallback(Callbacks.Termination, hist["Convergence"]),
                          Callbacks.newCallback(Callbacks.RelativeError, hist["RelativeError"], f_x_star=f_x_star)]

    # Train & Prediction
    svr, (infolog, time_elapsed) = JbSVR.train(svr_conf; callback_array=callbacks, 
                                               maxiter=maxit, maxtime=maxtime, 
                                               f_star=f_x_star, ε_star=ε_star, cf...)

    for (k, values) in hist
      histories[k][name] = values
    end
  end

  # Final report ..... #
  Callbacks.multiplotCallbacks(histories, plotconfig)
end


# ----------- ----------- ----------- ---------- #
# ----------- ----------- ----------- ---------- #
function test_known_svr(dataset, tr_split)
  x, y = Utils.SVR_input_read_csv(dataset, '.')
  tr_idx = if isa(tr_split, Integer) tr_split else Int(round(tr_split * length(x))) end 
  x_tr = x[1:tr_idx, :]
  y_tr = y[1:tr_idx, :]
  x_ts = x[tr_idx:end, :]
  y_ts = y[tr_idx:end, :] 

  better_svr = SVR.train(y_tr, permutedims(x_tr); kernel_type=SVR.RBF)
  ŷ_ts = SVR.predict(better_svr, permutedims(x_ts));
  l2_ts = mean((y_ts .- ŷ_ts).^2)
  SVR.freemodel(better_svr)

  println("\n--------------------")
  println("Distance from true ground: $l2_ts")
  println("--------------------")
end


function train_test_svr(dataset, tr_split; show_plots=true, precision=1e-3, svr_args...)
  x, y = Utils.SVR_input_read_csv(dataset, '.')
  tr_idx = if isa(tr_split, Integer) tr_split else Int(round(tr_split * length(x))) end 
  x_tr = x[1:tr_idx, :]
  y_tr = y[1:tr_idx, :]
  x_ts = x[tr_idx:end, :]
  y_ts = y[tr_idx:end, :] 

  svr_conf = JbSVR.SVR_conf(x_tr, y_tr, JbSVR.RBF_kernel; svr_args...)

  # Callbacks ..... #
  history = DefaultDict(SortedDict)
  callbacks = Function[ Callbacks.newCallback(Callbacks.SeriousStepCallback, history["Serieous Steps"]),
                        Callbacks.newCallback(Callbacks.Termination, history["Convergence"]),
                        Callbacks.newCallback(Callbacks.StabilityCenterCallback, history["StabilityCenter"]),
                        Callbacks.newCallback(Callbacks.L2Callback, history["L2train"], X=x_tr, Y=y_tr, svr=svr_conf, label="trainL2"),
                        Callbacks.newCallback(Callbacks.L2Callback, history["L2test"], X=x_ts, Y=y_ts, svr=svr_conf, label="testL2")]

  # Train & Prediction ..... #
  svr, (infolog, time_elapsed) = JbSVR.train(svr_conf, callback_array=callbacks, 
                                             ε=precision, maxiter=2000)

  # Final report ..... #
  plotconfig = DefaultDict(Dict)
  push!(plotconfig, "StabilityCenter" => Dict(:yaxis => :log, 
                                              :title => "StabilityCenter function value at each SS (log scale)"),
                    "RelativeError"   => Dict(:yaxis => :log, 
                                              :title => "Relative Error"),
                    "L2train"         => Dict(:yaxis => :log, 
                                              :title => "L2 train ground truth (log scale)"),
                    "Convergence"     => Dict(:yaxis => :log, 
                                              :label => "∆ = f(x̄) .- v",
                                              :title => "Convergence"),
                    "L2test"          => Dict(:yaxis => :log, 
                                              :title => "L2 test (log scale)"))
  if (show_plots)
    Callbacks.plotCallbacks(history, plotconfig)
  end
  println("\n [Total time: $time_elapsed]$infolog \n")
end
