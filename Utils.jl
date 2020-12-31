module Utils

using LinearAlgebra
using Plots
using Debugger
using CSV
using DataFrames

export SVR_input_read_csv 


# ----- ----- INPUT PIPELINE ------ ---- #
# ----------- -------------- ----------- #
function SVR_input_read_csv(input_filename::String, separator::Char=',')
  csv = DataFrame!(CSV.File(input_filename, decimal=separator))
  cols = size(csv, 2)
  x = convert(Matrix, csv[1:cols-1])
  y = csv[cols]
  return (x, y)
end

function ScatterTruePred(x, y, ŷ, fname)
  gr()
  ground_truth = scatter(x, y, label="ground truth")
  pred = scatter!(x, ŷ, label="prediction")
  savefig(fname)
end

end
