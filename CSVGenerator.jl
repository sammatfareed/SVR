using ArgParse
using CSV
using DataFrames


# ----- ----- CONSTANTS ----- ----- #
# ----- ----- --------- ----------- #
PNAME = "problem2csv"
PDESCR = 

PVERSION = "0.1"

GAUSS3 = "Gauss3" 
CARBON_NANOTUBES = "CarbonNanotubes"

function sigmoid(x) 
  return 1.0 ./ (1.0 .+ exp.(-x))
end


function ComputeGauss3(x::Array{Float64}, betaindex::Int=1)
  b1 = [9.4900000000E+01, 9.6000000000E+01, 9.8940368970E+01]
  b2 = [9.0000000000E-03, 9.6000000000E-03, 1.0945879335E-02]
  b3 = [9.0100000000E+01, 8.0000000000E+01, 1.0069553078E+02]
  b4 = [1.1300000000E+02, 1.1000000000E+02, 1.1163619459E+02]
  b5 = [2.0000000000E+01, 2.5000000000E+01, 2.3300500029E+01]
  b6 = [7.3800000000E+01, 7.4000000000E+01, 7.3705031418E+01]
  b7 = [1.4000000000E+02, 1.3900000000E+02, 1.4776164251E+02]
  b8 = [2.0000000000E+01, 2.5000000000E+01, 1.9668221230E+01]
  b = hcat(b1, b2, b3, b4, b5, b6, b7, b8)
  
  b1, b2, b3, b4, b5, b6, b7, b8 = b[betaindex, :]
  res = b1.*exp.( -b2.*x )
  res .+= b3.*exp.( -(x.-b4).^2 ./ b5.^2 )
  res .+= b6.*exp.( -(x.-b7).^2 ./ b8.^2 )
  res .+= ℯ

  m = minimum(res)
  M = maximum(res)
  return (res .- m) ./ (M - m)
end


# ----- ----- MAIN ------ ---- #
# ----------- ----- ----------- #
function parse_commandline()
  println("$PNAME ($PVERSION)")
  s = ArgParseSettings(prog=PNAME, description=PDESCR, version=PVERSION)
  @add_arg_table! s begin
    "--address"
      help = "The problem to be solved: [Gauss3, CarbonNanotubes]"
    "--outpath"
      help = "Output path"
      default = "/datasets"
    "--inpath"
      help = "Input path"
      default = "/data"
    "--train"
      help = "training set size"
      arg_type = Int
      default = 1000
    "--test"
      help = "test set size"
      arg_type = Int
      default = 200
    "--train_split"
      help = "Train set split size"
      arg_type = Int
      default = 70
  end

  return parse_args(s)
end


function main()
  args = parse_commandline()
  if args["address"] == GAUSS3
    x_train = rand(0:300., args["train"]) .+ rand(args["train"])
    y_train = ComputeGauss3(x_train)
    train = DataFrame(x=x_train, y=y_train)
    train_outpath = string(args["outpath"], "/gauss3_train.csv")
    CSV.write(train_outpath, train)

    x_test = rand(0:300., args["test"]) .+ rand(args["test"])
    y_test = ComputeGauss3(x_test)
    test = DataFrame(x=x_test, y=y_test)
    test_outpath = string(args["outpath"], "/gauss3_test.csv")
    CSV.write(test_outpath, test)
  elseif args["address"] == CARBON_NANOTUBES
    xcols = ["Chiral indice n", "Chiral indice m", 
            "Initial atomic coordinate u", "Initial atomic coordinate v", "Initial atomic coordinate w"]
    ycols = ["Calculated atomic coordinates u'"
             "Calculated atomic coordinates v'"
             "Calculated atomic coordinates w'"]
    csv = DataFrame!(CSV.File(args["inpath"] * "/carbon_nanotubes.csv"))

    dataset_row_number = size(csv, 1)
    dataset_column_number = size(csv, 2)
    train_end = floor(Int, dataset_row_number * args["train_split"] / 100)
    test_start = train_end + 1

    train = csv[1:train_end, 1:dataset_column_number-2]
    train_outpath = string(args["outpath"], "/carbon_nanotubes_train.csv")
    CSV.write(train_outpath, train, delim=";")

    test = csv[test_start:dataset_row_number, 1:dataset_column_number-2]
    test_outpath = string(args["outpath"], "/carbon_nanotubes_test.csv")
    CSV.write(test_outpath, test, delim=";")
  else 
    throw(ArgumentError("Unkwown probelm"))
  end
end

main()
