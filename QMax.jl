module QMax

using LinearAlgebra
using Statistics
using Debugger

export oracle

function oracle(inp)
  num = length(inp)
  l, i = findmax(inp.^2)
  k_d = zeros(num)
  k_d[i] = 1
  g = 2 * (inp.* k_d)
  return l, g
end
end
