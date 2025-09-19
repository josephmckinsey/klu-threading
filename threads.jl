using SparseArrays
using KLU
import Base.Threads: @spawn, nthreads
using Random
using Serialization


A = Serialization.deserialize("ABA_matrix.jls")

function test_klu(A)
    klu_fact = KLU.klu(A)
    for _ in 1:30000
        KLU.solve(klu_fact, randn(size(A, 1)))
    end
end


function threaded_klu_solve(A, n_repeats=5)
    for _ in 1:n_repeats
        @spawn test_klu(A)
    end
end


println("Solving Ax = b in parallel with $(nthreads()) threads...")
threaded_klu_solve(A, 50)
