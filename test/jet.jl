using JET
using MultiWellPotentialGrossPitaevskii

# test_package analyzes the module and throws an error if it finds
# method errors, type instabilities, or other static analysis failures.
# target_defined_modules=true ensures it analyzes all modules defined in your package.
JET.test_package(
    "MultiWellPotentialGrossPitaevskii";
    target_defined_modules = true,
)

println("JET.jl static analysis passed successfully.")
