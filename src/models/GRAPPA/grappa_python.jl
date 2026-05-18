function _GRAPPA_error(args...; kwargs...)
    error("""
    To use GRAPPA, `PythonCall` needs to be installed and loaded! This can be done by:
        using Pkg; Pkg.add("PythonCall")
        using PythonCall
    """)
    return nothing
end

const _GRAPPA = Ref{Function}(_GRAPPA_error)

function PyGRAPPA(components; userlocations = String[], reference_state = nothing, verbose = false)
    _GRAPPA[](components; userlocations, reference_state, verbose)
end
