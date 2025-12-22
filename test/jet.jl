using CliqueTrees
using JET
using JuliaInterpreter: JuliaInterpreter

@static if v"1.9" <= VERSION < v"1.10"
    using JET: ConcreteInterpreter, Frame
    using JET: ismoduleusage, to_simple_module_usages, collect_toplevel_signature!

    #=
    Apply a patch to avoid getting errors in the combination of JET.jl v0.8.21 and JuliaInterpreter.jl v0.9.36:
    """
    Test threw exception
    Expression: (JET.report_package)(TensorCrossInterpolation; toplevel_logger = nothing, target_defined_modules = true)
    MethodError: step_expr!(::Any, ::Any, ::Any, ::Bool) is ambiguous.

    Candidates:
        step_expr!(interp::JET.ConcreteInterpreter, frame::JuliaInterpreter.Frame, node, istoplevel::Bool)
        @ JET ~/.julia/packages/JET/lopE4/src/toplevel/virtualprocess.jl:1202
        step_expr!(recurse, frame::JuliaInterpreter.Frame, node, istoplevel::Bool)
        @ JuliaInterpreter ~/.julia/packages/JuliaInterpreter/cxlKp/src/interpret.jl:457
    To resolve the ambiguity, try making one of the methods more specific, or adding a new method more specific than any of the existing applicable methods.
    """
    =#
    function JuliaInterpreter.step_expr!(
            interp::ConcreteInterpreter, frame::Frame, @nospecialize(node), istoplevel::Bool
        )
        @assert istoplevel "JET.ConcreteInterpreter can only work for top-level code"

        if ismoduleusage(node)
            for ex in to_simple_module_usages(node)
                interp.usemodule_with_err_handling(interp.context, ex)
            end
            return frame.pc += 1
        end
        # the original implementation:
        # res = @invoke JuliaInterpreter.step_expr!(interp::Any, frame::Any, node::Any, true::Bool)
        # our patch:
        res = @invoke JuliaInterpreter.step_expr!(
            interp::Any, frame::Frame, node::Any, true::Bool
        )

        interp.config.analyze_from_definitions &&
            collect_toplevel_signature!(interp, frame, node)
        return res
    end
end

if v"1.9" <= VERSION
    #test_package(CliqueTrees; target_modules = (CliqueTrees,))
end
