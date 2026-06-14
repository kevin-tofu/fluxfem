#!/usr/bin/env julia

include("../tutorials/neo_hookean_cantilever.jl")

using Printf

function json_escape(s::AbstractString)
    replace(s, "\\" => "\\\\", "\"" => "\\\"")
end

function write_json(io, results)
    println(io, "{")
    println(io, "  \"mode\": \"gridap_same_session\",")
    println(io, "  \"results\": [")
    for (i, r) in enumerate(results)
        println(io, "    {")
        println(io, "      \"mesh\": \"", json_escape(r["mesh"]), "\",")
        println(io, "      \"n_nodes\": ", r["n_nodes"], ",")
        println(io, "      \"load_steps\": ", r["load_steps"], ",")
        println(io, "      \"wall_time_s\": ", r["wall_time_s"], ",")
        println(io, "      \"assembly_time_s\": ", r["assembly_time_s"], ",")
        println(io, "      \"solve_time_s\": ", r["solve_time_s"], ",")
        println(io, "      \"warmup_solve_s\": ", r["warmup_solve_s"], ",")
        println(io, "      \"first_step_s\": ", r["first_step_s"], ",")
        println(io, "      \"remaining_steps_avg_s\": ", r["remaining_steps_avg_s"], ",")
        println(io, "      \"compile_only_proxy_s\": ", r["compile_only_proxy_s"], ",")
        println(io, "      \"wrapper_time_s\": ", r["wrapper_time_s"], ",")
        println(io, "      \"max_disp\": ", r["max_disp"])
        print(io, "    }")
        println(io, i < length(results) ? "," : "")
    end
    println(io, "  ]")
    println(io, "}")
end

function main(args)
    if isempty(args)
        error("usage: julia bench/gridap_warmrun.jl mesh1.msh mesh2.msh ... [--out result.json]")
    end

    meshes = String[]
    out_json = "result/bench/gridap_warmrun/results.json"
    nstep = 20
    warmup_solve = false
    i = 1
    while i <= length(args)
        if args[i] == "--out"
            i += 1
            out_json = args[i]
        elseif args[i] == "--nstep"
            i += 1
            nstep = parse(Int, args[i])
        elseif args[i] == "--warmup-solve"
            warmup_solve = true
        else
            push!(meshes, args[i])
        end
        i += 1
    end

    results = Any[]
    for mesh in meshes
        coords = nothing
        u = nothing
        times = nothing
        t_wrap = @elapsed begin
            coords, u, times = run_gridap(mesh; nstep=nstep, warmup_solve=warmup_solve, write_outputs=false)
        end
        compile_only_proxy = (
            isnan(times.warmup_solve) || isnan(times.remaining_avg)
            ? NaN
            : max(0.0, times.warmup_solve - times.remaining_avg)
        )
        push!(results, Dict(
            "mesh" => mesh,
            "n_nodes" => size(coords, 1),
            "load_steps" => nstep,
            "wall_time_s" => times.total,
            "assembly_time_s" => times.assembly,
            "solve_time_s" => times.solve,
            "warmup_solve_s" => times.warmup_solve,
            "first_step_s" => times.first_step,
            "remaining_steps_avg_s" => times.remaining_avg,
            "compile_only_proxy_s" => compile_only_proxy,
            "wrapper_time_s" => t_wrap,
            "max_disp" => maximum(norm.(eachrow(u))),
        ))
        @printf("mesh=%s total=%.3f asm=%.3f solve=%.3f warmup=%.3f compile_proxy=%.3f wrap=%.3f\n",
            mesh,
            results[end]["wall_time_s"],
            results[end]["assembly_time_s"],
            results[end]["solve_time_s"],
            results[end]["warmup_solve_s"],
            results[end]["compile_only_proxy_s"],
            results[end]["wrapper_time_s"],
        )
    end

    mkpath(dirname(out_json))
    open(out_json, "w") do io
        write_json(io, results)
    end
    println(out_json)
end

main(ARGS)
