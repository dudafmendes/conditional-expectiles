using CairoMakie

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = get(
    ENV,
    "MC_BOXPLOT_STATS",
    joinpath(ROOT, "results", "mc", "figures", "mc_interval_length_ratio_boxstats.csv"),
)
const OUTPUT_DIR = get(
    ENV,
    "MC_FIGURE_OUTPUT_DIR",
    joinpath(ROOT, "results", "mc", "figures"),
)

function read_box_statistics(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Boxplot-statistics file is empty: $(path)")
    header = split(first(lines), ',')
    column = Dict(name => index for (index, name) in enumerate(header))
    required = [
        "level", "benchmark", "model", "distribution", "persistence", "n",
        "valid", "lower_whisker", "q1", "median", "q3", "upper_whisker",
    ]
    all(haskey(column, name) for name in required) || error("Unexpected boxplot-statistics schema.")
    return [
        (
            level=parse(Float64, values[column["level"]]),
            benchmark=values[column["benchmark"]],
            model=values[column["model"]],
            distribution=values[column["distribution"]],
            persistence=values[column["persistence"]],
            n=parse(Int, values[column["n"]]),
            valid=parse(Int, values[column["valid"]]),
            lower=parse(Float64, values[column["lower_whisker"]]),
            q1=parse(Float64, values[column["q1"]]),
            median=parse(Float64, values[column["median"]]),
            q3=parse(Float64, values[column["q3"]]),
            upper=parse(Float64, values[column["upper_whisker"]]),
        )
        for values in (split(line, ',') for line in Iterators.drop(lines, 1))
    ]
end

function draw_box!(axis, x, row, color; width=0.15)
    poly!(
        axis,
        Rect(x - width / 2, row.q1, width, row.q3 - row.q1);
        color=(color, 0.86), strokecolor="#404040", strokewidth=0.8,
    )
    lines!(axis, [x, x], [row.lower, row.q1]; color="#555555", linewidth=0.8)
    lines!(axis, [x, x], [row.q3, row.upper]; color="#555555", linewidth=0.8)
    lines!(axis, [x - width / 3, x + width / 3], [row.lower, row.lower]; color="#555555", linewidth=0.8)
    lines!(axis, [x - width / 3, x + width / 3], [row.upper, row.upper]; color="#555555", linewidth=0.8)
    lines!(axis, [x - width / 2, x + width / 2], [row.median, row.median]; color=:black, linewidth=1.15)
end

function make_figure(rows, level::Float64, token::String)
    distributions = ["normal", "t8", "t4"]
    distribution_titles = Dict("normal" => "Normal", "t8" => "Student t₈", "t4" => "Student t₄")
    benchmarks = ["VaR", "ES"]
    sample_sizes = [500, 1000, 2500, 5000]
    structures = [
        ("GARCH", "low"), ("GARCH", "high"),
        ("GJR-GARCH", "low"), ("GJR-GARCH", "high"),
    ]
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    offsets = [-0.27, -0.09, 0.09, 0.27]

    figure = Figure(size=(1450, 830), fontsize=17, backgroundcolor=:white)
    axes = Matrix{Axis}(undef, 2, 3)
    for (row_index, benchmark) in enumerate(benchmarks), (column_index, distribution) in enumerate(distributions)
        percent = round(Int, 100 * level)
        ylabel = column_index == 1 ? "Interval length relative to $(percent)% $(benchmark)" : ""
        xlabel = row_index == 2 ? "Sample size" : ""
        axis = Axis(
            figure[row_index + 1, column_index];
            title=distribution_titles[distribution],
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=(1:4, ["500", "1,000", "2,500", "5,000"]),
            xgridvisible=false,
            ygridvisible=true,
            ygridcolor=(:gray, 0.18),
            topspinevisible=false,
            rightspinevisible=false,
        )
        axes[row_index, column_index] = axis
        hlines!(axis, [1.0]; color=(:black, 0.72), linestyle=:dash, linewidth=1.0)

        for (structure_index, (model, persistence)) in enumerate(structures), (n_index, n) in enumerate(sample_sizes)
            matches = filter(
                row -> isapprox(row.level, level; atol=1e-12) &&
                       row.benchmark == benchmark && row.distribution == distribution &&
                       row.model == model && row.persistence == persistence && row.n == n,
                rows,
            )
            length(matches) == 1 || error("Missing or duplicate boxplot cell.")
            draw_box!(axis, n_index + offsets[structure_index], only(matches), colors[structure_index])
        end
        xlims!(axis, 0.55, 4.45)
    end

    linkyaxes!(axes[1, 1], axes[1, 2], axes[1, 3])
    linkyaxes!(axes[2, 1], axes[2, 2], axes[2, 3])
    for axis in axes[:, 2:3]
        hideydecorations!(axis; grid=false)
    end

    labels = [
        "GARCH, low persistence", "GARCH, high persistence",
        "GJR-GARCH, low persistence", "GJR-GARCH, high persistence",
    ]
    elements = [PolyElement(color=(color, 0.86), strokecolor="#404040") for color in colors]
    Legend(figure[1, 1:3], elements, labels; orientation=:horizontal, framevisible=false, labelsize=16)
    rowgap!(figure.layout, 12)
    colgap!(figure.layout, 18)

    mkpath(OUTPUT_DIR)
    output_pdf = joinpath(OUTPUT_DIR, "mc_interval_length_ratios_$(token).pdf")
    output_png = joinpath(OUTPUT_DIR, "mc_interval_length_ratios_$(token).png")
    save(output_pdf, figure)
    save(output_png, figure; px_per_unit=2)
    println("Generated $(output_pdf)")
    println("Generated $(output_png)")
end

function main()
    CairoMakie.activate!()
    rows = read_box_statistics(INPUT)
    length(rows) == 192 || error("Expected 192 boxplot cells, found $(length(rows)).")
    make_figure(rows, 0.01, "0p01")
    make_figure(rows, 0.05, "0p05")
end

main()
