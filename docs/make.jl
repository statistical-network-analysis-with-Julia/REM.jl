using Documenter
using REM
using Networks

DocMeta.setdocmeta!(REM, :DocTestSetup, :(using REM); recursive=true)

makedocs(
    sitename = "REM.jl",
    modules = [REM],
    authors = "Simone Santoni",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Statistical-network-analysis-with-Julia.github.io/REM.jl",
        edit_link = "main",
    ),
    repo = Documenter.Remotes.GitHub("Statistical-network-analysis-with-Julia", "REM.jl"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Events and Data" => "guide/events.md",
            "Statistics" => "guide/statistics.md",
            "Model Estimation" => "guide/estimation.md",
            "Temporal Decay" => "guide/decay.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Statistics" => "api/statistics.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    # STRICT. Undefined bindings, bad cross-references, duplicate docs and
    # malformed markdown are build ERRORS, so they cannot silently accumulate
    # again (a docs build that passes while warning is one that will rot).
    #
    # `checkdocs = :exports` is the one deliberate exclusion: every *exported*
    # name must be documented, but internal machinery (materialized/private
    # types, `Base`/`Graphs` method extensions, inner constructors) need not be
    # -- filler docstrings for names a user never types are worse than none.
    warnonly = false,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Statistical-network-analysis-with-Julia/REM.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # serve dev docs at /stable until a release is tagged
        "dev" => "dev",
    ],
    push_preview = true,
)
