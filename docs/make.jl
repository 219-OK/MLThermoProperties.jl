using Documenter
using DocumenterVitepress
using DocumenterCitations
using Literate
using MLThermoProperties, Clapeyron

## Generate tutorial markdown from Literate sources
Literate.markdown(
    joinpath(@__DIR__, "src", "tutorials", "px_diagram.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    documenter = true,
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    sitename = "MLThermoProperties.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/se-schmitt/MLThermoProperties.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Tutorials" => [
            "p-x Diagram with HANNA" => "tutorials/px_diagram.md",
        ],
        "Models" => "models.md",
        "References" => "references.md",
    ],
    plugins=[bib]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/se-schmitt/MLThermoProperties.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)