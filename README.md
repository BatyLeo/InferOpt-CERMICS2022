# InferOpt presentation and warcraft tutorial

You need to have Julia 1.7 or higher installed in order to run the notebook tutorial.

## Get started

1) Before running the notebook, please unzip the dataset in the same folder as the repository.
(Note: you can also find the dataset [here](http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip) for download)

2) Open a Julia REPL in the folder of this repo, and run:

Install Pluto if not already installed:
```julia
using Pkg
Pkg.add("Pluto")
```

Run Pluto:
```julia
using Pluto
Pluto.run()
```

A browser window opens, select the `inferopt.jl` notebook file.
