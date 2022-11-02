# InferOpt presentation and warcraft tutorial

An online version of the notebook can be found [here](https://batyleo.github.io/InferOpt-CERMICS2022/inferopt.html).

## Get started

1) Before running the notebook, please download the [dataset](http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip) and place it in the 
`/data` folder of this repo.

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
