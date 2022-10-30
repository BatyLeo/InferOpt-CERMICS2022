### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6160785a-d455-40b4-ab74-e61c46e31537
# ╠═╡ show_logs = false
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add(Pkg.PackageSpec(name="Colors"))
	Pkg.add(Pkg.PackageSpec(name="CSV"))
	Pkg.add(Pkg.PackageSpec(name="Flux"))
	Pkg.add(Pkg.PackageSpec(name="ForwardDiff"))
	Pkg.add(Pkg.PackageSpec(name="GZip"))
	Pkg.add(Pkg.PackageSpec(name="Graphs"))
	Pkg.add(Pkg.PackageSpec(url="https://github.com/gdalle/GridGraphs.jl.git"))
	Pkg.add(Pkg.PackageSpec(name="Images"))
	Pkg.add(Pkg.PackageSpec(name="InferOpt"))
	Pkg.add(Pkg.PackageSpec(name="JSON"))
	Pkg.add(Pkg.PackageSpec(name="LaTeXStrings"))
	Pkg.add(Pkg.PackageSpec(name="LinearAlgebra"))
	Pkg.add(Pkg.PackageSpec(name="Metalhead"))
	Pkg.add(Pkg.PackageSpec(name="Markdown"))
	Pkg.add(Pkg.PackageSpec(name="NPZ"))
	Pkg.add(Pkg.PackageSpec(name="Plots"))
	Pkg.add(Pkg.PackageSpec(name="PlutoUI"))
	Pkg.add(Pkg.PackageSpec(name="ProgressLogging"))
	Pkg.add(Pkg.PackageSpec(name="Random"))
	Pkg.add(Pkg.PackageSpec(name="Statistics"))
	Pkg.add(Pkg.PackageSpec(name="Tables"))
	Pkg.add(Pkg.PackageSpec(name="Tar"))
	#Pkg.add(Pkg.PackageSpec(name="TikzPictures"))
	Pkg.add(Pkg.PackageSpec(name="UnicodePlots"))

	using Colors
	using CSV
	using Flux
	using ForwardDiff
	using GZip
	using Graphs
	using GridGraphs
	using Images
	using InferOpt
	using JSON
	using LaTeXStrings
	using LinearAlgebra
	using Markdown: MD, Admonition, Code
	using Metalhead
	using NPZ
	using Plots
	using ProgressLogging
	using Random
	using Statistics
	using Tables
	using Tar
	#using TikzPictures
	using PlutoUI
	using UnicodePlots
	Random.seed!(63)
end;

# ╔═╡ e279878d-9c8d-47c8-9453-3aee1118818b
md"""
**Utilities (hidden)**
"""

# ╔═╡ 8b7876e4-2f28-42f8-87a1-459b665cff30
md"""
Imports
"""

# ╔═╡ a0d14396-cb6a-4f35-977a-cf3b63b44d9e
md"""
TOC
"""

# ╔═╡ b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
TableOfContents(depth=2)

# ╔═╡ 2182d4d2-6506-4fd6-936f-0e7c30d73851
html"""
<script>
    const calculate_slide_positions = (/** @type {Event} */ e) => {
        const notebook_node = /** @type {HTMLElement?} */ (e.target)?.closest("pluto-editor")?.querySelector("pluto-notebook")
		console.log(e.target)
        if (!notebook_node) return []
        const height = window.innerHeight
        const headers = Array.from(notebook_node.querySelectorAll("pluto-output h1, pluto-output h2"))
        const pos = headers.map((el) => el.getBoundingClientRect())
        const edges = pos.map((rect) => rect.top + window.pageYOffset)
        edges.push(notebook_node.getBoundingClientRect().bottom + window.pageYOffset)
        const scrollPositions = headers.map((el, i) => {
            if (el.tagName == "H1") {
                // center vertically
                const slideHeight = edges[i + 1] - edges[i] - height
                return edges[i] - Math.max(0, (height - slideHeight) / 2)
            } else {
                // align to top
                return edges[i] - 20
            }
        })
        return scrollPositions
    }
    const go_previous_slide = (/** @type {Event} */ e) => {
        const positions = calculate_slide_positions(e)
        const pos = positions.reverse().find((y) => y < window.pageYOffset - 10)
        if (pos) window.scrollTo(window.pageXOffset, pos)
    }
    const go_next_slide = (/** @type {Event} */ e) => {
        const positions = calculate_slide_positions(e)
        const pos = positions.find((y) => y - 10 > window.pageYOffset)
        if (pos) window.scrollTo(window.pageXOffset, pos)
    }
	const left_button = document.querySelector(".changeslide.prev")
	const right_button = document.querySelector(".changeslide.next")
	left_button.addEventListener("click", go_previous_slide)
	right_button.addEventListener("click", go_next_slide)
</script>
"""

# ╔═╡ 1f0c5b88-f903-4a67-9581-b3a07c504d5c
md"""
Two columns
"""

# ╔═╡ a8d0f8be-01a8-4a2a-84e3-ca16e7ef5203
begin
	struct TwoColumn{L,R}
	    left::L
	    right::R
	end

	function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
	    write(io, """<div style="display: flex;"><div style="flex: 50%;">""")
	    show(io, mime, tc.left)
	    write(io, """</div><div style="flex: 50%;">""")
	    show(io, mime, tc.right)
	    write(io, """</div></div>""")
	end
end

# ╔═╡ 86735dcf-de5b-4f32-8bf9-501e006f58d5
begin
	info(text; title="Info") = MD(Admonition("info", title, [text]))
	tip(text; title="Tip") = MD(Admonition("tip", title, [text]))
	warning(text; title="Warning") = MD(Admonition("warning", title, [text]))
	danger(text; title="Danger") = MD(Admonition("danger", title, [text]))
	hint(text; title="Hint") = MD(Admonition("hint", title, [text]))
	not_defined(var) = warning(md"You must give a value to $(Code(string(var)))."; title="Undefined variable")
	keep_working() = info(md"You're almost there."; title="Keep working!")
	correct() = tip(md"Well done."; title="Correct!")
end;

# ╔═╡ f4800c47-0f98-4ec7-85e6-5c2a19f784f5
md"""
Polytope animation
"""

# ╔═╡ 3e7077cd-4a14-4971-801d-9b9eadd59624
function get_angle(v)
	v = v ./ norm(v)
	if v[2] >= 0
		return acos(v[1])
	else
		return π + acos(-v[1])
	end
end

# ╔═╡ 14af0338-554a-4f71-a290-3b4f16cc6af5
md"""
# InferOpt.jl: combinatorial optimization in machine learning pipelines

**[Guillaume Dalle](https://gdalle.github.io/), [Léo Baty](https://batyleo.github.io/), [Louis Bouvier](https://louisbouvier.github.io/) & [Axel Parmentier](https://cermics.enpc.fr/~parmenta/)**

CERMICS, École des Ponts

Notebook access: [https://github.com/BatyLeo/InferOpt-CERMICS2022](https://github.com/BatyLeo/InferOpt-CERMICS2022)
"""

# ╔═╡ a6cb796f-32bf-4dbd-a9d4-61b454dad548
begin
	ptetes = load("./images/tetes.png")
end

# ╔═╡ 83a3efe5-d3ed-4f00-b90f-b065ca5ac06a
html"<button onclick='present()'>Toggle presentation mode</button>"

# ╔═╡ f13bf21c-33db-4620-add8-bfd82f493d7c
md"""
# 1. What is it for ?
"""

# ╔═╡ f99d6992-dc3e-41d1-8922-4958886dade2
md"""

**Points of view**: 
- Enrich learning pipelines with combinatorial algorithms.
- Enhance combinatorial algorithms with learning pipelines.

```math
\xrightarrow[\text{instance}]{\text{Initial}}
\fbox{ML predictor}
\xrightarrow[\text{instance}]{\text{Encoded}}
\fbox{CO algorithm}
\xrightarrow[\text{solution}]{\text{Candidate}}
\text{Loss}
```

**Challenge:** Differentiating through CO algorithms.

**Two learning settings:**
- Learning by imitation.
- Learning by experience.
"""

# ╔═╡ 0d20da65-1e53-4b6e-b302-28243c94fb4c
md"""
## Many possible applications

- Shortest paths on Warcraft maps
- Stochastic Vehicle Scheduling
- Two-stage Minimum Spanning Tree
- Single-machine scheduling
- etc.
"""

# ╔═╡ 3277f951-dd62-4fd0-be19-82c4572b6361
md"""
## Shortest paths on Warcraft maps

Source: [Vlastelica et al. (2020)](https://openreview.net/forum?id=BkevoJSYPB)

**Dataset**: Each point contains

 - an image of a map, where each terrain type has a specific cost
 - a shortest path from the top-left to the bottom-right corners.

**Goal**: Learn to recognize terrain types, in order to find shortest paths from new map images.
"""

# ╔═╡ d0c4164c-328d-46ab-bd97-81f989a9e081
begin
	mapterrain = plot(load("./images/Warcraft/map.png"), title = "Terrain map")
	labelpath = plot(load("./images/Warcraft/path.png"), title = "Label shortest path")
	plot(mapterrain, labelpath, layout = (1,2), ticks = nothing, border = nothing, size = (800, 400))
end

# ╔═╡ 103dc4a8-e1a9-425a-807a-7c5a4446bea7
md"""
## ML-CO pipeline
"""

# ╔═╡ 5cd285ca-9f45-42bf-ad8a-937d0e47581b
begin
	warcraftpipeline = load("./images/Warcraft/warcraft_pipeline.png")
end

# ╔═╡ 3a42e362-b826-4bb9-a063-bbdb71f0b2b4
md"""
## Test set prediction (1)

We can compare the predicted costs $\theta = \varphi_w(x)$ and the true costs on samples from the test set.
"""

# ╔═╡ 64c69987-6416-488b-8b7f-55d48771184d
begin 
	true_costs = plot(load("./images/Warcraft/costs.png"), title = "True costs")
	computed_costs = plot(load("./images/Warcraft/computed_costs.png"), title = "Computed costs")
	plot(mapterrain, true_costs, computed_costs, layout = (1,3), ticks = nothing, border = nothing, size = (800, 300))
end

# ╔═╡ aac53339-1091-4308-8d61-5ab4d3334c26
md"""
## Test set prediction (2)
"""

# ╔═╡ 16963349-5467-4019-be3d-d1b5375cf90e
begin 
	computed_path = plot(load("./images/Warcraft/computed_path.png"), title = "Computed shortest path")
	plot(mapterrain, labelpath, computed_path, layout = (1,3), ticks = nothing, border = nothing, size = (800, 300))
end

# ╔═╡ 1574b408-cf50-4c57-9fb8-eaa22bb3ece1
md"""
## Results

!!! todo
	TODO
"""

# ╔═╡ 400867ad-11e6-411b-8b1f-c64685630fdc
md"""
## Stochastic Vehicle Scheduling
"""

# ╔═╡ dda2e192-36fa-418b-8f4e-4cb3afd69360
begin
	vsp_instance = load("./images/VSP/vsp_instance.pdf")
end

# ╔═╡ 5764e92d-7fc4-4b62-a709-79979fb4b90c
md"""
- Set of **delivery tasks** ``t\in T``.
- Predefined time and duration for each task.
- **Decision variables**: schedule of vehicle tours.
- **Constraint**: all tasks must be fulfilled by exactly one vehicle.
"""

# ╔═╡ 799fcc82-6a25-47a7-8b52-32a754d4e875
md"""
## Why "stochastic"?

- Once vehicle tours are set, random delays occur:
  - Set of scenarios ``S``
  - Intrinsic delay of task ``t``: ``\varepsilon_t^s``
  - Slack tasks between tasks ``u`` and ``t``: ``\Delta_{u, t}^s``
  - Delay propagation equation: ``d_t^s = \varepsilon_t^s + \max(d_u^s - \Delta_{u, t}^s, 0)``

- **Objective**: minimize vehicle costs + average delay cost
"""

# ╔═╡ ca8e2520-89c0-4364-b893-877974d9854f
begin
	vsp_slack = load("./images/VSP/slack.png")
end

# ╔═╡ dc2158c8-4396-4d53-8bba-59fbc2cffa79
md"""
## MILP formulation
"""

# ╔═╡ 5c9a3382-ff79-4d72-a153-a0b768e5d8e1
md"""
We have the following hard problem: 

```math
(H)\left\{
\begin{aligned}
\min & \frac{1}{|S|} \sum_{s \in S} \sum_{p \in \mathcal{P}}c_p^s y_p &\\

\text{s.t.} & \sum_{p \ni v} y_p = 1 & \forall v \in V \backslash\{o, d\} \quad(\lambda_v \in \mathbb{R})\\

& y_p \in \{0,1\} & \forall p \in \mathcal{P}
\end{aligned}
\right.
```

"""

# ╔═╡ 9a7b8a7b-2e17-4b3b-b177-fef39e1e0354
md"""
## Approximate the hard problem

... with an easier one
```math
(E)\left\{
\begin{aligned}
\max_y &\quad \sum_{a\in\ A}\theta_a y_a\\
\text{s.t.} & \sum_{a\in\delta^-(t)}y_a = \sum_{a\in\delta^+(t)}y_a,\quad & \forall t\in T\\
& \sum_{a\in\delta^+(t)}y_a = 1, & \forall t\in T
\end{aligned}
\right.
```

``\implies`` Vehicle Scheduling Problem that can be solved with flow algorithms, or linear programming

Goal: for an instance ``x`` of ``(H)``, find ``\theta(x)`` such that the solution ``\hat f(\theta)`` of ``(E)`` is a good solution of ``(H)``
"""

# ╔═╡ 4db56292-397a-4c49-b7ff-a6a85264041d
md"""
## ML-CO pipeline
"""

# ╔═╡ 95f96123-2bcf-4935-9738-e0efd42a0daf
begin
	vsp_pipeline = load("./images/VSP/stovsp_pipeline.png")
end

# ╔═╡ e99c8278-a7bf-40af-adcc-21f41d4857b4
md"""
Machine learning predictor:
- ``\theta_a = w^T \phi(a, x)``
- learnable weights ``w``
- features $\phi$
"""

# ╔═╡ 5de471fa-4806-4b74-a1af-0cb25d81ba91
md"""
## Results: delays propagation
"""

# ╔═╡ e9d1aee8-b312-4540-8179-e9648e59fc53
begin 
	vspexperience = plot(load("./images/VSP/vsp_experience.png"))
	vspheuristic = plot(load("./images/VSP/vsp_heuristic.png"))
	plot(vspheuristic, vspexperience, layout = (1,2), ticks = nothing, border = nothing, size = (800, 300))
end

# ╔═╡ f8a2cace-50e1-4d5f-86b6-91c820bace26
md"""
!!! success
	TODO more detailed results
"""

# ╔═╡ c59d4022-fdd5-469f-8fb1-abbcb6a81c8a
md"""
## Two stage minimum weight spanning tree

TODO
"""

# ╔═╡ fde5498e-3d07-4276-b5d7-263c44d29da1
md"""
## Single machine scheduling problem

TODO
"""

# ╔═╡ 44039f2f-a1d8-4370-98b0-b7985d7d65bd
md"""
# 2. Theoretical background

A brief introduction to structured learning by imitation
"""

# ╔═╡ 87040fd6-bd1a-47dd-875c-2caf5b50d2ce
md"""
## Smoothing by regularization

```math
\xrightarrow[\text{instance $x$}]{\text{Problem}}
\fbox{NN $\varphi_w$}
\xrightarrow[\text{direction $\theta$}]{\text{Objective}}
\fbox{MILP $\underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y$}
\xrightarrow[\text{solution $\widehat{y}$}]{\text{Candidate}}
\fbox{Loss}
```

The combinatorial layer function

```math
f\colon \theta\mapsto \underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y
```
is piecewise constant $\implies$ no gradient information.

Given a convex function $\Omega$, the regularized optimizer is defined by:

```math
\hat{f}(\theta) = \underset{y \in \mathrm{conv}(\mathcal{Y})}{\mathrm{argmax}} \{\theta^\top y - \Omega(y)\} 
```

``\implies`` becomes differentiable.

Can be seen as an expectation over the vertices of $\mathrm{conv}(\mathcal{Y})$.

```math

\hat{f}(\theta) = \mathbb{E}_{\hat{p}(\cdot|\theta)}[Y] = \sum_{y\in\mathcal{Y}}~y~\hat{p}(y|\theta)
```
"""

# ╔═╡ 53f7468d-0015-4339-8e27-48812f541329
md"""
## Visualization
"""

# ╔═╡ 3db85997-e3f2-47b8-aa73-94080197be05
md"""
``\text{Number of vertices: n = } `` $(@bind N Slider(3:10; default=7, show_value=true))
"""

# ╔═╡ 79aa1f6b-553f-4873-925c-4db728f9f9eb
Y = [
	[(0.5 + 0.5*rand()) * cospi(2k/N), (0.5 + 0.5*rand()) * sinpi(2k/N)]
	for k in 0:N-1
];

# ╔═╡ 97d09291-910c-4c91-bc02-5c911c31a9a3
polytope_maximizer(θ) = Y[argmax(dot(θ, y) for y in Y)];

# ╔═╡ 72f8e7ec-193c-44d3-892d-aec4d4a631bb
function plot_polytope(α, predictor; title=nothing)
	θ = 0.4 .* [cos(α), sin(α)]
	ŷ = polytope_maximizer(θ)
	if predictor isa RegularizedGeneric
		probadist = compute_probability_distribution(
			predictor, θ;
			fw_kwargs=(
				epsilon=1e-3,
				max_iteration=100,
				away_steps=false
			)
		)
	elseif predictor isa InferOpt.AbstractPerturbed
		probadist = compute_probability_distribution(predictor, θ)
		InferOpt.compress_distribution!(probadist)
	else
		error("No proba distribution")
	end
	V = probadist.atoms
	Vs = sort(V, by=get_angle)
	p = probadist.weights
	ŷΩ = compute_expectation(probadist)

	pl = plot(;
		aspect_ratio=:equal,
		legend=:outerright,
		xlim=(-1.1, 1.1),
		ylim=(-1.1, 1.1),
	)
	if isnothing(title)
		plot!(framestyle=:none,)
	else
		plot!(title=title,)
	end
	plot!(
		vcat(map(first, Y), first(Y[1])),
		vcat(map(last, Y), last(Y[1]));
		fillrange=0,
		fillcolor=:gray,
		fillalpha=0.2,
		linecolor=:black,
		label=L"\mathrm{conv}(\mathcal{V})"
	)
	plot!(
		[0., θ[1]],
		[0., θ[2]],
		color=:black,
		arrow=true,
		lw=2,
		label=nothing
	)
	Plots.annotate!(
		[-0.2*θ[1]],
		[-0.2*θ[2]],
		[L"\theta"]
	)
	scatter!(
		[ŷ[1]],
		[ŷ[2]];
		color=:red,
		markersize=7,
		markershape=:square,
		label=L"f(\theta)"
	)
	plot!(
		vcat(map(first, Vs), first(Vs[1])),
		vcat(map(last, Vs), last(Vs[1]));
		fillrange=0,
		fillcolor=:blue,
		fillalpha=0.1,
		linestyle=:dash,
		linecolor=:black,
		label=L"\mathrm{conv}(\hat{p}(\theta))"
	)
	scatter!(
		map(first, V),
		map(last, V);
		markersize=25 .* p .^ 0.5,
		markercolor=:blue,
		markerstrokewidth=0,
		markeralpha=0.4,
		label=L"\hat{p}(\theta)"
	)
	scatter!(
		[ŷΩ[1]],
		[ŷΩ[2]];
		color=:blue,
		markersize=7,
		markershape=:hexagon,
		label=L"\hat{f}(\theta)"
	)
	scatter!(
		[ŷ[1]],
		[ŷ[2]];
		color=:red,
		markersize=7,
		markershape=:square,
		label=nothing
	)
	return pl
end

# ╔═╡ fb287847-98a8-4c64-9674-749f7ec22f24
md"""
``\alpha =`` $(@bind α_pert Slider(0:0.01:2π; default=π))
"""

# ╔═╡ a3907487-a5bb-4e35-a444-be0868bef029
begin
	set_ε_pert = md"""
	``\varepsilon = `` $(@bind ε_pert Slider(0.0:0.01:1; default=0.0, show_value=true))
	"""
	set_nb_samples_pert = md"""
	``M = `` $(@bind nb_samples_pert Slider(1:100; default=10, show_value=true))
	"""
	TwoColumn(set_ε_pert, set_nb_samples_pert)
end

# ╔═╡ 00cbb51e-75ca-46ae-8c0e-fce1182a3f8f
perturbed = PerturbedAdditive(polytope_maximizer; ε=ε_pert, nb_samples=nb_samples_pert, seed=0);

# ╔═╡ 0fc739a0-661f-4fca-8e61-b20779c537ff
plot_polytope(α_pert, perturbed; title="")

# ╔═╡ c72435b7-59ae-4f77-86fb-feb175ba88e6
md"""
``\alpha`` is the angle of the objective ``\theta``, ``\varepsilon`` is the size of the regularization, and $M$ is the number of samples drawn in order to estimate the expectation.
"""

# ╔═╡ 6801811b-f68a-43b4-8b78-2f27c0dc6331
md"""
## Fenchel-Young loss (learning by imitation)
Natural non-negative & convex loss based on regularization:
```math
\boxed{
\mathcal{L}_{\Omega}^{\text{FY}}(\theta, \bar{y}) = \Omega^*(\theta) + \Omega(\bar{y}) - \theta^\top \bar{y}
}
```
Given a target solution $\bar{y}$ and a parameter $\theta$, a subgradient is given by:
```math
\widehat{f}_{\Omega}(\theta) - \bar{y} \in \partial_\theta \mathcal{L}_{\Omega}^{\text{FY}}(\theta, \bar{y}).
```
The optimization block has meaningful gradients $\implies$ we can backpropagate through the whole pipeline.
"""

# ╔═╡ 3a84fd20-41fa-4156-9be5-a0371754b394
md"""
# 3. Pathfinding on Warcraft maps
"""

# ╔═╡ b7ef70d9-2b93-448a-b916-46655a857c8b
html"<button onclick='present()'>Toggle presentation mode</button>"

# ╔═╡ ee87d357-318f-40f1-a82a-fe680286e6cd
md"""
In this notebook, we define learning pipelines for the Warcraft shortest path problem. 
We have a sub-dataset of Warcraft terrain images, corresponding black-box cost functions, and optionally the label shortest path solutions and cell costs. 
We want to learn the cost of the cells, using a neural network embedding, to predict good shortest paths on new test images.
More precisely, each point in our dataset consists in:
- an image of terrain ``I``.
- a black-box cost function ``c`` to evaluate any given path.
- a label shortest path ``P`` from the top-left to the bottom-right corners (optional). 
- the true cost of each cell of the grid (optional).
We can exploit the images to approximate the true cell costs, so that when considering a new test image of terrain, we predict a good shortest path from its top-left to its bottom-right corners.
The question is: how should we combine these features?
We use `InferOpt` to learn the appropriate costs.
"""

# ╔═╡ 5c231f46-02b0-43f9-9101-9eb222cff972
begin
	load("./images/Warcraft/warcraft_pipeline.png")
end

# ╔═╡ 94192d5b-c4e9-487f-a36d-0261d9e86801
md"""
## I - Dataset and plots
"""

# ╔═╡ 98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
md"""
We first give the path of the dataset folder:
"""

# ╔═╡ 8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
decompressed_path = joinpath(".", "data")

# ╔═╡ 4e2a1703-5953-4901-b598-9c1a98a5fc2b
md"""
### a) Gridgraphs
"""

# ╔═╡ 6d1545af-9fd4-41b2-9a9b-b4472d6c360e
md"""For the purposes of this TP, we consider grid graphs, as implemented in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
In such graphs, each vertex corresponds to a couple of coordinates ``(i, j)``, where ``1 \leq i \leq h`` and ``1 \leq j \leq w``.
"""

# ╔═╡ e2c4292f-f2e8-4465-b3e3-66be158cacb5
h, w = 12, 12;

# ╔═╡ bd7a9013-199a-4bec-a5a4-4165da61f3cc
g = GridGraph(exp.(rand(100, 100)))

# ╔═╡ c04157e6-52a9-4d2e-add8-680dc71e5aaa
md"""For convenience, `GridGraphs.jl` also provides custom functions to compute shortest paths efficiently. We use the Dijkstra implementation.
Let us see what those paths look like.
"""

# ╔═╡ 16cae90f-6a37-4240-8608-05f3d9ab7eb5
begin
	p = path_to_matrix(g, grid_dijkstra(g, 1, nv(g)));
	UnicodePlots.spy(p)
end

# ╔═╡ 3044c025-bfb4-4563-8563-42a783e625e2
md"""
### b) Dataset functions
"""

# ╔═╡ 6d21f759-f945-40fc-aaa3-7374470c4ef0
md"""
The first dataset function `read_dataset` is used to read the images, cell costs and shortest path labels stored in files of the dataset folder.
"""

# ╔═╡ 3c141dfd-b888-4cf2-8304-7282aabb5aef
begin 
	"""
	    read_dataset(decompressed_path::String, dtype::String="train")

	Read the dataset of type `dtype` at the `decompressed_path` location.
	The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
	They are returned separately, with proper axis permutation and image scaling to be consistent with 
	`Flux` embeddings.
	"""
	function read_dataset(decompressed_path::String, dtype::String="train")
	    # Open files
	    data_dir = joinpath(decompressed_path, "warcraft_shortest_path_oneskin", "12x12")
	    data_suffix = "maps"
	    terrain_images = npzread(joinpath(data_dir, dtype * "_" * data_suffix * ".npy"))
	    terrain_weights = npzread(joinpath(data_dir, dtype * "_vertex_weights.npy"))
	    terrain_labels = npzread(joinpath(data_dir, dtype * "_shortest_paths.npy"))
	    # Reshape for Flux
	    terrain_images = permutedims(terrain_images, (2, 3, 4, 1))
	    terrain_labels = permutedims(terrain_labels, (2, 3, 1))
	    terrain_weights = permutedims(terrain_weights, (2, 3, 1))
	    # Normalize images
	    terrain_images = Array{Float32}(terrain_images ./ 255)
	    println("Train images shape: ", size(terrain_images))
	    println("Train labels shape: ", size(terrain_labels))
	    println("Weights shape:", size(terrain_weights))
	    return terrain_images, terrain_labels, terrain_weights
	end
end

# ╔═╡ c18d4b8f-2ae1-4fde-877b-f53823a42ab1
md"""
Once the files are read, we want to give an adequate format to the dataset, so that we can easily load samples to train and test models. The function `create_dataset` therefore calls the previous `read_dataset` function: 
"""

# ╔═╡ 8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
begin
	"""
	    create_dataset(decompressed_path::String, nb_samples::Int=10000)

	Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.
	The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
	It is a `Vector` of tuples, each `Tuple` being a dataset point.
	"""
	function create_dataset(decompressed_path::String, nb_samples::Int=10000)
	    terrain_images, terrain_labels, terrain_weights = read_dataset(
	        decompressed_path, "train"
	    )
	    X = [
	        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1)) for
	        i in 1:nb_samples
	    ]
	    Y = [terrain_labels[:, :, i] for i in 1:nb_samples]
	    WG = [(wg=GridGraph(terrain_weights[:, :, i]),) for i in 1:nb_samples]
	    return collect(zip(X, Y, WG))
	end
end

# ╔═╡ 4a9ed677-e294-4194-bf32-9580d1e47bda
md"""
Last, as usual in machine learning implementations, we split a dataset into train and test sets. The function `train_test_split` does the job:

"""

# ╔═╡ 0514cde6-b425-4fe7-ac1e-2678b64bbee5
begin
	"""
	    train_test_split(X::AbstractVector, train_percentage::Real=0.5)

	Split a dataset contained in `X` into train and test datasets.
	The proportion of the initial dataset kept in the train set is `train_percentage`.
	"""
	function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
	    N = length(X)
	    N_train = floor(Int, N * train_percentage)
	    N_test = N - N_train
	    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
	    X_train, X_test = X[train_ind], X[test_ind]
	    return X_train, X_test
	end
end

# ╔═╡ caf02d68-3418-4a6a-ae25-eabbbc7cae3f
md"""
### c) Plot functions
"""

# ╔═╡ 61db4159-84cd-4e3d-bc1e-35b35022b4be
md"""
In the following cell, we define utility plot functions to have a glimpse at images, cell costs and paths. Their implementation is not at the core of this tutorial, they are thus hidden.
"""

# ╔═╡ 08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
begin 
		"""
	    convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
	Convert `image` to the proper data format to enable plots in Julia.
	"""
	function convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
	    new_img = Array{RGB{N0f8},2}(undef, size(image)[1], size(image)[2])
	    for i = 1:size(image)[1]
	        for j = 1:size(image)[2]
	            new_img[i,j] = RGB{N0f8}(image[i,j,1], image[i,j,2], image[i,j,3])
	        end
	    end
	    return new_img
	end
	
	"""
	    plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})
	Plot the image `im` and the path `zero_one_path` on the same Figure.
	"""
	function plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})
	    p1 = plot(im, title = "Terrain map", ticks = nothing, border = nothing)
	    p2 = plot(Gray.(zero_one_path), title = "Path", ticks = nothing, border = nothing)
	    plot(p1, p2, layout = (1, 2))
	end
	
	"""
		plot_image_weights_path(;im, weights, path)
	Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
	"""
	function plot_image_weights_path(;im, weights, path)
		img = convert_image_for_plot(im)
	    p1 = Plots.plot(
	        img;
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(1000, 1000),
			title = "Terrain image"
	    )
	    p2 = Plots.heatmap(
			weights;
			yflip=true,
			aspect_ratio=:equal,
			framestyle=:none,
			padding=(0., 0.),
			size=(1000, 1000),
			legend = false,
			title = "Weights"
		)
	    p3 = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(1000, 1000),
			title = "Path"
	    )
	    plot(p1, p2, p3, layout = (1, 3), size = (3000, 1000))
	end
	
	"""
	    plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)
	
	Plot the train and test losses, as well as the train and test gaps computed over epochs.
	"""
	function plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)
	    x = collect(1:options.nb_epochs)
	    p1 = plot(x, losses, title = "Loss", xlabel = "epochs", ylabel = "loss", label = ["train" "test"])
	    p2 = plot(x, gaps, title = "Gap", xlabel = "epochs", ylabel = "ratio", label = ["train" "test"])
	    pl = plot(p1, p2, layout = (1, 2))
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	"""
	    plot_weights_path(;weights, path)
	Plot both the cell costs and path on the same colormap Figure.
	"""
	function plot_weights_path(;weights, path, weight_title="Weights", path_title="Path")
	    p1 = Plots.heatmap(
			weights;
			yflip=true,
			aspect_ratio=:equal,
			framestyle=:none,
			padding=(0., 0.),
			size=(500, 500),
			legend = false,
			title = weight_title
		)
	    p2 = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500),
			title = path_title
	    )
	    plot(p1, p2, layout = (1, 2), size = (1000, 500))
	end
	
	function plot_map(map_matrix::Array{<:Real,3}; filepath=nothing)
	    img = convert_image_for_plot(map_matrix)
	    pl = Plots.plot(
	        img;
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	function plot_weights(weights::Matrix{<:Real}; filepath=nothing)
	    pl = Plots.heatmap(
	        weights;
	        yflip=true,
	        aspect_ratio=:equal,
	        framestyle=:none,
	        padding=(0., 0.),
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	function plot_path(path::Matrix{<:Integer}; filepath=nothing)
	    pl = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
			end;

# ╔═╡ d58098e8-bba5-445c-b1c3-bfb597789916
md"""
### d) Import and explore the dataset
"""

# ╔═╡ a0644bb9-bf62-46aa-958e-aeeaaba3482e
md"""
Once we have both defined the functions to read and create a dataset, and to visualize it, we want to have a look at images and paths. Before that, we set the size of the dataset, as well as the train proportion: 
"""

# ╔═╡ eaf0cf1f-a7be-4399-86cc-66c131a57e44
nb_samples, train_prop = 100, 0.8;

# ╔═╡ 2470f5ab-64d6-49d5-9816-0c958714ca73
info(md"We focus only on $nb_samples dataset points, and use a $(trunc(Int, train_prop*100))% / $(trunc(Int, 100 - train_prop*100))% train/test split.")

# ╔═╡ 73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
begin
	dataset = create_dataset(decompressed_path, nb_samples)
	train_dataset, test_dataset = train_test_split(dataset, train_prop);
end;

# ╔═╡ c9a05a6e-90c3-465d-896c-74bbb429f66a
md"""
We can have a glimpse at the dataset, use the slider to visualize each tuple (image, weights, label path).
"""

# ╔═╡ fd83cbae-638e-49d7-88da-588fe055c963
md"""
``n =`` $(@bind n Slider(1:length(dataset); default=1, show_value=true))
"""

# ╔═╡ fe3d8a72-f68b-4162-b5f2-cc168e80a3c6
begin
	x, y, kwargs = dataset[n]
	plot_map(dropdims(x; dims=4))
end

# ╔═╡ 3ca72cd5-58f8-47e1-88ca-cd115b181e74
plot_weights_path(weights = kwargs.wg.weights, path =y)

# ╔═╡ fa62a7b3-8f17-42a3-8428-b2ac7eae737a
md"""
## II - Combinatorial functions
"""

# ╔═╡ 0f299cf1-f729-4999-af9d-4b39730100d8
md"""
We focus on additional optimization functions to define the combinatorial layer of our pipelines.
"""

# ╔═╡ e59b06d9-bc20-4d70-8940-5f0a53389738
md"""
### a) Recap on the shortest path problem
"""

# ╔═╡ 75fd015c-335a-481c-b2c5-4b33ca1a186a
md"""
Let $D = (V, A)$ be a digraph, $(c_a)_{a \in A}$ the cost associated to the arcs of the digraph, and $(o, d) \in V^2$ the origin and destination nodes. The problem we consider is the following:

**Shortest path problem:** Find an elementary path $P$ from node $o$ to node $d$ in the digraph $D$ with minimum cost $c(P) = \sum_{a \in P} c_a$.
"""

# ╔═╡ 7b653840-6292-4e6b-a6d6-91aadca3f6d4
md"""
!!! danger "Question"
	When the cost function is non-negative, which algorithm can we use ?
"""

# ╔═╡ 487eb4f1-cd50-47a7-8d61-b141c1b272f0
md"""
!!! danger "Question" 
	In the case the graph contains no absorbing cycle, which algorithm can we use ? 	On which principle is it based ?
"""

# ╔═╡ 654066dc-98fe-4c3b-92a9-d09efdfc8080
md"""
In the following, we will perturb or regularize the output of a neural network to define the candidate cell costs to predict shortest paths. We therefore need to deal with possibly negative costs. 

!!! danger "Question"
	In the general case, can we fix the maximum length of a feasible solution of the shortest path problem ? How ? Can we derive an dynamic programming algorithm based on this ?
"""

# ╔═╡ f18ad74f-ef8b-4c70-8af3-e6dac8305dd0
begin
	
"""
    grid_bellman_ford_warcraft(g, s, d, length_max)

Apply the Bellman-Ford algorithm on an `GridGraph` `g`, and return a `ShortestPathTree` with source `s` and destination `d`,
among the paths having length smaller than `length_max`.
"""
function grid_bellman_ford_warcraft(g::GridGraph{T,R,W,A}, s::Integer, d::Integer, length_max::Int = nv(g)) where {T,R,W,A}
    # Init storage
    parents = zeros(T, nv(g), length_max+1)
    dists = Matrix{Union{Nothing,R}}(undef, nv(g), length_max+1)
    fill!(dists, Inf)
    # Add source
    dists[s,1] = zero(R)
    # Main loop
    for k in 1:length_max
        for v in vertices(g)
            for u in inneighbors(g, v)
                d_u = dists[u, k]
                if !isinf(d_u)
                    d_v = dists[v, k+1]
                    d_v_through_u = d_u + GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k+1] = d_v_through_u
                        parents[v, k+1] = u
                    end
                end
            end
        end
    end
    # Get length of the shortest path
    k_short = argmin(dists[d,:])
    if isinf(dists[d, k_short])
        println("No shortest path with less than $length_max arcs")
        return T[]
    end
    # Deduce the path
    v = d
    path = [v]
    k = k_short
    while v != s
        v = parents[v, k]
        if v == zero(T)
            return T[]
        else
            pushfirst!(path, v)
            k = k-1
        end
    end
    return path
end
end

# ╔═╡ dc359052-19d9-4f29-903c-7eb9b210cbcd
md"""
###  b) From shortest path to generic maximizer
"""

# ╔═╡ b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
md"""
Now that we have defined and implemented an algorithm to deal with the shortest path problem, we wrap it in a maximizer function to match the generic framework of structured prediction.
"""

# ╔═╡ 20999544-cefd-4d00-a68c-cb6cfea36b1a
function dijkstra_maximizer(θ::AbstractMatrix{<:Real}; kwargs...)
	g = GridGraph(-θ)
	path = grid_dijkstra(g, 1, nv(g))
	y = path_to_matrix(g, path)
	return y
end

# ╔═╡ b2ea7e31-82c6-4b01-a8c6-26c3d7a2d562
function bellman_maximizer(θ::AbstractMatrix{<:Real}; kwargs...)
	g = GridGraph(-θ)
	path = grid_bellman_ford_warcraft(g, 1, nv(g))
	y = path_to_matrix(g, path)
	return y
end

# ╔═╡ 76d4caa4-a10c-4247-a624-b6bfa5a743bc
md"""
!!! info "The maximizer function will depend on the pipeline"
	Note that we use the function `grid_dijkstra` already implemented in the `GridGraphs.jl` package when we deal with non-negative cell costs. In the following, we will use either Dijkstra or Ford-Bellman algorithm depending on the learning pipeline. You will have to modify the maximizer function to use depending on the experience you do.
"""

# ╔═╡ 91ec470d-f2b5-41c1-a50f-fc337995c73f
md"""
## III - Learning functions
"""

# ╔═╡ f899c053-335f-46e9-bfde-536f642700a1
md"""
### a) Convolutional neural network: predictor for the cost vector
"""

# ╔═╡ 6466157f-3956-45b9-981f-77592116170d
md"""
We implement several elementary functions to define our machine learning predictor for the cell costs.
"""

# ╔═╡ 211fc3c5-a48a-41e8-a506-990a229026fc
begin
	"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x, dims = [3])/size(x)[3]
end
end

# ╔═╡ 7b8b659c-9c7f-402d-aa7b-63c17179560e
begin 
	"""
    neg_exponential_tensor(x)

Compute minus exponential element-wise on tensor `x`.
"""
function neg_exponential_tensor(x)
    return -exp.(x)
end
end

# ╔═╡ e392008f-1a92-4937-8d8e-820211e44422
begin
	"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x)[1], size(x)[2])
end
end

# ╔═╡ 8f23f8cc-6393-4b11-9966-6af67c6ecd40
md"""
!!! info "CNN as predictor"
	The following function defines the convolutional neural network we will use as cell costs predictor.
"""

# ╔═╡ 51a44f11-646c-4f1a-916e-6c83750f8f20
begin
	"""
    create_warcraft_embedding()

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
    1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
    2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
    3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
    4) The element-wize [`neg_exponential_tensor`](@ref) function to get cell weights of proper sign to apply shortest path algorithms.
    4) A squeeze function to forget the two last dimensions. 
"""
function create_warcraft_embedding()
    resnet18 = ResNet(18, pretrain = false, nclasses = 1)
    model_embedding = Chain(
		resnet18.layers[1][1:4], 
        AdaptiveMaxPool((12,12)), 
        average_tensor, 
        neg_exponential_tensor, 
        squeeze_last_dims,
    )
    return model_embedding
end
end

# ╔═╡ d793acb0-fd30-48ba-8300-dff9caac536a
md"""
We can build the encoder in this way:
"""

# ╔═╡ d9f5281b-f34b-485c-a781-804b8472e38c
initial_encoder = create_warcraft_embedding()

# ╔═╡ 9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
md"""
### b) Loss and gap utility functions
"""

# ╔═╡ 596734af-cf81-43c9-a525-7ea88a209a53
md"""
In the cell below, we define the `cost` function seen as black-box to evaluate the cost of a given path on the grid.
"""

# ╔═╡ 0ae90d3d-c718-44b2-81b5-25ce43f42988
cost(y; c_true, kwargs...) = dot(y, c_true)

# ╔═╡ 201ec4fd-01b1-49c4-a104-3d619ffb447b
md"""
The following cell defines the scaled half square norm function and its gradient.
"""

# ╔═╡ 8b544491-b892-499f-8146-e7d1f02aaac1
begin
	scaled_half_square_norm(x::AbstractArray{<:Real}, ϵ::R = 25.) where {R<:Real} = ϵ*sum(abs2, x) / 2
	
	grad_scaled_half_square_norm(x::AbstractArray{<:Real}, ϵ::R = 25.) where {R<:Real} = ϵ*identity(x)
end

# ╔═╡ 6a482757-8a04-4724-a3d2-33577748bd4e
md"""
During training, we want to evaluate the quality of the predicted paths, both on the train and test datasets. We define the shortest path cost ratio between a candidate shortest path $\hat{y}$ and the label shortest path $y$ as: $r(\hat{y},y) = c(\hat{y}) / c(y)$.
"""

# ╔═╡ c89f17b8-fccb-4d62-a0b7-a84bbfa543f7
md"""
!!! danger "Question"
	What is the link in our problem between the shortest path cost ratio and the gap of a given solution with respect to the optimal solution ?
"""

# ╔═╡ 9eb0ca01-bd65-48df-ab32-beaca2e38482
md"""
!!! danger "Todo"
	Complete the following code to define the `shortest_path_cost_ratio` function. The candidate path $\hat{y}$ is given by the output of `model` applied on image `x`, and `y` is the target shortest path.
"""

# ╔═╡ b25f438f-832c-4717-bb73-acbb22aec384
md"""
The two following functions extend the shortest path cost ratio to a batch and a dataset.

"""

# ╔═╡ dd1791a8-fa59-4a36-8794-fccdcd7c912a
begin
	"""
	    shortest_path_cost_ratio(model, batch)
	Compute the average cost ratio between computed and true shorest paths over `batch`. 
	"""
	function shortest_path_cost_ratio(model, batch)
	    return sum(shortest_path_cost_ratio(model, item[1], item[2], item[3]) for item in batch)/length(batch)
	end
end

# ╔═╡ 633e9fea-fba3-4fe6-bd45-d19f89cb1808
begin
	"""
	    shortest_path_cost_ratio(;model, dataset)
	Compute the average cost ratio between computed and true shorest paths over `dataset`. 
	"""
	function shortest_path_cost_ratio(;model, dataset)
	    return sum(shortest_path_cost_ratio(model, batch) for batch in dataset)/length(dataset)
	end
end

# ╔═╡ 8c8b514e-8478-4b2b-b062-56832115c670
md"""
### c) Main training function:
"""

# ╔═╡ 93dd97e6-0d37-4d94-a3f6-c63dc856fa66
md"""
We now consider the generic learning function. We want to minimize a given `flux_loss` over the `train_dataset`, by updating the parameters of `encoder`. We do so using `Flux.jl` package which contains utility functions to backpropagate in a stochastic gradient descent framework. We also track the loss and cost ratio metrics both on the train and test sets. The hyper-parameters are stored in the `options` tuple. 
"""

# ╔═╡ d35f0e8b-6634-412c-b5f3-ffd11246276c
md"""
The following block defines the generic learning function.
"""

# ╔═╡ 920d94cd-bfb5-4c02-baa3-f346d5c95e2e
md"""
## IV - Pipelines
"""

# ╔═╡ 658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
md"""
!!! info "Preliminary remark"
	Here come the specific learning experiments. The following code cells will have to be modified to deal with different settings.
"""

# ╔═╡ f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
md"""
As you know, the solution of a linear program is not differentiable with respect to its cost vector. Therefore, we need additional tricks to be able to update the parameters of the CNN defined by `create_warcraft_embedding`. Two points of view can be adopted: perturb or regularize the maximization problem. They can be unified when introducing probabilistic combinatorial layers, detailed in this [paper](https://arxiv.org/pdf/2207.13513.pdf). They are used in two different frameworks:

- Learning by imitation when we have target shortest path examples in the dataset.
- Learning by experience when we only have access to the images and to a black-box cost function to evaluate any candidate path.

In this section, we explore different combinatorial layers, as well as the learning by imitation and learning by experience settings.
"""

# ╔═╡ 9a670af7-cc20-446d-bf22-4e833cc9d854
md"""
### 1) Learning by imitation with additive perturbation
"""

# ╔═╡ f6949520-d10f-4bae-8d41-2ec880ac7484
md"""
In this framework, we use a perturbed maximizer to learn the parameters of the neural network. Given a maximization problem $f(\theta) := \operatorname{argmax}_{y \in \mathcal{C}} \langle y, \theta \rangle$, we define the additive perturbed maximization as:
"""

# ╔═╡ 9bef7690-0db3-4ba5-be77-0933ceb6215e
md"""
```math
\hat{f}^+_\varepsilon (\theta) := \mathbb{E}_{Z}\big[ \underset{y \in \mathcal{C}}{\mathrm{argmax}} (\theta + \varepsilon Z)^\top y \big]
```
"""

# ╔═╡ c872d563-421c-4581-a8fa-a02cee58bc85
md"""
```math
$F^+_\varepsilon (\theta) := \mathbb{E}_{Z}\big[ \max_{y \in \mathcal{C}} (\theta + \varepsilon Z)^\top y \big]$ 
```
"""

# ╔═╡ 4d50d263-eca0-48ad-b32c-9b767cc57914
md"""
!!! danger "Question"
	From your homework, what can you say about $F^+_\epsilon (\theta)$ and $y^+_\epsilon (\theta)$ ? What are their properties ? 
"""

# ╔═╡ e4b13e58-2d54-47df-b865-95ae2946756a
md"""
Let $\Omega_\varepsilon^+$ be the Fenchel conjugate of $F^+_\varepsilon (\theta)$, we can define the natural Fenchel-Young loss as:

"""

# ╔═╡ 9c05cae5-af20-4f63-99c9-86032358ffd3
md"""
$L_\varepsilon^+ (\theta, y) := F^+_{\varepsilon} (\theta) + \Omega_{\varepsilon}^+ (y) - \langle \theta, y \rangle$
"""

# ╔═╡ d2e5f60d-199a-41f5-ba5d-d21ab2030fb8
md"""
!!! danger "Question"
	What are the properties of $L_\varepsilon^+ (\theta, y)$ ?
"""

# ╔═╡ 6293fde0-3cfc-4d0d-bed6-74caa54b6ead
md"""
!!! note "How to implement it ?"
	The Fenchel-Young loss with additive perturbation is implemented below, using `FenchelYoungLoss` and `PerturbedAdditive` implemented in `InferOpt.jl`.
"""

# ╔═╡ 9a9b3942-72f2-4c9e-88a5-af927634468c
md"""
### 2) Learning by imitation with multiplicative perturbation
"""

# ╔═╡ 1ff198ea-afd5-4acc-bb67-019051ff149b
md"""
We introduce a variant of the additive pertubation defined above, which is simply based on an element-wise product $\odot$:
"""

# ╔═╡ 44ece9ce-f9f1-46f3-90c6-cb0502c92c67
md"""
${y}_\varepsilon^\odot (\theta) := \mathbb{E}_Z \bigg[\operatorname{argmax}_{y \in \mathcal{C}} \langle \theta \odot e^{\epsilon Z - \varepsilon^2 \mathbf{1} / 2},  y \rangle \bigg]$
"""

# ╔═╡ 5d8d34bb-c207-40fc-ab10-c579e7e2d04c
md"""
!!! danger "Question"
	What is the advantage of this perturbation compared with the additive one in terms of combinatorial problem ? Which algorithm can we use to compute shortest paths ?
"""

# ╔═╡ 43d68541-84a5-4a63-9d8f-43783cc27ccc
md"""
We omit the details of the loss derivations and concentrate on implementation.

!!! danger "Todo"
	You can modify the previous additive implementation below, by replacing the `PerturbedAdditive` regularization with a `PerturbedMultiplicative` one. You can also modify use `dijkstra_maximizer` instead of `belmann_maximizer` as the `true_maximizer`, which runs faster.
"""

# ╔═╡ 0fd29811-9e17-4c97-b9b7-ec9cc51b435f
md"""
## 3) Smart Predict then optimize

TODO
"""

# ╔═╡ 90a47e0b-b911-4728-80b5-6ed74607833d
md"""
### 4) Learning by experience with multiplicative perturbation
"""

# ╔═╡ 5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
md"""
When we restrict the train dataset to images $I$ and black-box cost functions $c$, we can not learn by imitation. We can instead derive a surrogate version of the regret that is differentiable. 

!!! info "Reading"
	Read Section 4.1 of this [paper](https://arxiv.org/pdf/2207.13513.pdf).

!!! danger "Todo"
	Modify the code below to learn by experience using a multiplicative perturbation and the black-box cost function.
"""

# ╔═╡ a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
md"""
### 5) Learning by experience with half square norm regularization (bonus). 
"""

# ╔═╡ a7b6ecbd-1407-44dd-809e-33311970af12
md"""
For the moment, we have only considered perturbations to derive meaningful gradient information. We now focus on a half square norm regularization.

!!! danger "Todo"
	Based on the functions `scaled_half_square_norm` and `grad_scaled_half_square_norm`, use the `RegularizedGeneric` implementation of [`InferOpt.jl`](https://axelparmentier.github.io/InferOpt.jl/dev/algorithms/) to learn by experience. Modify the cells below to do so.
"""

# ╔═╡ b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
md"""
#### a) Hyperparameters
"""

# ╔═╡ e0e97839-884a-49ed-bee4-f1f2ace5f5e0
md"""
We first define the hyper-parameters for the learning process. They include:
- The regularization size $\varepsilon$.
- The number of samples drawn for the approximation of the expectation $M$.
- The number of learning epochs `nb_epochs`.
- The number of samples in the sub-dataset (train + test) `nb_samples`.
- The batch size for the stochastic gradient descent `batch_size`.
- The starting learning rate for ADAM optimizer `lr_start`.
"""

# ╔═╡ bcdd60b8-e0d8-4a70-88d6-725269447c9b
begin 
	ε = 0.05
	M = 20
	nb_epochs = 50
	batch_size = 80
	lr_start = 0.001
	options = (ε=ε, M=M, nb_epochs=nb_epochs, nb_samples=nb_samples, batch_size = batch_size, lr_start = lr_start)
end

# ╔═╡ 9de99f4a-9970-4be1-9e16-e64ed4e10277
md"""
#### b) Specific pipeline
"""

# ╔═╡ 518e7077-d61b-4f60-987f-d556e3eb1d0d
md"""
!!! info "What is a pipeline ?"
	This portion of code is the crucial part to define the learning pipeline. It contains: 
	- an encoder, the machine learning predictor, in our case a CNN.
	- a maximizer possibly applied to the output of the encoder before computing the loss.
	- a differentiable loss to evaluate the quality of the output of the pipeline.
	
	Its definition depends on the learning setting we consider.
"""

# ╔═╡ 1337513f-995f-4dfa-827d-797a5d2e5e1a
begin
	true_maximizer = bellman_maximizer
	pipeline = (
	    encoder=deepcopy(initial_encoder),
	    maximizer=identity, # TODO: remove that ?? more confusing than anything else
		loss = FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=options.ε, nb_samples=options.M)),
		#loss=SPOPlusLoss(bellman_maximizer)
	)
end

# ╔═╡ 26c71a94-5b30-424f-8242-c6510d41bb52
begin 
	"""
	    shortest_path_cost_ratio(model, x, y, kwargs)
	Compute the ratio between the cost of the solution given by the `model` cell costs and the cost of the true solution.
	We evaluate both the shortest path with respect to the weights given by `model(x)` and the labelled shortest path `y`
	using the true cell costs stored in `kwargs.wg.weights`. 
	This ratio is by definition greater than one. The closer it is to one, the better is the solution given by the current 
	weights of `model`. We thus track this metric during training.
	"""
	function shortest_path_cost_ratio(model, x, y, kwargs)
	    true_weights = kwargs.wg.weights
	    θ_computed = model(x)
	    shortest_path_computed = true_maximizer(θ_computed)
	    return dot(true_weights, shortest_path_computed)/dot(y, true_weights)
	end
end

# ╔═╡ a6a56523-90c9-40d2-9b68-26e20c1a5527
begin 
	"""
	    train_function!(;encoder, flux_loss, train_dataset, test_dataset, options::NamedTuple)
	Train `encoder` model over `train_dataset` and test on `test_dataset` by minimizing `flux_loss` loss. 
	This training involves differentiation through argmax with perturbed maximizers, using [InferOpt](https://github.com/axelparmentier/InferOpt.jl) package.
	The task is to learn the best parameters for the `encoder`, so that when solving the shortest path problem with its output cell costs, the 
	given solution is close to the labelled shortest path corresponding to the input Warcraft terrain image.
	Hyperparameters are passed with `options`. During training, the average train and test losses are stored, as well as the average 
	cost ratio computed with [`shortest_path_cost_ratio`](@ref) both on the train and test datasets.
	"""
	function train_function!(; encoder, flux_loss, train_dataset, test_dataset, options::NamedTuple)
	    # Store the train loss
	    losses = Matrix{Float64}(undef, options.nb_epochs, 2)
	    cost_ratios = Matrix{Float64}(undef, options.nb_epochs, 2)
	    # Optimizer
	    opt = ADAM(options.lr_start)
	    # model parameters
	    par = Flux.params(encoder)
	    # Train loop
	    @progress "Training epoch: " for epoch in 1:options.nb_epochs
	        for batch in train_dataset
	            batch_loss = 0
	            gs = gradient(par) do
	                batch_loss = flux_loss(batch)
	            end
	            losses[epoch, 1] += batch_loss
	            Flux.update!(opt, par, gs)
	        end
	        losses[epoch, 1] = losses[epoch, 1]/(options.nb_samples*0.8)
	        losses[epoch, 2] = sum([flux_loss(batch) for batch in test_dataset])/(options.nb_samples*0.2)
	        cost_ratios[epoch, 1] = shortest_path_cost_ratio(model = encoder, dataset = train_dataset)
	        cost_ratios[epoch, 2] = shortest_path_cost_ratio(model = encoder, dataset = test_dataset)
	    end
	     return losses, cost_ratios
	end

end

# ╔═╡ f5e789b2-a62e-4818-90c3-76f39ea11aaa
md"""
#### c) Flux loss definition
"""

# ╔═╡ efa7736c-22c0-410e-94da-1df315f22bbf
md"""
From the generic definition of the pipeline we define a loss function compatible with `Flux.jl` package. Its definition depends on the learning setting we consider.
"""

# ╔═╡ e9df3afb-fa04-440f-9664-3496da85696b
begin
	(; encoder, maximizer, loss) = pipeline
	# For learning by imitation
	flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), y; fw_kwargs = (max_iteration=50,))

	# For learning by experience
	#flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)); c_true = #kwargs.wg.weights, fw_kwargs = (max_iteration=50,))

	# For SPO+ loss
	#flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), -kwargs.wg.weights, y)

	flux_loss_batch(batch) = sum(flux_loss_point(item[1], item[2], item[3]) for item in batch)
end

# ╔═╡ 58b7267d-491d-40f0-b4ba-27ed0c9cc855
md"""
#### d) Apply the learning function
"""

# ╔═╡ ac76b646-7c28-4384-9f04-5e4de5df154f
md"""
Given the specific pipeline and loss, we can apply our generic train function to update the weights of the CNN predictor.
"""

# ╔═╡ 83a14158-33d1-4f16-85e1-2726c8fbbdfc
begin
	Losses, Cost_ratios = train_function!(;
	    encoder=encoder,
	    flux_loss = flux_loss_batch,
	    train_dataset=Flux.DataLoader(train_dataset; batchsize=batch_size),
	    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
	    options=options,
	)
	Gaps = (Cost_ratios .- 1) .* 100
end;

# ╔═╡ 4b31dca2-0195-4899-8a3a-e9772fabf495
md"""
#### e) Plot results
"""

# ╔═╡ 79e0deab-1e36-4863-ad10-187ed8555c72
md"""
Loss and gap over epochs, train and test datasets.
"""

# ╔═╡ 66d385ba-9c6e-4378-b4e0-e54a4df346a5
plot_loss_and_gap(Losses, Gaps, options)

# ╔═╡ db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
md"""
To assess performance, we can compare the true and predicted paths.
"""

# ╔═╡ eb3a6009-e181-443c-bb77-021e867030e4
md"""
!!! info "Visualize the model performance"
	We now want to see the effect of the learning process on the predicted costs and shortest paths. Use the slider to swipe through the test dataset.
"""

# ╔═╡ 521f5ffa-2c22-44c5-8bdb-67410431ca2e
begin
	test_predictions = []
	for (x, y, k) in test_dataset
		θ0 = initial_encoder(x)
		y0 = UInt8.(true_maximizer(θ0))
		θp = encoder(x)
		yp = UInt8.(true_maximizer(θp))
		push!(test_predictions, (x, y, k, θ0, y0, θp, yp))
	end
end

# ╔═╡ f9b35e98-347f-4ebd-a690-790c7b0e03d8
md"""
``j =`` $(@bind j Slider(1:length(test_dataset); default=1, show_value=true))
"""

# ╔═╡ 842bf89d-45eb-462d-ba74-ca260a8b177d
begin
	x_test, y_test, kwargs_test, θini, yini, θpred, ypred = test_predictions[j]
	plot_map(dropdims(x_test; dims=4))
end

# ╔═╡ 80fa8831-924f-4093-a89c-bf8fc440da6b
plot_weights_path(weights=kwargs_test.wg.weights, path=y_test, weight_title="True weights", path_title="Label shortest path")

# ╔═╡ 4a3630ca-c8dd-4e81-8ee2-bb0fc6b01a93
plot_weights_path(weights=-θpred, path=ypred, weight_title="Predicted weight", path_title="Predicted path")

# ╔═╡ e2b38ec3-2de9-49f1-b29a-e746014e4fe1
plot_weights_path(weights=-θini, path=yini, weight_title="Initial predicted weight", path_title="Initial predicted path")

# ╔═╡ cfc02683-589b-44dc-a126-257703ed5f85
(minimum(-θpred), maximum(-θpred))

# ╔═╡ 4caa7341-8750-44ec-ba4e-4d425836996d
(minimum(-θini), maximum(-θini))

# ╔═╡ d56a9e90-b0f6-4bde-8b3b-ebf1d962f6b4
md"""
# Conclusion
"""

# ╔═╡ 9852e80b-1f8d-445e-96bf-f7e071d6715c
md"""
### For more information

- **Main package:** <https://github.com/axelparmentier/InferOpt.jl>
- **This notebook:** <https://github.com/BatyLeo/InferOpt-CERMICS2022>
- Our paper on ArXiV: https://arxiv.org/abs/2207.13513

Detailed application examples:
- **Shortest paths on Warcraft maps:** <https://github.com/LouisBouvier/WarcraftShortestPaths.jl>
- **Stochastic vehicle scheduling:** <https://github.com/BatyLeo/StochasticVehicleScheduling.jl>
- **Single machine scheduling:** <https://github.com/axelparmentier/SingleMachineScheduling.jl>
- **Two stage spanning tree:** <https://github.com/axelparmentier/MinimumWeightTwoStageSpanningTree.jl>
"""

# ╔═╡ Cell order:
# ╟─e279878d-9c8d-47c8-9453-3aee1118818b
# ╟─8b7876e4-2f28-42f8-87a1-459b665cff30
# ╠═6160785a-d455-40b4-ab74-e61c46e31537
# ╟─a0d14396-cb6a-4f35-977a-cf3b63b44d9e
# ╟─b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
# ╟─2182d4d2-6506-4fd6-936f-0e7c30d73851
# ╟─1f0c5b88-f903-4a67-9581-b3a07c504d5c
# ╟─a8d0f8be-01a8-4a2a-84e3-ca16e7ef5203
# ╟─86735dcf-de5b-4f32-8bf9-501e006f58d5
# ╟─f4800c47-0f98-4ec7-85e6-5c2a19f784f5
# ╟─3e7077cd-4a14-4971-801d-9b9eadd59624
# ╟─72f8e7ec-193c-44d3-892d-aec4d4a631bb
# ╟─14af0338-554a-4f71-a290-3b4f16cc6af5
# ╟─a6cb796f-32bf-4dbd-a9d4-61b454dad548
# ╟─83a3efe5-d3ed-4f00-b90f-b065ca5ac06a
# ╟─f13bf21c-33db-4620-add8-bfd82f493d7c
# ╟─f99d6992-dc3e-41d1-8922-4958886dade2
# ╟─0d20da65-1e53-4b6e-b302-28243c94fb4c
# ╟─3277f951-dd62-4fd0-be19-82c4572b6361
# ╟─d0c4164c-328d-46ab-bd97-81f989a9e081
# ╟─103dc4a8-e1a9-425a-807a-7c5a4446bea7
# ╟─5cd285ca-9f45-42bf-ad8a-937d0e47581b
# ╟─3a42e362-b826-4bb9-a063-bbdb71f0b2b4
# ╟─64c69987-6416-488b-8b7f-55d48771184d
# ╟─aac53339-1091-4308-8d61-5ab4d3334c26
# ╟─16963349-5467-4019-be3d-d1b5375cf90e
# ╟─1574b408-cf50-4c57-9fb8-eaa22bb3ece1
# ╟─400867ad-11e6-411b-8b1f-c64685630fdc
# ╟─dda2e192-36fa-418b-8f4e-4cb3afd69360
# ╟─5764e92d-7fc4-4b62-a709-79979fb4b90c
# ╟─799fcc82-6a25-47a7-8b52-32a754d4e875
# ╟─ca8e2520-89c0-4364-b893-877974d9854f
# ╟─dc2158c8-4396-4d53-8bba-59fbc2cffa79
# ╟─5c9a3382-ff79-4d72-a153-a0b768e5d8e1
# ╟─9a7b8a7b-2e17-4b3b-b177-fef39e1e0354
# ╟─4db56292-397a-4c49-b7ff-a6a85264041d
# ╟─95f96123-2bcf-4935-9738-e0efd42a0daf
# ╟─e99c8278-a7bf-40af-adcc-21f41d4857b4
# ╟─5de471fa-4806-4b74-a1af-0cb25d81ba91
# ╟─e9d1aee8-b312-4540-8179-e9648e59fc53
# ╠═f8a2cace-50e1-4d5f-86b6-91c820bace26
# ╟─c59d4022-fdd5-469f-8fb1-abbcb6a81c8a
# ╟─fde5498e-3d07-4276-b5d7-263c44d29da1
# ╟─44039f2f-a1d8-4370-98b0-b7985d7d65bd
# ╟─87040fd6-bd1a-47dd-875c-2caf5b50d2ce
# ╟─79aa1f6b-553f-4873-925c-4db728f9f9eb
# ╟─00cbb51e-75ca-46ae-8c0e-fce1182a3f8f
# ╟─97d09291-910c-4c91-bc02-5c911c31a9a3
# ╟─53f7468d-0015-4339-8e27-48812f541329
# ╟─3db85997-e3f2-47b8-aa73-94080197be05
# ╟─0fc739a0-661f-4fca-8e61-b20779c537ff
# ╟─fb287847-98a8-4c64-9674-749f7ec22f24
# ╟─a3907487-a5bb-4e35-a444-be0868bef029
# ╟─c72435b7-59ae-4f77-86fb-feb175ba88e6
# ╟─6801811b-f68a-43b4-8b78-2f27c0dc6331
# ╟─3a84fd20-41fa-4156-9be5-a0371754b394
# ╟─b7ef70d9-2b93-448a-b916-46655a857c8b
# ╟─ee87d357-318f-40f1-a82a-fe680286e6cd
# ╟─5c231f46-02b0-43f9-9101-9eb222cff972
# ╟─94192d5b-c4e9-487f-a36d-0261d9e86801
# ╟─98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
# ╠═8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
# ╟─4e2a1703-5953-4901-b598-9c1a98a5fc2b
# ╟─6d1545af-9fd4-41b2-9a9b-b4472d6c360e
# ╠═e2c4292f-f2e8-4465-b3e3-66be158cacb5
# ╠═bd7a9013-199a-4bec-a5a4-4165da61f3cc
# ╟─c04157e6-52a9-4d2e-add8-680dc71e5aaa
# ╠═16cae90f-6a37-4240-8608-05f3d9ab7eb5
# ╟─3044c025-bfb4-4563-8563-42a783e625e2
# ╟─6d21f759-f945-40fc-aaa3-7374470c4ef0
# ╟─3c141dfd-b888-4cf2-8304-7282aabb5aef
# ╟─c18d4b8f-2ae1-4fde-877b-f53823a42ab1
# ╟─8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
# ╟─4a9ed677-e294-4194-bf32-9580d1e47bda
# ╟─0514cde6-b425-4fe7-ac1e-2678b64bbee5
# ╟─caf02d68-3418-4a6a-ae25-eabbbc7cae3f
# ╟─61db4159-84cd-4e3d-bc1e-35b35022b4be
# ╟─08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
# ╟─d58098e8-bba5-445c-b1c3-bfb597789916
# ╟─a0644bb9-bf62-46aa-958e-aeeaaba3482e
# ╠═eaf0cf1f-a7be-4399-86cc-66c131a57e44
# ╟─2470f5ab-64d6-49d5-9816-0c958714ca73
# ╠═73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
# ╟─c9a05a6e-90c3-465d-896c-74bbb429f66a
# ╟─fe3d8a72-f68b-4162-b5f2-cc168e80a3c6
# ╟─fd83cbae-638e-49d7-88da-588fe055c963
# ╟─3ca72cd5-58f8-47e1-88ca-cd115b181e74
# ╟─fa62a7b3-8f17-42a3-8428-b2ac7eae737a
# ╟─0f299cf1-f729-4999-af9d-4b39730100d8
# ╟─e59b06d9-bc20-4d70-8940-5f0a53389738
# ╟─75fd015c-335a-481c-b2c5-4b33ca1a186a
# ╟─7b653840-6292-4e6b-a6d6-91aadca3f6d4
# ╟─487eb4f1-cd50-47a7-8d61-b141c1b272f0
# ╟─654066dc-98fe-4c3b-92a9-d09efdfc8080
# ╟─f18ad74f-ef8b-4c70-8af3-e6dac8305dd0
# ╟─dc359052-19d9-4f29-903c-7eb9b210cbcd
# ╟─b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
# ╠═20999544-cefd-4d00-a68c-cb6cfea36b1a
# ╠═b2ea7e31-82c6-4b01-a8c6-26c3d7a2d562
# ╟─76d4caa4-a10c-4247-a624-b6bfa5a743bc
# ╟─91ec470d-f2b5-41c1-a50f-fc337995c73f
# ╟─f899c053-335f-46e9-bfde-536f642700a1
# ╟─6466157f-3956-45b9-981f-77592116170d
# ╟─211fc3c5-a48a-41e8-a506-990a229026fc
# ╟─7b8b659c-9c7f-402d-aa7b-63c17179560e
# ╟─e392008f-1a92-4937-8d8e-820211e44422
# ╟─8f23f8cc-6393-4b11-9966-6af67c6ecd40
# ╟─51a44f11-646c-4f1a-916e-6c83750f8f20
# ╟─d793acb0-fd30-48ba-8300-dff9caac536a
# ╠═d9f5281b-f34b-485c-a781-804b8472e38c
# ╟─9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
# ╟─596734af-cf81-43c9-a525-7ea88a209a53
# ╠═0ae90d3d-c718-44b2-81b5-25ce43f42988
# ╟─201ec4fd-01b1-49c4-a104-3d619ffb447b
# ╠═8b544491-b892-499f-8146-e7d1f02aaac1
# ╟─6a482757-8a04-4724-a3d2-33577748bd4e
# ╟─c89f17b8-fccb-4d62-a0b7-a84bbfa543f7
# ╟─9eb0ca01-bd65-48df-ab32-beaca2e38482
# ╟─26c71a94-5b30-424f-8242-c6510d41bb52
# ╟─b25f438f-832c-4717-bb73-acbb22aec384
# ╟─dd1791a8-fa59-4a36-8794-fccdcd7c912a
# ╟─633e9fea-fba3-4fe6-bd45-d19f89cb1808
# ╟─8c8b514e-8478-4b2b-b062-56832115c670
# ╟─93dd97e6-0d37-4d94-a3f6-c63dc856fa66
# ╟─d35f0e8b-6634-412c-b5f3-ffd11246276c
# ╠═a6a56523-90c9-40d2-9b68-26e20c1a5527
# ╟─920d94cd-bfb5-4c02-baa3-f346d5c95e2e
# ╟─658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
# ╟─f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
# ╟─9a670af7-cc20-446d-bf22-4e833cc9d854
# ╟─f6949520-d10f-4bae-8d41-2ec880ac7484
# ╟─9bef7690-0db3-4ba5-be77-0933ceb6215e
# ╟─c872d563-421c-4581-a8fa-a02cee58bc85
# ╟─4d50d263-eca0-48ad-b32c-9b767cc57914
# ╟─e4b13e58-2d54-47df-b865-95ae2946756a
# ╟─9c05cae5-af20-4f63-99c9-86032358ffd3
# ╟─d2e5f60d-199a-41f5-ba5d-d21ab2030fb8
# ╟─6293fde0-3cfc-4d0d-bed6-74caa54b6ead
# ╟─9a9b3942-72f2-4c9e-88a5-af927634468c
# ╟─1ff198ea-afd5-4acc-bb67-019051ff149b
# ╟─44ece9ce-f9f1-46f3-90c6-cb0502c92c67
# ╟─5d8d34bb-c207-40fc-ab10-c579e7e2d04c
# ╟─43d68541-84a5-4a63-9d8f-43783cc27ccc
# ╠═0fd29811-9e17-4c97-b9b7-ec9cc51b435f
# ╟─90a47e0b-b911-4728-80b5-6ed74607833d
# ╟─5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
# ╟─a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
# ╟─a7b6ecbd-1407-44dd-809e-33311970af12
# ╟─b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
# ╟─e0e97839-884a-49ed-bee4-f1f2ace5f5e0
# ╠═bcdd60b8-e0d8-4a70-88d6-725269447c9b
# ╟─9de99f4a-9970-4be1-9e16-e64ed4e10277
# ╟─518e7077-d61b-4f60-987f-d556e3eb1d0d
# ╠═1337513f-995f-4dfa-827d-797a5d2e5e1a
# ╟─f5e789b2-a62e-4818-90c3-76f39ea11aaa
# ╟─efa7736c-22c0-410e-94da-1df315f22bbf
# ╠═e9df3afb-fa04-440f-9664-3496da85696b
# ╟─58b7267d-491d-40f0-b4ba-27ed0c9cc855
# ╟─ac76b646-7c28-4384-9f04-5e4de5df154f
# ╠═83a14158-33d1-4f16-85e1-2726c8fbbdfc
# ╟─4b31dca2-0195-4899-8a3a-e9772fabf495
# ╟─79e0deab-1e36-4863-ad10-187ed8555c72
# ╟─66d385ba-9c6e-4378-b4e0-e54a4df346a5
# ╟─db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
# ╟─eb3a6009-e181-443c-bb77-021e867030e4
# ╠═521f5ffa-2c22-44c5-8bdb-67410431ca2e
# ╟─842bf89d-45eb-462d-ba74-ca260a8b177d
# ╟─f9b35e98-347f-4ebd-a690-790c7b0e03d8
# ╟─80fa8831-924f-4093-a89c-bf8fc440da6b
# ╟─4a3630ca-c8dd-4e81-8ee2-bb0fc6b01a93
# ╟─e2b38ec3-2de9-49f1-b29a-e746014e4fe1
# ╠═cfc02683-589b-44dc-a126-257703ed5f85
# ╠═4caa7341-8750-44ec-ba4e-4d425836996d
# ╟─d56a9e90-b0f6-4bde-8b3b-ebf1d962f6b4
# ╟─9852e80b-1f8d-445e-96bf-f7e071d6715c
