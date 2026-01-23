# Sonnet <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->
[![R-CMD-check](https://github.com/username/Sonnet/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/username/Sonnet/actions/workflows/R-CMD-check.yaml)
[![CRAN status](https://www.r-pkg.org/badges/version/Sonnet)](https://CRAN.R-project.org/package=Sonnet)
<!-- badges: end -->

**Sonnet** is a modern, extensible R package for network visualization. It provides high-quality static network plots with a pipe-friendly API, accepting adjacency matrices, edge lists, or igraph objects as input.

## Features

- **Multiple input formats**: Adjacency matrices, edge lists, igraph objects
- **Flexible layouts**: Circle, spring (Fruchterman-Reingold), groups, grid, and more
- **Rich aesthetics**: Customizable node shapes, sizes, colors, and edge styles
- **Built-in themes**: Classic, colorblind-friendly, dark, minimal, and more
- **ggplot2 integration**: Convert to ggplot2 for further customization
- **Publication-ready output**: Export to PDF, PNG, SVG

## Installation

```r
# Install from CRAN (when available)
install.packages("Sonnet")

# Or install the development version from GitHub
# install.packages("devtools")
devtools::install_github("username/Sonnet")
```

## Quick Start

```r
library(Sonnet)

# Create a simple network from an adjacency matrix
adj <- matrix(c(
  0, 1, 1, 0,
  1, 0, 1, 1,
  1, 1, 0, 1,
  0, 1, 1, 0
), nrow = 4, byrow = TRUE)

# Basic plot
sonnet(adj)

# Customized plot with pipe syntax
adj |>
  sonnet(layout = "circle") |>
  sn_nodes(size = 0.08, fill = "steelblue") |>
  sn_edges(width = 1.5, color = "gray50") |>
  sn_theme("minimal") |>
  sn_render()
```

## Main Functions

| Function | Purpose |
|----------|---------|
| `sonnet()` | Create network from matrix, edge list, or igraph |
| `sn_layout()` | Apply layout algorithm |
| `sn_nodes()` | Customize node aesthetics |
| `sn_edges()` | Customize edge aesthetics |
| `sn_theme()` | Apply visual theme |
| `sn_palette()` | Apply color palette |
| `sn_render()` | Render to current device |
| `sn_save()` | Save to file (PDF, PNG, SVG) |
| `sn_ggplot()` | Convert to ggplot2 object |

## Layouts

```r
# Circular layout
sonnet(adj, layout = "circle")

# Force-directed (Fruchterman-Reingold)
sonnet(adj, layout = "spring")

# Group-based layout
groups <- c(1, 1, 2, 2)
sonnet(adj) |> sn_layout("groups", groups = groups)

# Custom coordinates
coords <- matrix(c(0, 0, 1, 0, 1, 1, 0, 1), ncol = 2, byrow = TRUE)
sonnet(adj) |> sn_layout(coords)
```

## Node Styles

```r
# Available shapes
# circle, square, triangle, diamond, pentagon, hexagon,
# ellipse, star, heart, pie, cross

adj |>
  sonnet() |>
  sn_nodes(
    size = 0.06,
    shape = "diamond",
    fill = "coral",
    border_color = "darkred",
    border_width = 2,
    alpha = 0.8
  )

# Per-node customization
adj |>
  sonnet() |>
  sn_nodes(
    size = c(0.04, 0.06, 0.08, 0.05),
    fill = c("red", "green", "blue", "orange")
  )
```

## Edge Styles

```r
# Weighted edges with color coding
weighted_adj <- matrix(c(
  0, 0.8, -0.5, 0,
  0.8, 0, 0.3, -0.7,
  -0.5, 0.3, 0, 0.6,
  0, -0.7, 0.6, 0
), nrow = 4, byrow = TRUE)

weighted_adj |>
  sonnet() |>
  sn_edges(
    width = "weight",        # Scale width by weight
    color = "weight",        # Green = positive, red = negative
    positive_color = "darkgreen",
    negative_color = "darkred"
  )

# Curved edges for directed graphs
sonnet(adj, directed = TRUE) |>
  sn_edges(curvature = 0.2, arrow_size = 0.02)
```

## Themes

```r
# Built-in themes
sonnet(adj) |> sn_theme("classic")
sonnet(adj) |> sn_theme("colorblind")
sonnet(adj) |> sn_theme("dark")
sonnet(adj) |> sn_theme("minimal")
sonnet(adj) |> sn_theme("gray")
```

## ggplot2 Integration

```r
library(ggplot2)

# Convert to ggplot2 for further customization
p <- adj |>
  sonnet() |>
  sn_nodes(fill = "steelblue") |>
  sn_ggplot()

# Add ggplot2 elements
p +
  labs(title = "My Network", subtitle = "Created with Sonnet") +
  theme(plot.title = element_text(hjust = 0.5))
```

## Saving Plots

```r
net <- sonnet(adj) |>
  sn_nodes(fill = "steelblue") |>
  sn_theme("minimal")

# Save as PDF
sn_save(net, "network.pdf", width = 8, height = 8)

# Save as PNG
sn_save(net, "network.png", width = 8, height = 8, dpi = 300)

# Save as SVG
sn_save(net, "network.svg", width = 8, height = 8)
```

## Working with igraph

```r
library(igraph)

# Create igraph object
g <- make_ring(10) |>
  set_vertex_attr("name", value = LETTERS[1:10])

# Visualize with Sonnet
sonnet(g) |>
  sn_nodes(fill = "lightblue") |>
  sn_render()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.
