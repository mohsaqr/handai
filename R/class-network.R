#' @title SonnetNetwork R6 Class
#'
#' @description
#' Core class representing a network for visualization. Stores nodes, edges,
#' layout coordinates, and aesthetic mappings.
#'
#' @export
#' @examples
#' # Create network from adjacency matrix
#' adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
#' net <- SonnetNetwork$new(adj)
#'
#' # Access properties
#' net$n_nodes
#' net$n_edges
#' net$is_directed
SonnetNetwork <- R6::R6Class(
  "SonnetNetwork",
  public = list(
    #' @description Create a new SonnetNetwork object.
    #' @param input Network input (matrix, edge list, or igraph object).
    #' @param directed Logical. Force directed interpretation. NULL for auto-detect.
    #' @param node_labels Character vector of node labels.
    #' @return A new SonnetNetwork object.
    initialize = function(input = NULL, directed = NULL, node_labels = NULL) {
      if (!is.null(input)) {
        parsed <- parse_input(input, directed = directed)
        private$.nodes <- parsed$nodes
        private$.edges <- parsed$edges
        private$.directed <- parsed$directed
        private$.weights <- parsed$weights

        # Set node labels
        if (!is.null(node_labels)) {
          if (length(node_labels) != nrow(private$.nodes)) {
            stop("node_labels length must match number of nodes", call. = FALSE)
          }
          private$.nodes$label <- node_labels
        }
      }

      # Initialize aesthetics with defaults
      private$.node_aes <- list(
        size = 0.05,
        shape = "circle",
        fill = "#4A90D9",
        border_color = "#2C5AA0",
        border_width = 1,
        alpha = 1,
        label_size = 10,
        label_color = "black",
        label_position = "center"
      )

      private$.edge_aes <- list(
        width = 1,
        color = "gray50",
        positive_color = "#2E7D32",
        negative_color = "#C62828",
        alpha = 0.8,
        style = "solid",
        curvature = 0,
        arrow_size = 0.015,
        show_arrows = NULL  # NULL = auto (TRUE if directed)
      )

      invisible(self)
    },

    #' @description Clone the network with optional modifications.
    #' @return A new SonnetNetwork object.
    clone_network = function() {
      new_net <- SonnetNetwork$new()
      new_net$set_nodes(private$.nodes)
      new_net$set_edges(private$.edges)
      new_net$set_directed(private$.directed)
      new_net$set_weights(private$.weights)
      new_net$set_layout_coords(private$.layout)
      new_net$set_node_aes(private$.node_aes)
      new_net$set_edge_aes(private$.edge_aes)
      new_net$set_theme(private$.theme)
      if (!is.null(private$.layout_info)) {
        new_net$set_layout_info(private$.layout_info)
      }
      if (!is.null(private$.plot_params)) {
        new_net$set_plot_params(private$.plot_params)
      }
      new_net
    },

    #' @description Set nodes data frame.
    #' @param nodes Data frame with node information.
    set_nodes = function(nodes) {
      private$.nodes <- nodes
      invisible(self)
    },

    #' @description Set edges data frame.
    #' @param edges Data frame with edge information.
    set_edges = function(edges) {
      private$.edges <- edges
      invisible(self)
    },

    #' @description Set directed flag.
    #' @param directed Logical.
    set_directed = function(directed) {
      private$.directed <- directed
      invisible(self)
    },

    #' @description Set edge weights.
    #' @param weights Numeric vector of weights.
    set_weights = function(weights) {
      private$.weights <- weights
      invisible(self)
    },

    #' @description Set layout coordinates.
    #' @param coords Matrix or data frame with x, y columns.
    set_layout_coords = function(coords) {
      if (!is.null(coords)) {
        if (is.matrix(coords)) {
          coords <- as.data.frame(coords)
          if (is.null(names(coords))) {
            names(coords) <- c("x", "y")
          }
        }
        private$.layout <- coords
        # Update node positions
        if (!is.null(private$.nodes) && nrow(private$.nodes) == nrow(coords)) {
          private$.nodes$x <- coords$x
          private$.nodes$y <- coords$y
        }
      }
      invisible(self)
    },

    #' @description Set node aesthetics.
    #' @param aes List of aesthetic parameters.
    set_node_aes = function(aes) {
      private$.node_aes <- utils::modifyList(private$.node_aes, aes)
      invisible(self)
    },

    #' @description Set edge aesthetics.
    #' @param aes List of aesthetic parameters.
    set_edge_aes = function(aes) {
      private$.edge_aes <- utils::modifyList(private$.edge_aes, aes)
      invisible(self)
    },

    #' @description Set theme.
    #' @param theme SonnetTheme object or theme name.
    set_theme = function(theme) {
      private$.theme <- theme
      invisible(self)
    },

    #' @description Get nodes data frame.
    #' @return Data frame with node information.
    get_nodes = function() {
      private$.nodes
    },

    #' @description Get edges data frame.
    #' @return Data frame with edge information.
    get_edges = function() {
      private$.edges
    },

    #' @description Get layout coordinates.
    #' @return Data frame with x, y coordinates.
    get_layout = function() {
      private$.layout
    },

    #' @description Get node aesthetics.
    #' @return List of node aesthetic parameters.
    get_node_aes = function() {
      private$.node_aes
    },

    #' @description Get edge aesthetics.
    #' @return List of edge aesthetic parameters.
    get_edge_aes = function() {
      private$.edge_aes
    },

    #' @description Get theme.
    #' @return SonnetTheme object.
    get_theme = function() {
      private$.theme
    },

    #' @description Set layout info.
    #' @param info List with layout information (name, seed, etc.).
    set_layout_info = function(info) {
      private$.layout_info <- info
      invisible(self)
    },

    #' @description Get layout info.
    #' @return List with layout information.
    get_layout_info = function() {
      private$.layout_info
    },

    #' @description Set plot parameters.
    #' @param params List of all plot parameters used.
    set_plot_params = function(params) {
      private$.plot_params <- params
      invisible(self)
    },

    #' @description Get plot parameters.
    #' @return List of plot parameters.
    get_plot_params = function() {
      private$.plot_params
    },

    #' @description Print network summary.
    print = function() {
      cat("SonnetNetwork\n")
      cat("  Nodes:", self$n_nodes, "\n")
      cat("  Edges:", self$n_edges, "\n")
      cat("  Directed:", self$is_directed, "\n")
      cat("  Layout:", if (is.null(private$.layout)) "none" else "set", "\n")
      invisible(self)
    }
  ),

  active = list(
    #' @field n_nodes Number of nodes in the network.
    n_nodes = function() {
      if (is.null(private$.nodes)) 0L else nrow(private$.nodes)
    },

    #' @field n_edges Number of edges in the network.
    n_edges = function() {
      if (is.null(private$.edges)) 0L else nrow(private$.edges)
    },

    #' @field is_directed Whether the network is directed.
    is_directed = function() {
      private$.directed
    },

    #' @field has_weights Whether edges have weights.
    has_weights = function() {
      !is.null(private$.weights) && any(private$.weights != 1)
    },

    #' @field node_labels Vector of node labels.
    node_labels = function() {
      if (is.null(private$.nodes)) NULL else private$.nodes$label
    }
  ),

  private = list(
    .nodes = NULL,
    .edges = NULL,
    .directed = FALSE,
    .weights = NULL,
    .layout = NULL,
    .node_aes = NULL,
    .edge_aes = NULL,
    .theme = NULL,
    .layout_info = NULL,
    .plot_params = NULL
  )
)

#' @title Check if object is a SonnetNetwork
#' @param x Object to check.
#' @return Logical.
#' @keywords internal
is_sonnet_network <- function(x) {
  inherits(x, "SonnetNetwork")
}

#' @title Create sonnet_network S3 class wrapper
#' @param network SonnetNetwork R6 object.
#' @return Object with sonnet_network class.
#' @keywords internal
as_sonnet_network <- function(network) {
  obj <- structure(
    list(network = network),
    class = c("sonnet_network", "list")
  )
  # Add direct access to layout and plot params
  obj$layout <- network$get_layout()
  obj$layout_info <- network$get_layout_info()
  obj$plot_params <- network$get_plot_params()
  obj$nodes <- network$get_nodes()
  obj$edges <- network$get_edges()
  obj$node_aes <- network$get_node_aes()
  obj$edge_aes <- network$get_edge_aes()
  obj
}
