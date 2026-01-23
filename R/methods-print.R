#' @title Print Methods
#' @description S3 print methods for Sonnet objects.
#' @name methods-print
NULL

#' Print sonnet_network Object
#'
#' @param x A sonnet_network object.
#' @param ... Ignored.
#' @return Invisible x.
#' @export
print.sonnet_network <- function(x, ...) {
  net <- x$network
  cat("Sonnet Network\n")
  cat("==============\n")
  cat("Nodes:", net$n_nodes, "\n")
  cat("Edges:", net$n_edges, "\n")
  cat("Directed:", net$is_directed, "\n")
  cat("Weighted:", net$has_weights, "\n")

  layout <- net$get_layout()
  cat("Layout:", if (is.null(layout)) "not computed" else "computed", "\n")

  theme <- net$get_theme()
  cat("Theme:", if (is.null(theme)) "none" else theme$name, "\n")

  cat("\nUse plot() or sn_render() to visualize\n")
  cat("Use sn_ggplot() to convert to ggplot2\n")

  invisible(x)
}
