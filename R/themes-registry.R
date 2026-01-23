#' @title Theme Registry Functions
#' @description Functions for registering built-in themes.
#' @name themes-registry
#' @keywords internal
NULL

#' Register Built-in Themes
#'
#' Register all built-in themes.
#'
#' @keywords internal
register_builtin_themes <- function() {
  register_theme("classic", theme_sonnet_classic())
  register_theme("colorblind", theme_sonnet_colorblind())
  register_theme("gray", theme_sonnet_gray())
  register_theme("grey", theme_sonnet_gray())  # Alias
  register_theme("dark", theme_sonnet_dark())
  register_theme("minimal", theme_sonnet_minimal())
  register_theme("viridis", theme_sonnet_viridis())
  register_theme("nature", theme_sonnet_nature())
}
