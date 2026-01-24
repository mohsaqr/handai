#' @title Special Node Shapes
#' @description Special node shape drawing functions (ellipse, heart, star, pie).
#' @name shapes-special
#' @keywords internal
NULL

#' Draw Ellipse Node
#' @keywords internal
draw_ellipse <- function(x, y, size, fill, border_color, border_width,
                         alpha = 1, aspect = 0.6, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)

  # Ellipse as polygon approximation
  n_points <- 50
  angles <- seq(0, 2*pi, length.out = n_points + 1)[-1]
  xs <- x + size * cos(angles)
  ys <- y + size * aspect * sin(angles)

  grid::polygonGrob(
    x = grid::unit(xs, "npc"),
    y = grid::unit(ys, "npc"),
    gp = grid::gpar(
      fill = fill_col,
      col = border_col,
      lwd = border_width
    )
  )
}

#' Draw Heart Node
#' @keywords internal
draw_heart <- function(x, y, size, fill, border_color, border_width,
                       alpha = 1, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)

  # Heart shape using parametric equations
  n_points <- 100
  t <- seq(0, 2*pi, length.out = n_points)

  # Heart parametric equations
  hx <- 16 * sin(t)^3
  hy <- 13 * cos(t) - 5 * cos(2*t) - 2 * cos(3*t) - cos(4*t)

  # Normalize and scale
  hx <- hx / max(abs(hx))
  hy <- hy / max(abs(hy))

  xs <- x + size * 0.8 * hx
  ys <- y + size * 0.8 * hy

  grid::polygonGrob(
    x = grid::unit(xs, "npc"),
    y = grid::unit(ys, "npc"),
    gp = grid::gpar(
      fill = fill_col,
      col = border_col,
      lwd = border_width
    )
  )
}

#' Draw Star Node
#' @keywords internal
draw_star <- function(x, y, size, fill, border_color, border_width,
                      alpha = 1, n_points = 5, inner_ratio = 0.4, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)

  # Alternating outer and inner points
  n_vertices <- n_points * 2
  angles <- seq(pi/2, pi/2 + 2*pi * (1 - 1/n_vertices), length.out = n_vertices)
  radii <- rep(c(size, size * inner_ratio), n_points)

  xs <- x + radii * cos(angles)
  ys <- y + radii * sin(angles)

  grid::polygonGrob(
    x = grid::unit(xs, "npc"),
    y = grid::unit(ys, "npc"),
    gp = grid::gpar(
      fill = fill_col,
      col = border_col,
      lwd = border_width
    )
  )
}

#' Draw Pie Node
#'
#' Draw a pie chart node with multiple segments.
#'
#' @param pie_border_width Border width for pie segments (optional, defaults to border_width * 0.5).
#' @param default_color Fallback color when colors is NULL and there's a single segment.
#' @keywords internal
draw_pie <- function(x, y, size, fill, border_color, border_width,
                     alpha = 1, values = NULL, colors = NULL,
                     pie_border_width = NULL, default_color = NULL, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)

  # Use specific pie_border_width if provided, else default
  segment_border <- if (!is.null(pie_border_width)) pie_border_width else border_width * 0.5

  # If no values, draw a simple circle
  if (is.null(values) || length(values) <= 1) {
    # Use default_color if provided
    actual_fill <- if (!is.null(default_color)) default_color else fill
    return(draw_circle(x, y, size, actual_fill, border_color, border_width, alpha, ...))
  }

  # Normalize values to proportions
  props <- values / sum(values)

  # Default colors if not provided
  if (is.null(colors)) {
    if (!is.null(default_color) && length(values) == 1) {
      colors <- adjust_alpha(default_color, alpha)
    } else {
      colors <- grDevices::rainbow(length(values), alpha = alpha)
    }
  } else {
    colors <- sapply(colors, adjust_alpha, alpha = alpha)
  }

  # Create pie slices
  grobs <- list()
  start_angle <- pi/2

  for (i in seq_along(props)) {
    end_angle <- start_angle - 2 * pi * props[i]

    # Create arc
    n_points <- max(20, ceiling(50 * props[i]))
    angles <- seq(start_angle, end_angle, length.out = n_points)

    xs <- c(x, x + size * cos(angles), x)
    ys <- c(y, y + size * sin(angles), y)

    grobs[[i]] <- grid::polygonGrob(
      x = grid::unit(xs, "npc"),
      y = grid::unit(ys, "npc"),
      gp = grid::gpar(
        fill = colors[i],
        col = border_col,
        lwd = segment_border
      )
    )

    start_angle <- end_angle
  }

  # Add outer border
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"),
    y = grid::unit(y, "npc"),
    r = grid::unit(size, "npc"),
    gp = grid::gpar(
      fill = NA,
      col = border_col,
      lwd = border_width
    )
  )

  do.call(grid::gList, grobs)
}

#' Draw Donut Node
#'
#' Draw a donut chart node. If given a single value (0-1), shows that proportion
#' filled with the fill color, and the remainder in the background color.
#' If given multiple values, works like a pie chart with a hole.
#'
#' @param donut_border_width Border width for the donut ring (optional, defaults to border_width).
#' @keywords internal
draw_donut <- function(x, y, size, fill, border_color, border_width,
                       alpha = 1, values = NULL, colors = NULL,
                       inner_ratio = 0.5, bg_color = "gray90",
                       show_value = TRUE, value_size = 8, value_color = "black",
                       donut_border_width = NULL, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)
  bg_col <- adjust_alpha(bg_color, alpha)

  # Use specific donut_border_width if provided, else default to border_width
  ring_border <- if (!is.null(donut_border_width)) donut_border_width else border_width

  outer_r <- size
  inner_r <- size * inner_ratio

  grobs <- list()
  center_value <- NULL

  # Use symbols() approach - convert to grob after drawing
  # This ensures proper aspect ratio handling

  # Get viewport dimensions to match circleGrob's radius calculation
  # circleGrob with NPC units uses min(width, height) as reference
  vp_width <- grid::convertWidth(grid::unit(1, "npc"), "inches", valueOnly = TRUE)
  vp_height <- grid::convertHeight(grid::unit(1, "npc"), "inches", valueOnly = TRUE)

  # Calculate scaling factors to match circleGrob behavior
  min_dim <- min(vp_width, vp_height)
  x_scale <- min_dim / vp_width
  y_scale <- min_dim / vp_height

  # Helper function to create pie wedge coordinates for a ring segment
  # Uses same scaling as circleGrob for perfect alignment
  make_ring_coords <- function(start_ang, end_ang, outer_radius, inner_radius, cx, cy, n_pts = 100) {
    angles <- seq(start_ang, end_ang, length.out = n_pts)

    # Apply same scaling as circleGrob
    outer_x <- cx + (outer_radius * x_scale) * cos(angles)
    outer_y <- cy + (outer_radius * y_scale) * sin(angles)

    inner_x <- cx + (inner_radius * x_scale) * cos(rev(angles))
    inner_y <- cy + (inner_radius * y_scale) * sin(rev(angles))

    list(x = c(outer_x, inner_x), y = c(outer_y, inner_y))
  }

  # Handle single value case
  if (is.null(values) || length(values) == 1) {
    prop <- if (is.null(values)) 1 else values[1]
    prop <- max(0, min(1, prop))
    center_value <- prop

    # Inset factor to keep fill inside border
    inset <- 0.97

    # 1. Draw background ring (full circle) - slightly inside the border
    bg_coords <- make_ring_coords(0, 2 * pi, outer_r * inset, inner_r / inset, x, y, 200)
    grobs[[length(grobs) + 1]] <- grid::polygonGrob(
      x = grid::unit(bg_coords$x, "npc"),
      y = grid::unit(bg_coords$y, "npc"),
      gp = grid::gpar(fill = bg_col, col = NA)
    )

    # 2. Draw filled portion (from 12 o'clock clockwise)
    if (prop > 0) {
      start_ang <- pi / 2
      end_ang <- pi / 2 - 2 * pi * prop
      n_pts <- max(100, ceiling(300 * prop))
      fill_coords <- make_ring_coords(start_ang, end_ang, outer_r * inset, inner_r / inset, x, y, n_pts)
      grobs[[length(grobs) + 1]] <- grid::polygonGrob(
        x = grid::unit(fill_coords$x, "npc"),
        y = grid::unit(fill_coords$y, "npc"),
        gp = grid::gpar(fill = fill_col, col = NA)
      )
    }

    # 3. Fill inner hole with white
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(inner_r, "npc"),
      gp = grid::gpar(fill = "white", col = NA)
    )

    # 4. Redraw borders for clean edges
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(outer_r, "npc"),
      gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
    )
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(inner_r, "npc"),
      gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
    )

  } else {
    # Multiple values: donut with segments
    props <- values / sum(values)

    if (is.null(colors)) {
      colors <- grDevices::rainbow(length(values), alpha = alpha)
    } else {
      colors <- sapply(colors, adjust_alpha, alpha = alpha)
    }

    # Inset factor to keep fill inside border
    inset <- 0.97

    # Draw arc segments
    start_ang <- pi / 2
    for (i in seq_along(props)) {
      end_ang <- start_ang - 2 * pi * props[i]
      n_pts <- max(50, ceiling(150 * props[i]))
      seg_coords <- make_ring_coords(start_ang, end_ang, outer_r * inset, inner_r / inset, x, y, n_pts)
      grobs[[length(grobs) + 1]] <- grid::polygonGrob(
        x = grid::unit(seg_coords$x, "npc"),
        y = grid::unit(seg_coords$y, "npc"),
        gp = grid::gpar(fill = colors[i], col = NA)
      )
      start_ang <- end_ang
    }

    # Fill inner hole with white
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(inner_r, "npc"),
      gp = grid::gpar(fill = "white", col = NA)
    )

    # Draw borders
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(outer_r, "npc"),
      gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
    )
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(inner_r, "npc"),
      gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
    )
  }

  # Add outer border
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"),
    y = grid::unit(y, "npc"),
    r = grid::unit(outer_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )

  # Add inner border
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"),
    y = grid::unit(y, "npc"),
    r = grid::unit(inner_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )

  # Add value text in center (for single value donut)
  if (show_value && !is.null(center_value)) {
    grobs[[length(grobs) + 1]] <- grid::textGrob(
      label = round(center_value, 2),
      x = grid::unit(x, "npc"),
      y = grid::unit(y, "npc"),
      gp = grid::gpar(fontsize = value_size, col = value_color, fontface = "bold")
    )
  }

  do.call(grid::gList, grobs)
}

#' Draw Donut with Inner Pie Node
#'
#' Draw a node with an outer donut ring showing a proportion and an inner
#' pie chart with multiple segments.
#'
#' @param pie_border_width Border width for pie segments (optional).
#' @param donut_border_width Border width for donut ring (optional).
#' @keywords internal
draw_donut_pie <- function(x, y, size, fill, border_color, border_width,
                           alpha = 1, donut_value = NULL, pie_values = NULL,
                           pie_colors = NULL, inner_ratio = 0.5,
                           bg_color = "gray90", pie_border_width = NULL,
                           donut_border_width = NULL, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)
  bg_col <- adjust_alpha(bg_color, alpha)

  # Use specific border widths if provided
  ring_border <- if (!is.null(donut_border_width)) donut_border_width else border_width
  pie_segment_border <- if (!is.null(pie_border_width)) pie_border_width else border_width * 0.5

  outer_r <- size
  inner_r <- size * inner_ratio

  grobs <- list()

  # Get viewport dimensions for aspect ratio correction
 vp_width <- grid::convertWidth(grid::unit(1, "npc"), "inches", valueOnly = TRUE)
  vp_height <- grid::convertHeight(grid::unit(1, "npc"), "inches", valueOnly = TRUE)
  min_dim <- min(vp_width, vp_height)
  x_scale <- min_dim / vp_width
  y_scale <- min_dim / vp_height

  # Helper function for ring coordinates
  make_ring_coords <- function(start_ang, end_ang, outer_radius, inner_radius, cx, cy, n_pts = 100) {
    angles <- seq(start_ang, end_ang, length.out = n_pts)
    outer_x <- cx + (outer_radius * x_scale) * cos(angles)
    outer_y <- cy + (outer_radius * y_scale) * sin(angles)
    inner_x <- cx + (inner_radius * x_scale) * cos(rev(angles))
    inner_y <- cy + (inner_radius * y_scale) * sin(rev(angles))
    list(x = c(outer_x, inner_x), y = c(outer_y, inner_y))
  }

  # Helper function for pie slice coordinates
  make_pie_coords <- function(start_ang, end_ang, radius, cx, cy, n_pts = 50) {
    angles <- seq(start_ang, end_ang, length.out = n_pts)
    xs <- c(cx, cx + (radius * x_scale) * cos(angles), cx)
    ys <- c(cy, cy + (radius * y_scale) * sin(angles), cy)
    list(x = xs, y = ys)
  }

  inset <- 0.97

  # 1. Draw outer donut ring (background)
  bg_coords <- make_ring_coords(0, 2 * pi, outer_r * inset, inner_r / inset, x, y, 200)
  grobs[[length(grobs) + 1]] <- grid::polygonGrob(
    x = grid::unit(bg_coords$x, "npc"),
    y = grid::unit(bg_coords$y, "npc"),
    gp = grid::gpar(fill = bg_col, col = NA)
  )

  # 2. Draw donut filled portion (if donut_value provided)
  donut_prop <- if (is.null(donut_value)) 1 else max(0, min(1, donut_value))
  if (donut_prop > 0) {
    start_ang <- pi / 2
    end_ang <- pi / 2 - 2 * pi * donut_prop
    n_pts <- max(100, ceiling(300 * donut_prop))
    fill_coords <- make_ring_coords(start_ang, end_ang, outer_r * inset, inner_r / inset, x, y, n_pts)
    grobs[[length(grobs) + 1]] <- grid::polygonGrob(
      x = grid::unit(fill_coords$x, "npc"),
      y = grid::unit(fill_coords$y, "npc"),
      gp = grid::gpar(fill = fill_col, col = NA)
    )
  }

  # 3. Draw inner pie chart
  pie_radius <- inner_r * 0.95
  if (!is.null(pie_values) && length(pie_values) > 0) {
    props <- pie_values / sum(pie_values)

    if (is.null(pie_colors)) {
      pie_colors <- grDevices::rainbow(length(pie_values), alpha = alpha)
    } else {
      pie_colors <- sapply(pie_colors, adjust_alpha, alpha = alpha)
      pie_colors <- rep(pie_colors, length.out = length(pie_values))
    }

    start_ang <- pi / 2
    for (i in seq_along(props)) {
      end_ang <- start_ang - 2 * pi * props[i]
      n_pts <- max(30, ceiling(100 * props[i]))
      pie_coords <- make_pie_coords(start_ang, end_ang, pie_radius, x, y, n_pts)
      grobs[[length(grobs) + 1]] <- grid::polygonGrob(
        x = grid::unit(pie_coords$x, "npc"),
        y = grid::unit(pie_coords$y, "npc"),
        gp = grid::gpar(fill = pie_colors[i], col = NA)
      )
      start_ang <- end_ang
    }
  } else {
    # No pie values - fill inner with white
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(pie_radius, "npc"),
      gp = grid::gpar(fill = "white", col = NA)
    )
  }

  # 4. Draw borders
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
    r = grid::unit(outer_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
    r = grid::unit(inner_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )

  do.call(grid::gList, grobs)
}

#' Draw Double Donut with Inner Pie Node
#'
#' Draw a node with two concentric donut rings and an optional inner pie chart.
#' From outside to inside: outer donut ring, inner donut ring, center pie.
#'
#' @param pie_border_width Border width for pie segments (optional).
#' @param donut_border_width Border width for donut rings (optional).
#' @keywords internal
draw_double_donut_pie <- function(x, y, size, fill, border_color, border_width,
                                  alpha = 1, donut_values = NULL, donut_colors = NULL,
                                  donut2_values = NULL, donut2_colors = NULL,
                                  pie_values = NULL, pie_colors = NULL,
                                  outer_inner_ratio = 0.7, inner_inner_ratio = 0.4,
                                  bg_color = "gray90", pie_border_width = NULL,
                                  donut_border_width = NULL, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)
  bg_col <- adjust_alpha(bg_color, alpha)

  # Use specific border widths if provided
  ring_border <- if (!is.null(donut_border_width)) donut_border_width else border_width
  pie_segment_border <- if (!is.null(pie_border_width)) pie_border_width else border_width * 0.5

  # Define radii for the three layers
  outer_r <- size
  mid_r <- size * outer_inner_ratio
  inner_r <- size * inner_inner_ratio

  grobs <- list()

  # Get viewport dimensions for aspect ratio correction
  vp_width <- grid::convertWidth(grid::unit(1, "npc"), "inches", valueOnly = TRUE)
  vp_height <- grid::convertHeight(grid::unit(1, "npc"), "inches", valueOnly = TRUE)
  min_dim <- min(vp_width, vp_height)
  x_scale <- min_dim / vp_width
  y_scale <- min_dim / vp_height

  # Helper function for ring coordinates
  make_ring_coords <- function(start_ang, end_ang, r_outer, r_inner, cx, cy, n_pts = 100) {
    angles <- seq(start_ang, end_ang, length.out = n_pts)
    outer_x <- cx + (r_outer * x_scale) * cos(angles)
    outer_y <- cy + (r_outer * y_scale) * sin(angles)
    inner_x <- cx + (r_inner * x_scale) * cos(rev(angles))
    inner_y <- cy + (r_inner * y_scale) * sin(rev(angles))
    list(x = c(outer_x, inner_x), y = c(outer_y, inner_y))
  }

  # Helper function for pie slice coordinates
  make_pie_coords <- function(start_ang, end_ang, radius, cx, cy, n_pts = 50) {
    angles <- seq(start_ang, end_ang, length.out = n_pts)
    xs <- c(cx, cx + (radius * x_scale) * cos(angles), cx)
    ys <- c(cy, cy + (radius * y_scale) * sin(angles), cy)
    list(x = xs, y = ys)
  }

  inset <- 0.97

  # Helper to draw donut ring (handles both progress and segmented)
  draw_donut_ring_grid <- function(values, colors, r_outer, r_inner) {
    if (is.null(values)) {
      # Fill with background
      bg_coords <- make_ring_coords(0, 2 * pi, r_outer * inset, r_inner / inset, x, y, 200)
      return(list(grid::polygonGrob(
        x = grid::unit(bg_coords$x, "npc"),
        y = grid::unit(bg_coords$y, "npc"),
        gp = grid::gpar(fill = bg_col, col = NA)
      )))
    }

    ring_grobs <- list()

    if (length(values) == 1) {
      # Progress donut - draw background then filled portion
      bg_coords <- make_ring_coords(0, 2 * pi, r_outer * inset, r_inner / inset, x, y, 200)
      ring_grobs[[length(ring_grobs) + 1]] <- grid::polygonGrob(
        x = grid::unit(bg_coords$x, "npc"),
        y = grid::unit(bg_coords$y, "npc"),
        gp = grid::gpar(fill = bg_col, col = NA)
      )

      prop <- max(0, min(1, values))
      if (prop > 0) {
        fill_c <- if (!is.null(colors)) adjust_alpha(colors[1], alpha) else fill_col
        start_ang <- pi / 2
        end_ang <- pi / 2 - 2 * pi * prop
        n_pts <- max(100, ceiling(300 * prop))
        fill_coords <- make_ring_coords(start_ang, end_ang, r_outer * inset, r_inner / inset, x, y, n_pts)
        ring_grobs[[length(ring_grobs) + 1]] <- grid::polygonGrob(
          x = grid::unit(fill_coords$x, "npc"),
          y = grid::unit(fill_coords$y, "npc"),
          gp = grid::gpar(fill = fill_c, col = NA)
        )
      }
    } else {
      # Segmented donut
      props <- values / sum(values)

      if (is.null(colors)) {
        colors <- grDevices::rainbow(length(values), alpha = alpha)
      } else {
        colors <- sapply(colors, adjust_alpha, alpha = alpha)
        colors <- rep(colors, length.out = length(values))
      }

      start_ang <- pi / 2
      for (i in seq_along(props)) {
        end_ang <- start_ang - 2 * pi * props[i]
        n_pts <- max(50, ceiling(150 * props[i]))
        seg_coords <- make_ring_coords(start_ang, end_ang, r_outer * inset, r_inner / inset, x, y, n_pts)
        ring_grobs[[length(ring_grobs) + 1]] <- grid::polygonGrob(
          x = grid::unit(seg_coords$x, "npc"),
          y = grid::unit(seg_coords$y, "npc"),
          gp = grid::gpar(fill = colors[i], col = NA)
        )
        start_ang <- end_ang
      }
    }

    ring_grobs
  }

  # 1. Draw outer donut ring
  grobs <- c(grobs, draw_donut_ring_grid(donut_values, donut_colors, outer_r, mid_r))

  # 2. Draw inner donut ring
  grobs <- c(grobs, draw_donut_ring_grid(donut2_values, donut2_colors, mid_r, inner_r))

  # 3. Draw center pie (if values provided)
  pie_radius <- inner_r * 0.95
  if (!is.null(pie_values) && length(pie_values) > 0) {
    props <- pie_values / sum(pie_values)

    if (is.null(pie_colors)) {
      pie_colors <- grDevices::rainbow(length(pie_values), alpha = alpha)
    } else {
      pie_colors <- sapply(pie_colors, adjust_alpha, alpha = alpha)
      pie_colors <- rep(pie_colors, length.out = length(pie_values))
    }

    start_ang <- pi / 2
    for (i in seq_along(props)) {
      end_ang <- start_ang - 2 * pi * props[i]
      n_pts <- max(30, ceiling(100 * props[i]))
      pie_coords <- make_pie_coords(start_ang, end_ang, pie_radius, x, y, n_pts)
      grobs[[length(grobs) + 1]] <- grid::polygonGrob(
        x = grid::unit(pie_coords$x, "npc"),
        y = grid::unit(pie_coords$y, "npc"),
        gp = grid::gpar(fill = pie_colors[i], col = NA)
      )
      start_ang <- end_ang
    }
  } else {
    # No pie values - fill inner with white
    grobs[[length(grobs) + 1]] <- grid::circleGrob(
      x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
      r = grid::unit(pie_radius, "npc"),
      gp = grid::gpar(fill = "white", col = NA)
    )
  }

  # 4. Draw all borders
  # Outer border
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
    r = grid::unit(outer_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )
  # Middle border (between outer and inner donut)
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
    r = grid::unit(mid_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )
  # Inner border (between inner donut and pie)
  grobs[[length(grobs) + 1]] <- grid::circleGrob(
    x = grid::unit(x, "npc"), y = grid::unit(y, "npc"),
    r = grid::unit(inner_r, "npc"),
    gp = grid::gpar(fill = NA, col = border_col, lwd = ring_border)
  )

  do.call(grid::gList, grobs)
}

#' Draw Cross/Plus Node
#' @keywords internal
draw_cross <- function(x, y, size, fill, border_color, border_width,
                       alpha = 1, thickness = 0.3, ...) {
  fill_col <- adjust_alpha(fill, alpha)
  border_col <- adjust_alpha(border_color, alpha)

  # Cross shape
  t <- size * thickness  # Half thickness
  s <- size  # Half size

  # Horizontal bar
  xs1 <- c(x - s, x + s, x + s, x - s)
  ys1 <- c(y - t, y - t, y + t, y + t)

  # Vertical bar
  xs2 <- c(x - t, x + t, x + t, x - t)
  ys2 <- c(y - s, y - s, y + s, y + s)

  grid::gList(
    grid::polygonGrob(
      x = grid::unit(xs1, "npc"),
      y = grid::unit(ys1, "npc"),
      gp = grid::gpar(fill = fill_col, col = border_col, lwd = border_width)
    ),
    grid::polygonGrob(
      x = grid::unit(xs2, "npc"),
      y = grid::unit(ys2, "npc"),
      gp = grid::gpar(fill = fill_col, col = border_col, lwd = border_width)
    )
  )
}
