#' @title Base R Node Rendering
#' @description Node drawing functions for splot() using base R graphics.
#' @name splot-nodes
#' @keywords internal
NULL

#' Draw a Single Node
#'
#' Renders a node at the specified position with given aesthetics.
#'
#' @param x,y Node center coordinates.
#' @param size Node radius in user coordinates.
#' @param size2 Secondary size (for ellipse height).
#' @param shape Node shape name.
#' @param col Fill color.
#' @param border.col Border color.
#' @param border.width Border line width.
#' @param ... Additional parameters.
#' @keywords internal
draw_node_base <- function(x, y, size, size2 = NULL, shape = "circle",
                           col = "#4A90D9", border.col = "#2C5AA0",
                           border.width = 1, ...) {
  if (is.null(size2)) size2 <- size

  if (shape == "circle") {
    # Use symbols() for perfect circles
    graphics::symbols(
      x = x, y = y,
      circles = size,
      inches = FALSE,
      add = TRUE,
      fg = border.col,
      bg = col,
      lwd = border.width
    )

  } else if (shape == "square") {
    # Square using rect()
    graphics::rect(
      xleft = x - size,
      ybottom = y - size,
      xright = x + size,
      ytop = y + size,
      col = col,
      border = border.col,
      lwd = border.width
    )

  } else if (shape == "rectangle" || shape == "ellipse") {
    # Use polygon for ellipse/rectangle
    verts <- get_shape_vertices(shape, x, y, size, size2)
    graphics::polygon(
      x = verts$x,
      y = verts$y,
      col = col,
      border = border.col,
      lwd = border.width
    )

  } else {
    # All other shapes via polygon
    verts <- get_shape_vertices(shape, x, y, size, size2, ...)
    graphics::polygon(
      x = verts$x,
      y = verts$y,
      col = col,
      border = border.col,
      lwd = border.width
    )
  }
}

#' Draw Pie Chart Node
#'
#' Renders a node as a pie chart with multiple colored segments.
#'
#' @param x,y Node center coordinates.
#' @param size Node radius.
#' @param values Numeric vector of values (will be normalized to proportions).
#' @param colors Vector of colors for each segment.
#' @param default_color Fallback color when colors is NULL and values length is 1.
#' @param border.col Border color.
#' @param border.width Border line width.
#' @param pie_border.width Border width for pie slice dividers (NULL = use border.width).
#' @keywords internal
draw_pie_node_base <- function(x, y, size, values, colors = NULL,
                               default_color = NULL,
                               border.col = "black", border.width = 1,
                               pie_border.width = NULL) {
  if (is.null(values) || length(values) == 0) {
    return(invisible())
  }

  # Normalize to proportions
  props <- values / sum(values)
  n <- length(props)

  # Default colors - use default_color if provided and single segment
  if (is.null(colors)) {
    if (!is.null(default_color) && n == 1) {
      colors <- default_color
    } else {
      colors <- grDevices::rainbow(n, s = 0.7, v = 0.9)
    }
  }
  colors <- recycle_to_length(colors, n)

  # Use separate pie border width if provided
  slice_border_width <- if (!is.null(pie_border.width)) pie_border.width else border.width

  # Draw slices
  start_angle <- pi / 2  # Start at top
  n_points <- 50

  for (i in seq_len(n)) {
    if (props[i] <= 0) next

    end_angle <- start_angle - 2 * pi * props[i]

    # Create slice polygon
    angles <- seq(start_angle, end_angle, length.out = max(3, ceiling(n_points * props[i])))

    xs <- c(x, x + size * cos(angles), x)
    ys <- c(y, y + size * sin(angles), y)

    graphics::polygon(
      x = xs, y = ys,
      col = colors[i],
      border = NA
    )

    start_angle <- end_angle
  }

  # Draw slice dividers (if more than one slice)
  if (n > 1) {
    start_angle <- pi / 2
    for (i in seq_len(n)) {
      if (props[i] <= 0) next
      end_angle <- start_angle - 2 * pi * props[i]
      # Draw radial line at slice boundary
      graphics::lines(
        x = c(x, x + size * cos(start_angle)),
        y = c(y, y + size * sin(start_angle)),
        col = border.col,
        lwd = slice_border_width
      )
      start_angle <- end_angle
    }
  }

  # Draw border circle
  angles <- seq(0, 2 * pi, length.out = 100)
  graphics::lines(
    x = x + size * cos(angles),
    y = y + size * sin(angles),
    col = border.col,
    lwd = border.width
  )
}

#' Draw Donut Chart Node
#'
#' Renders a node as a donut chart with an inner hole.
#'
#' @param x,y Node center coordinates.
#' @param size Outer radius.
#' @param values Numeric vector of values (or single value 0-1 for progress).
#' @param colors Vector of colors for segments.
#' @param default_color Fallback color when colors is NULL and values length is 1.
#' @param inner_ratio Ratio of inner to outer radius (0-1).
#' @param bg_color Background color for unfilled portion.
#' @param border.col Border color.
#' @param border.width Border line width.
#' @param donut_border.width Border width for donut ring (NULL = use border.width).
#' @param show_value Logical: show value in center?
#' @param value_cex Text size for center value.
#' @param value_col Text color for center value.
#' @keywords internal
draw_donut_node_base <- function(x, y, size, values, colors = NULL,
                                 default_color = NULL,
                                 inner_ratio = 0.5, bg_color = "gray90",
                                 border.col = "black", border.width = 1,
                                 donut_border.width = NULL,
                                 show_value = TRUE, value_cex = 0.8,
                                 value_col = "black") {
  # Use separate donut border width if provided
  ring_border_width <- if (!is.null(donut_border.width)) donut_border.width else border.width
  outer_r <- size
  inner_r <- size * inner_ratio

  n_points <- 100

  # Helper to draw ring segment
  draw_ring_segment <- function(start_ang, end_ang, outer_r, inner_r, col) {
    n_pts <- max(10, ceiling(abs(end_ang - start_ang) / (2 * pi) * n_points))
    angles <- seq(start_ang, end_ang, length.out = n_pts)

    # Outer arc
    outer_x <- x + outer_r * cos(angles)
    outer_y <- y + outer_r * sin(angles)

    # Inner arc (reversed)
    inner_x <- x + inner_r * cos(rev(angles))
    inner_y <- y + inner_r * sin(rev(angles))

    graphics::polygon(
      x = c(outer_x, inner_x),
      y = c(outer_y, inner_y),
      col = col,
      border = NA
    )
  }

  center_value <- NULL

  if (length(values) == 1) {
    # Single value: progress donut
    prop <- max(0, min(1, values))
    center_value <- prop

    # Draw background ring
    draw_ring_segment(0, 2 * pi, outer_r, inner_r, bg_color)

    # Draw filled portion
    if (prop > 0) {
      start_ang <- pi / 2
      end_ang <- pi / 2 - 2 * pi * prop
      fill_col <- if (!is.null(colors)) colors[1] else if (!is.null(default_color)) default_color else "#4A90D9"
      draw_ring_segment(start_ang, end_ang, outer_r, inner_r, fill_col)
    }

  } else {
    # Multiple values: segmented donut
    props <- values / sum(values)
    n <- length(props)

    if (is.null(colors)) {
      if (!is.null(default_color) && n == 1) {
        colors <- default_color
      } else {
        colors <- grDevices::rainbow(n, s = 0.7, v = 0.9)
      }
    }
    colors <- recycle_to_length(colors, n)

    start_ang <- pi / 2
    for (i in seq_len(n)) {
      if (props[i] <= 0) next

      end_ang <- start_ang - 2 * pi * props[i]
      draw_ring_segment(start_ang, end_ang, outer_r, inner_r, colors[i])
      start_ang <- end_ang
    }
  }

  # Fill inner hole with white
  angles <- seq(0, 2 * pi, length.out = n_points)
  graphics::polygon(
    x = x + inner_r * cos(angles),
    y = y + inner_r * sin(angles),
    col = "white",
    border = NA
  )

  # Draw borders
  graphics::lines(
    x = x + outer_r * cos(seq(0, 2*pi, length.out = n_points)),
    y = y + outer_r * sin(seq(0, 2*pi, length.out = n_points)),
    col = border.col,
    lwd = ring_border_width
  )
  graphics::lines(
    x = x + inner_r * cos(seq(0, 2*pi, length.out = n_points)),
    y = y + inner_r * sin(seq(0, 2*pi, length.out = n_points)),
    col = border.col,
    lwd = ring_border_width
  )

  # Show value in center
  if (show_value && !is.null(center_value)) {
    graphics::text(
      x = x, y = y,
      labels = round(center_value, 2),
      cex = value_cex,
      col = value_col,
      font = 2
    )
  }
}

#' Draw Donut with Inner Pie
#'
#' Renders a node with outer donut ring and inner pie chart.
#'
#' @param x,y Node center coordinates.
#' @param size Outer radius.
#' @param donut_value Single value (0-1) for donut progress.
#' @param donut_color Fill color for donut ring.
#' @param pie_values Numeric vector for pie segments.
#' @param pie_colors Vector of colors for pie segments.
#' @param pie_default_color Default color for pie when pie_colors is NULL.
#' @param inner_ratio Ratio of inner to outer radius.
#' @param bg_color Background color.
#' @param border.col Border color.
#' @param border.width Border line width.
#' @param pie_border.width Border width for pie slice dividers (NULL = use border.width * 0.5).
#' @param donut_border.width Border width for donut ring (NULL = use border.width).
#' @keywords internal
draw_donut_pie_node_base <- function(x, y, size, donut_value = 1,
                                     donut_color = "#4A90D9",
                                     pie_values = NULL, pie_colors = NULL,
                                     pie_default_color = NULL,
                                     inner_ratio = 0.5, bg_color = "gray90",
                                     border.col = "black", border.width = 1,
                                     pie_border.width = NULL,
                                     donut_border.width = NULL) {
  # Use separate border widths if provided
  ring_border_width <- if (!is.null(donut_border.width)) donut_border.width else border.width
  pie_slice_border <- if (!is.null(pie_border.width)) pie_border.width else border.width * 0.5
  outer_r <- size
  inner_r <- size * inner_ratio
  n_points <- 100

  # Helper to draw ring segment
  draw_ring_segment <- function(start_ang, end_ang, outer_r, inner_r, col) {
    n_pts <- max(10, ceiling(abs(end_ang - start_ang) / (2 * pi) * n_points))
    angles <- seq(start_ang, end_ang, length.out = n_pts)

    outer_x <- x + outer_r * cos(angles)
    outer_y <- y + outer_r * sin(angles)
    inner_x <- x + inner_r * cos(rev(angles))
    inner_y <- y + inner_r * sin(rev(angles))

    graphics::polygon(
      x = c(outer_x, inner_x),
      y = c(outer_y, inner_y),
      col = col,
      border = NA
    )
  }

  # Draw donut ring background
  draw_ring_segment(0, 2 * pi, outer_r, inner_r, bg_color)

  # Draw donut filled portion
  donut_prop <- max(0, min(1, donut_value))
  if (donut_prop > 0) {
    start_ang <- pi / 2
    end_ang <- pi / 2 - 2 * pi * donut_prop
    draw_ring_segment(start_ang, end_ang, outer_r, inner_r, donut_color)
  }

  # Draw inner pie
  pie_r <- inner_r * 0.95

  if (!is.null(pie_values) && length(pie_values) > 0) {
    props <- pie_values / sum(pie_values)
    n <- length(props)

    if (is.null(pie_colors)) {
      if (!is.null(pie_default_color) && n == 1) {
        pie_colors <- pie_default_color
      } else {
        pie_colors <- grDevices::rainbow(n, s = 0.7, v = 0.9)
      }
    }
    pie_colors <- recycle_to_length(pie_colors, n)

    start_ang <- pi / 2
    for (i in seq_len(n)) {
      if (props[i] <= 0) next

      end_ang <- start_ang - 2 * pi * props[i]
      n_pts <- max(3, ceiling(50 * props[i]))
      angles <- seq(start_ang, end_ang, length.out = n_pts)

      xs <- c(x, x + pie_r * cos(angles), x)
      ys <- c(y, y + pie_r * sin(angles), y)

      graphics::polygon(x = xs, y = ys, col = pie_colors[i], border = NA)
      start_ang <- end_ang
    }

    # Draw pie slice dividers (if more than one slice)
    if (n > 1) {
      start_ang <- pi / 2
      for (i in seq_len(n)) {
        if (props[i] <= 0) next
        end_ang <- start_ang - 2 * pi * props[i]
        graphics::lines(
          x = c(x, x + pie_r * cos(start_ang)),
          y = c(y, y + pie_r * sin(start_ang)),
          col = border.col,
          lwd = pie_slice_border
        )
        start_ang <- end_ang
      }
    }
  } else {
    # Fill inner with white
    angles <- seq(0, 2 * pi, length.out = n_points)
    graphics::polygon(
      x = x + pie_r * cos(angles),
      y = y + pie_r * sin(angles),
      col = "white",
      border = NA
    )
  }

  # Draw borders
  angles <- seq(0, 2 * pi, length.out = n_points)
  graphics::lines(x = x + outer_r * cos(angles), y = y + outer_r * sin(angles),
                  col = border.col, lwd = ring_border_width)
  graphics::lines(x = x + inner_r * cos(angles), y = y + inner_r * sin(angles),
                  col = border.col, lwd = ring_border_width)
}

#' Draw Double Donut with Inner Pie
#'
#' Renders a node with two concentric donut rings and an optional inner pie chart.
#' From outside to inside: outer donut ring, inner donut ring, center pie.
#'
#' @param x,y Node center coordinates.
#' @param size Outer radius.
#' @param donut_values Values for outer donut ring (vector for segments, or single 0-1 for progress).
#' @param donut_colors Colors for outer donut segments.
#' @param donut2_values Values for inner donut ring (vector for segments, or single 0-1 for progress).
#' @param donut2_colors Colors for inner donut segments.
#' @param pie_values Numeric vector for center pie segments.
#' @param pie_colors Vector of colors for pie segments.
#' @param pie_default_color Default color for pie when pie_colors is NULL.
#' @param outer_inner_ratio Where outer donut ends (inner radius as ratio of outer radius). Default 0.7.
#' @param inner_inner_ratio Where inner donut ends (inner radius as ratio of outer radius). Default 0.4.
#' @param bg_color Background color for unfilled portions.
#' @param border.col Border color.
#' @param border.width Border line width.
#' @param pie_border.width Border width for pie slice dividers.
#' @param donut_border.width Border width for donut rings.
#' @keywords internal
draw_double_donut_pie_node_base <- function(x, y, size,
                                            donut_values = NULL, donut_colors = NULL,
                                            donut2_values = NULL, donut2_colors = NULL,
                                            pie_values = NULL, pie_colors = NULL,
                                            pie_default_color = NULL,
                                            outer_inner_ratio = 0.7,
                                            inner_inner_ratio = 0.4,
                                            bg_color = "gray90",
                                            border.col = "black", border.width = 1,
                                            pie_border.width = NULL,
                                            donut_border.width = NULL) {
  # Use separate border widths if provided
  ring_border_width <- if (!is.null(donut_border.width)) donut_border.width else border.width
  pie_slice_border <- if (!is.null(pie_border.width)) pie_border.width else border.width * 0.5

  # Define radii for the three layers
  outer_r <- size                        # Outermost edge
  mid_r <- size * outer_inner_ratio      # Between outer and inner donut
  inner_r <- size * inner_inner_ratio    # Inner edge of inner donut / outer edge of pie
  n_points <- 100

  # Helper to draw ring segment
  draw_ring_segment <- function(start_ang, end_ang, r_outer, r_inner, col) {
    n_pts <- max(10, ceiling(abs(end_ang - start_ang) / (2 * pi) * n_points))
    angles <- seq(start_ang, end_ang, length.out = n_pts)

    outer_x <- x + r_outer * cos(angles)
    outer_y <- y + r_outer * sin(angles)
    inner_x <- x + r_inner * cos(rev(angles))
    inner_y <- y + r_inner * sin(rev(angles))

    graphics::polygon(
      x = c(outer_x, inner_x),
      y = c(outer_y, inner_y),
      col = col,
      border = NA
    )
  }

  # Helper to draw donut ring (handles both progress and segmented)
  draw_donut_ring <- function(values, colors, default_color, r_outer, r_inner) {
    if (is.null(values)) return()

    if (length(values) == 1) {
      # Progress donut - draw background then filled portion
      draw_ring_segment(0, 2 * pi, r_outer, r_inner, bg_color)
      prop <- max(0, min(1, values))
      if (prop > 0) {
        fill_col <- if (!is.null(colors)) colors[1] else if (!is.null(default_color)) default_color else "#4A90D9"
        start_ang <- pi / 2
        end_ang <- pi / 2 - 2 * pi * prop
        draw_ring_segment(start_ang, end_ang, r_outer, r_inner, fill_col)
      }
    } else {
      # Segmented donut
      props <- values / sum(values)
      n <- length(props)

      if (is.null(colors)) {
        colors <- grDevices::rainbow(n, s = 0.7, v = 0.9)
      }
      colors <- recycle_to_length(colors, n)

      start_ang <- pi / 2
      for (i in seq_len(n)) {
        if (props[i] <= 0) next
        end_ang <- start_ang - 2 * pi * props[i]
        draw_ring_segment(start_ang, end_ang, r_outer, r_inner, colors[i])
        start_ang <- end_ang
      }
    }
  }

  # 1. Draw outer donut ring (if values provided)
  if (!is.null(donut_values)) {
    draw_donut_ring(donut_values, donut_colors, NULL, outer_r, mid_r)
  } else {
    # Fill with background if no outer donut
    draw_ring_segment(0, 2 * pi, outer_r, mid_r, bg_color)
  }

  # 2. Draw inner donut ring (if values provided)
  if (!is.null(donut2_values)) {
    draw_donut_ring(donut2_values, donut2_colors, NULL, mid_r, inner_r)
  } else {
    # Fill with background if no inner donut
    draw_ring_segment(0, 2 * pi, mid_r, inner_r, bg_color)
  }

  # 3. Draw center pie (if values provided)
  pie_r <- inner_r * 0.95
  if (!is.null(pie_values) && length(pie_values) > 0) {
    props <- pie_values / sum(pie_values)
    n <- length(props)

    if (is.null(pie_colors)) {
      if (!is.null(pie_default_color) && n == 1) {
        pie_colors <- pie_default_color
      } else {
        pie_colors <- grDevices::rainbow(n, s = 0.7, v = 0.9)
      }
    }
    pie_colors <- recycle_to_length(pie_colors, n)

    start_ang <- pi / 2
    for (i in seq_len(n)) {
      if (props[i] <= 0) next

      end_ang <- start_ang - 2 * pi * props[i]
      n_pts <- max(3, ceiling(50 * props[i]))
      angles <- seq(start_ang, end_ang, length.out = n_pts)

      xs <- c(x, x + pie_r * cos(angles), x)
      ys <- c(y, y + pie_r * sin(angles), y)

      graphics::polygon(x = xs, y = ys, col = pie_colors[i], border = NA)
      start_ang <- end_ang
    }

    # Draw pie slice dividers (if more than one slice)
    if (n > 1) {
      start_ang <- pi / 2
      for (i in seq_len(n)) {
        if (props[i] <= 0) next
        end_ang <- start_ang - 2 * pi * props[i]
        graphics::lines(
          x = c(x, x + pie_r * cos(start_ang)),
          y = c(y, y + pie_r * sin(start_ang)),
          col = border.col,
          lwd = pie_slice_border
        )
        start_ang <- end_ang
      }
    }
  } else {
    # Fill center with white
    angles <- seq(0, 2 * pi, length.out = n_points)
    graphics::polygon(
      x = x + pie_r * cos(angles),
      y = y + pie_r * sin(angles),
      col = "white",
      border = NA
    )
  }

  # 4. Draw all borders
  angles <- seq(0, 2 * pi, length.out = n_points)
  # Outer border
  graphics::lines(x = x + outer_r * cos(angles), y = y + outer_r * sin(angles),
                  col = border.col, lwd = ring_border_width)
  # Middle border (between outer and inner donut)
  graphics::lines(x = x + mid_r * cos(angles), y = y + mid_r * sin(angles),
                  col = border.col, lwd = ring_border_width)
  # Inner border (between inner donut and pie)
  graphics::lines(x = x + inner_r * cos(angles), y = y + inner_r * sin(angles),
                  col = border.col, lwd = ring_border_width)
}

#' Draw Node Label
#'
#' Renders a text label at or near a node.
#'
#' @param x,y Label position coordinates.
#' @param label Text to display.
#' @param cex Character expansion factor.
#' @param col Text color.
#' @param font Font face (1=plain, 2=bold, 3=italic, 4=bold italic).
#' @param pos Position relative to point (NULL=centered, 1=below, 2=left, 3=above, 4=right).
#' @param offset Offset distance when using pos.
#' @keywords internal
draw_node_label_base <- function(x, y, label, cex = 1, col = "black",
                                 font = 1, pos = NULL, offset = 0.5) {
  if (is.null(label) || is.na(label) || label == "") {
    return(invisible())
  }

  graphics::text(
    x = x, y = y,
    labels = label,
    cex = cex,
    col = col,
    font = font,
    pos = pos,
    offset = offset
  )
}

#' Render All Nodes
#'
#' Renders all nodes in the network.
#'
#' @param layout Matrix with x, y columns.
#' @param vsize Vector of node sizes.
#' @param vsize2 Vector of secondary sizes (for ellipse).
#' @param shape Vector of shape names.
#' @param color Vector of fill colors.
#' @param border.color Vector of border colors.
#' @param border.width Vector of border widths.
#' @param pie List of pie value vectors (one per node) or NULL.
#' @param pieColor List of pie color vectors or NULL.
#' @param donut List of donut values or NULL.
#' @param donutColor List of donut color vectors or NULL.
#' @param labels Vector of labels or NULL.
#' @param label.cex Vector of label sizes.
#' @param label.color Vector of label colors.
#' @keywords internal
render_nodes_base <- function(layout, vsize, vsize2 = NULL, shape = "circle",
                              color = "#4A90D9", border.color = "#2C5AA0",
                              border.width = 1, pie = NULL, pieColor = NULL,
                              donut = NULL, donutColor = NULL,
                              labels = NULL, label.cex = 1, label.color = "black") {
  n <- nrow(layout)
  if (n == 0) return(invisible())

  # Vectorize all parameters
  vsize <- recycle_to_length(vsize, n)
  vsize2 <- if (!is.null(vsize2)) recycle_to_length(vsize2, n) else vsize
  shape <- recycle_to_length(shape, n)
  color <- recycle_to_length(color, n)
  border.color <- recycle_to_length(border.color, n)
  border.width <- recycle_to_length(border.width, n)
  label.cex <- recycle_to_length(label.cex, n)
  label.color <- recycle_to_length(label.color, n)

  # Render order: largest to smallest
  order_idx <- get_node_order(vsize)

  for (i in order_idx) {
    x <- layout[i, 1]
    y <- layout[i, 2]

    # Check for pie/donut
    has_pie <- !is.null(pie) && length(pie) >= i && !is.null(pie[[i]]) && length(pie[[i]]) > 0
    has_donut <- !is.null(donut) && length(donut) >= i && !is.null(donut[[i]])

    if (has_donut && has_pie) {
      # Donut with inner pie
      donut_val <- if (length(donut[[i]]) == 1) donut[[i]] else 1
      donut_col <- if (!is.null(donutColor) && length(donutColor) >= i) donutColor[[i]][1] else color[i]
      pie_vals <- pie[[i]]
      pie_cols <- if (!is.null(pieColor) && length(pieColor) >= i) pieColor[[i]] else NULL

      draw_donut_pie_node_base(
        x, y, vsize[i],
        donut_value = donut_val,
        donut_color = donut_col,
        pie_values = pie_vals,
        pie_colors = pie_cols,
        border.col = border.color[i],
        border.width = border.width[i]
      )

    } else if (has_donut) {
      # Donut only
      donut_vals <- donut[[i]]
      donut_cols <- if (!is.null(donutColor) && length(donutColor) >= i) donutColor[[i]] else color[i]

      draw_donut_node_base(
        x, y, vsize[i],
        values = donut_vals,
        colors = donut_cols,
        border.col = border.color[i],
        border.width = border.width[i]
      )

    } else if (has_pie) {
      # Pie only
      pie_vals <- pie[[i]]
      pie_cols <- if (!is.null(pieColor) && length(pieColor) >= i) pieColor[[i]] else NULL

      draw_pie_node_base(
        x, y, vsize[i],
        values = pie_vals,
        colors = pie_cols,
        border.col = border.color[i],
        border.width = border.width[i]
      )

    } else {
      # Standard node
      draw_node_base(
        x, y, vsize[i], vsize2[i],
        shape = shape[i],
        col = color[i],
        border.col = border.color[i],
        border.width = border.width[i]
      )
    }
  }

  # Render labels (all at once, on top of nodes)
  if (!is.null(labels)) {
    for (i in seq_len(n)) {
      if (!is.null(labels[i]) && !is.na(labels[i]) && labels[i] != "") {
        draw_node_label_base(
          layout[i, 1], layout[i, 2],
          label = labels[i],
          cex = label.cex[i],
          col = label.color[i]
        )
      }
    }
  }
}
