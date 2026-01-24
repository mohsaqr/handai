library(Sonnet)

# Test reciprocal edge separation
# This asymmetric matrix has edges going both directions between nodes
mat_directed <- matrix(c(
  0.37, 0.08, 0.14, 0.06, 0.33, 0.10, 0.24, 0.22, 0.08,
  0.40, 0.00, 0.23, 0.14, 0.47, 0.12, 0.16, 0.09, 0.07,
  0.14, 0.00, 0.00, 0.27, 0.07, 0.06, 0.29, 0.06, 0.12,
  0.00, 0.00, 0.00, 0.00, 0.48, 0.32, 0.06, 0.00, 0.00,
  0.00, 0.00, 0.50, 0.00, 0.08, 0.13, 0.12, 0.00, 0.00,
  0.00, 0.00, 0.00, 0.19, 0.32, 0.00, 0.19, 0.27, 0.06,
  0.00, 0.00, 0.00, 0.17, 0.00, 0.08, 0.19, 0.38, 0.06,
  0.15, 0.00, 0.00, 0.00, 0.00, 0.07, 0.11, 0.08, 0.09,
  0.00, 0.00, 0.00, 0.00, 0.00, 0.07, 0.12, 0.00, 0.00
), nrow = 9, byrow = TRUE)

labels <- c("plan", "synthesis", "adapt", "cohesion", "consensus",
            "coregulate", "discuss", "emotion", "monitor")

# Test: Reciprocal edges should automatically be curved apart
soplot(mat_directed,
  title = "Reciprocal Edges Auto-Separated",
  layout = "circle",
  labels = labels,
  node_size = 0.06,
  node_fill = "#87CEEB",
  label_position = "center",
  edge_color = "#191970",
  edge_width = 2,
  threshold = 0.05,
  arrow_size = 0.012
)

# Original test below

mat <- matrix(c(
  0.37, 0.08, 0.14, 0.06, 0.33, 0.10, 0.24, 0.22, 0.08,
  0.40, 0.00, 0.23, 0.14, 0.47, 0.12, 0.16, 0.09, 0.07,
  0.14, 0.00, 0.00, 0.27, 0.07, 0.06, 0.29, 0.06, 0.12,
  0.00, 0.00, 0.00, 0.00, 0.48, 0.32, 0.06, 0.00, 0.00,
  0.00, 0.00, 0.50, 0.00, 0.08, 0.13, 0.12, 0.00, 0.00,
  0.00, 0.00, 0.00, 0.19, 0.32, 0.00, 0.19, 0.27, 0.06,
  0.00, 0.00, 0.00, 0.17, 0.00, 0.08, 0.19, 0.38, 0.06,
  0.15, 0.00, 0.00, 0.00, 0.00, 0.07, 0.11, 0.08, 0.09,
  0.00, 0.00, 0.00, 0.00, 0.00, 0.07, 0.12, 0.00, 0.00
), nrow = 9, byrow = TRUE)

# Node labels
labels <- c("plan", "synthesis", "adapt", "cohesion", "consensus", 
            "coregulate", "discuss", "emotion", "monitor")

# Pastel colors for nodes (outer donut ring)
node_colors <- c(
  "#FFB6C1",  # plan - pink
  "#D3D3D3",  # synthesis - gray
  "#7FDBDB",  # adapt - teal
  "#FFFACD",  # cohesion - light yellow
  "#D8BFD8",  # consensus - lavender
  "#F08080",  # coregulate - coral
  "#87CEEB",  # discuss - sky blue
  "#FFB347",  # emotion - orange
  "#90EE90"   # monitor - light green
)

# Random values for donut ring (outer proportion)
set.seed(123)
donut_vals <- runif(9, 0.4, 0.85)

# Random values for inner pie (3 segments per node)
pie_vals <- lapply(1:9, function(i) runif(3, 1, 5))

# Colors for pie segments
pie_cols <- c("#E41A1C", "#377EB8", "#4DAF4A")



soplot(mat,
  title = "Donut + Pie Nodes (Thin Ring)",
  layout = "circle",
  labels = labels,
  node_shape = "donut_pie",
  node_size = 0.08,
  node_fill = node_colors,
  donut_values = donut_vals,
  pie_values = pie_vals,
  pie_colors = pie_cols,
  donut_inner_ratio = 0.7,        # Thin ring (higher = thinner)
  node_border_width = 1.5,
  node_border_color = "#333366",
  label_position = "center",
  label_size = 9,
  label_color = "black",
  edge_color = "#191970",
  edge_width = "weight",
  maximum = 0.5,
  threshold = 0.05,
  edge_labels = TRUE,
  edge_label_size = 6,
  edge_label_color = "#333333",
  curvature = 0.2,
  arrow_size = 0.012
)


# Test 2: Simple donut nodes
cat("Test 2: donut shape\n")
png("test_donut.png", width = 1000, height = 1000, res = 100)
soplot(mat,
  title = "Donut Nodes",
  layout = "circle",
  labels = labels,
  node_shape = "donut",
  node_size = 0.07,
  node_fill = node_colors,
  pie_values = donut_vals,
  donut_inner_ratio = 0.6,
  donut_show_value = FALSE,
  node_border_width = 1.5,
  node_border_color = "#333366",
  label_position = "center",
  label_color = "black",
  edge_color = "#191970",
  edge_width = "weight",
  threshold = 0.05,
  curvature = 0.15,
  arrow_size = 0.012
)

# Test 3: Pie nodes
cat("Test 3: pie shape\n")
png("test_pie.png", width = 1000, height = 1000, res = 100)
soplot(mat,
  title = "Pie Chart Nodes",
  layout = "circle",
  labels = labels,
  node_shape = "pie",
  node_size = 0.07,
  pie_values = pie_vals,
  pie_colors = pie_cols,
  node_border_width = 1.5,
  node_border_color = "#333366",
  label_position = "below",
  label_color = "black",
  edge_color = "#191970",
  edge_width = "weight",
  threshold = 0.05,
  arrow_size = 0.012
)


# Test 4: Simple circles with transparency
cat("Test 4: circles with transparency\n")
png("test_circles.png", width = 1000, height = 1000, res = 100)
soplot(mat,
  title = "Circle Nodes with Transparency",
  layout = "spring",
  labels = labels,
  node_shape = "circle",
  node_size = 0.06,
  node_fill = node_colors,
  node_alpha = 0.8,
  node_border_width = 2,
  node_border_color = "#333333",
  label_position = "center",
  edge_color = "#191970",
  edge_width = "weight",
  edge_alpha = 0.7,
  threshold = 0.1,
  edge_labels = TRUE,
  edge_label_size = 7,
  arrow_size = 0.01
)

# Test 5: PDF output
cat("Test 5: PDF output\n")
pdf("test_network.pdf", width = 10, height = 10)
soplot(mat,
  title = "Network Visualization Test",
  layout = "circle",
  labels = labels,
  node_shape = "donut_pie",
  node_size = 0.08,
  node_fill = node_colors,
  donut_values = donut_vals,
  pie_values = pie_vals,
  pie_colors = pie_cols,
  donut_inner_ratio = 0.7,
  node_border_width = 1.5,
  node_border_color = "#333366",
  label_position = "center",
  label_size = 9,
  edge_color = "#191970",
  edge_width = "weight",
  maximum = 0.5,
  threshold = 0.05,
  edge_labels = TRUE,
  edge_label_size = 6,
  curvature = 0.2,
  arrow_size = 0.012
)
