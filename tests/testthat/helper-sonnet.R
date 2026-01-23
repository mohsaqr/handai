# Test helper - load the package
library(Sonnet)

# Make internal functions available for testing
parse_matrix <- Sonnet:::parse_matrix
parse_edgelist <- Sonnet:::parse_edgelist
draw_circle <- Sonnet:::draw_circle
draw_square <- Sonnet:::draw_square
draw_triangle <- Sonnet:::draw_triangle
draw_diamond <- Sonnet:::draw_diamond
draw_ellipse <- Sonnet:::draw_ellipse
draw_heart <- Sonnet:::draw_heart
draw_star <- Sonnet:::draw_star
layout_circle <- Sonnet:::layout_circle
layout_spring <- Sonnet:::layout_spring
layout_groups <- Sonnet:::layout_groups
recycle_to_length <- Sonnet:::recycle_to_length
