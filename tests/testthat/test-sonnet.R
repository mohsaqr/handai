test_that("sonnet() creates network from matrix", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj)

  expect_s3_class(net, "sonnet_network")
  expect_equal(net$network$n_nodes, 3)
  expect_equal(net$network$n_edges, 3)
})

test_that("sonnet() creates network from edge list", {
  edges <- data.frame(from = c("A", "B"), to = c("B", "C"))
  net <- sonnet(edges)

  expect_s3_class(net, "sonnet_network")
  expect_equal(net$network$n_nodes, 3)
})

test_that("sonnet() applies default layout", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj)

  layout <- net$network$get_layout()
  expect_false(is.null(layout))
  expect_equal(nrow(layout), 3)
})

test_that("sn_layout() changes layout", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net1 <- sonnet(adj, layout = "spring", seed = 42)
  net2 <- net1 |> sn_layout("circle")

  coords1 <- net1$network$get_layout()
  coords2 <- net2$network$get_layout()

  # Layouts should be different
  expect_false(all(coords1$x == coords2$x))
})

test_that("sn_nodes() modifies node aesthetics", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj) |>
    sn_nodes(size = 0.1, fill = "red")

  aes <- net$network$get_node_aes()
  expect_true(all(aes$size == 0.1))
  expect_true(all(aes$fill == "red"))
})

test_that("sn_edges() modifies edge aesthetics", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj) |>
    sn_edges(width = 2, color = "blue")

  aes <- net$network$get_edge_aes()
  expect_true(all(aes$width == 2))
  expect_true(all(aes$color == "blue"))
})

test_that("sn_theme() applies theme", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj) |> sn_theme("dark")

  theme <- net$network$get_theme()
  expect_equal(theme$name, "dark")
  expect_equal(theme$get("background"), "#1a1a2e")
})

test_that("pipe chain works correctly", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- adj |>
    sonnet(layout = "circle") |>
    sn_nodes(size = 0.08, fill = "steelblue") |>
    sn_edges(width = 1.5) |>
    sn_theme("minimal")

  expect_s3_class(net, "sonnet_network")
  expect_true(all(net$network$get_node_aes()$fill == "steelblue"))
  expect_equal(net$network$get_theme()$name, "minimal")
})

test_that("print method works", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj)

  expect_output(print(net), "Sonnet Network")
  expect_output(print(net), "Nodes: 3")
})

test_that("summary method works", {
  adj <- matrix(c(0, 1, 1, 1, 0, 1, 1, 1, 0), nrow = 3)
  net <- sonnet(adj)

  expect_output(summary(net), "Sonnet Network Summary")
  result <- summary(net)
  expect_equal(result$n_nodes, 3)
})
