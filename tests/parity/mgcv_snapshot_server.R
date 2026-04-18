#!/usr/bin/env Rscript

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript mgcv_snapshot_server.R <mgcv_snapshot_script_path>")
}

snapshot_script <- args[[1]]

write_response <- function(resp) {
  write_lines <- toJSON(resp, auto_unbox = TRUE, digits = 17)
  cat(write_lines, "\n")
  flush.console()
}

serve_loop <- function(snapshot_script_path) {
  while (TRUE) {
    line <- readLines("stdin", n = 1, warn = FALSE)
    if (length(line) == 0L) {
      break
    }
    line <- trimws(line)
    if (nchar(line) == 0L) {
      next
    }

    request <- tryCatch(
      fromJSON(line, simplifyVector = FALSE),
      error = function(e) {
        write_response(list(status = "error", message = paste("invalid request JSON:", conditionMessage(e))))
        NULL
      }
    )
    if (is.null(request)) {
      next
    }

    if (isTRUE(request$action == "shutdown")) {
      break
    }

    if (!identical(request$action, "snapshot")) {
      write_response(list(id = request$id, status = "error", message = "unsupported action"))
      next
    }

    csv_path <- request$csv_path
    formula <- request$formula
    family <- request$family
    method <- request$method
    select <- request$select
    weights_column <- request$weights_column

    output_json <- tempfile(fileext = ".json")
    snapshot_args <- c(
      csv_path,
      output_json,
      formula,
      family,
      method,
      ifelse(isTRUE(select), "true", "false")
    )
    if (!is.null(weights_column) && nzchar(weights_column)) {
      snapshot_args <- c(snapshot_args, weights_column)
    }

    env <- new.env(parent = baseenv())
    env$commandArgs <- function(trailingOnly = TRUE) {
      stopifnot(length(snapshot_args) >= 1L)
      snapshot_args
    }

    run <- tryCatch(
      {
        source(snapshot_script_path, local = env, echo = FALSE, keep.source = FALSE)
        snapshot <- read_json(output_json, simplifyVector = FALSE)
        list(status = "ok", result = snapshot)
      },
      error = function(e) {
        list(status = "error", message = conditionMessage(e))
      }
    )

    if (file.exists(output_json)) {
      file.remove(output_json)
    }

    run$id <- request$id
    write_response(run)
  }
}

serve_loop(snapshot_script)
