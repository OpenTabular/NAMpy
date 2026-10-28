# Run the unmodified k.check body with explicit shared sampling inputs.
# Only sample() is supplied; residuals, nearest neighbors, rescaling and
# permutation statistics remain the upstream implementation.
script <- sub('^--file=', '', grep('^--file=',commandArgs(),value=TRUE)[1])
source(file.path(dirname(script),'mgcv_registered_pipeline.R'))
permutations <- do.call(cbind,lapply(p$permutations,unlist)) + 1L
selected_rows <- unlist(p$selected_rows) + 1L
cursor <- 0L
sample_shared <- function(x,size,...) {
  if (length(x) == nrow(d) && size < nrow(d)) return(selected_rows)
  cursor <<- cursor + 1L
  stopifnot(length(x) == nrow(permutations),size == nrow(permutations))
  permutations[,cursor]
}
checker <- mgcv::k.check
environment(checker) <- list2env(list(sample=sample_shared),parent=environment(checker))
answer <- checker(b,subsample=p$subsample,n.rep=ncol(permutations))
write_json(list(case_name=p$name,engine=p$engine,
                version=as.character(packageVersion('mgcv')),
                source_warnings=source_warnings,
                upstream_notes='Shared rows and permutations replace only sample(); the upstream k.check body and numerical calculations are unchanged.',
                values=unname(answer),labels=rownames(answer),
                permutation_calls=cursor),a[2],digits=17,auto_unbox=TRUE,na='null')
