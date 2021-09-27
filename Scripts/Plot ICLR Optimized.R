# We first set the working directory and import the necessary libraries.

setwd("..") # change working directory to main repository 
library("ggplot2") # plotting
library("gridExtra") # arrange multiple plots

# We load the topologically optimized point cloud data sets representing the ICLR acronym.

df <- list()
for(file in list.files(file.path("Data", "ICLR Optimized"))){
  full_path <- file.path("Data", "ICLR Optimized", file)
  epoch <- strsplit(file, "\\D+")[[1]][-1]
  df[[epoch]] <- read.table(full_path, sep=",", col.names=c("x", "y"))
}

# We plot the optimized point clouds.

group <- read.table(file.path("Data", "ICLR.csv"))[["group"]]
epochPlots <- list()
for(epoch in as.character(sort(as.numeric(names(df))))){
  this_df <- cbind(df[[epoch]], group)
  epochPlots[[length(epochPlots) + 1]] <- ggplot(this_df, aes(x=x, y=y, fill=group)) +
    geom_point(size=1.25, pch=21) +
    xlim(5, 95) +
    ylim(25, 75) +
    coord_fixed() +
    theme_bw() +
    ggtitle(paste(epoch, " epochs")) +
    theme(plot.title=element_text(hjust=0.5, size=22), text=element_text(size=20), 
          legend.position="none")
}

grid.arrange(grobs=epochPlots, nrow=2, ncol=3)
