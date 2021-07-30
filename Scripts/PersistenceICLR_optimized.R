# We first set the working directory and import the necessary libraries.

setwd("..") # change working directory to main repository 
library("TDA") # persistent homology in R
library("ggplot2") # plotting

# We load and plot a topologically optimized point cloud data set representing the ICLR acronym.

df <- read.table(file.path("Data", "ICLR_optimized.csv"), sep=",", col.names=c("x", "y"))
ggplot(df, aes(x=x, y=y)) +
  geom_point(size=1.5) +
  coord_fixed() +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=22), text=element_text(size=20))

# The alpha filtration is now obtained follows.

filtration <- alphaComplexFiltration(df)

# We can compute the persistence for this filtration as follows.

diag <- filtrationDiag(filtration, maxdimension=1)

# We can visualize the results of persistent homology through a persistence diagram.
# In the diagram, a point (b, d) represent a topological hole/feature that persists from time b to d.

op <- par(mar = c(3.25, 3.25, 1, 1))
plot.diagram(diag[["diagram"]], diagLim=c(0, 300))
abline(h=300, lty=2) # line marking infinite death time
legend(x=250, y=100, legend=c("H0", "H1"), col=c("black", "red"), pch=c(19, 2), pt.lwd=2, box.lty=0); par(op)
