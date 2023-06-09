import sys
import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class runAnalysis:

    def __init__(self):
        self.data_folder = "sample2/test/" # set a standard directory

    @staticmethod
    def sc_settings():
        sc.logging.print_versions()
        sc.set_figure_params(facecolor="white", figsize=(8, 8))
        sc.settings.verbosity = 3

    def read_data(self, data_folder: str):
        """
            Takes data_folder and reads in Visium data using scanpy library
        """
        mydata = sc.read_visium(self.data_folder)
        mydata.var_names_make_unique()
        mydata.var["mt"] = mydata.var_names.str.startswith("MT-")
        print(sc.pp.calculate_qc_metrics(mydata, qc_vars=["mt"], inplace=True))
        print(mydata)
        return mydata
    
    def plot_counts(self, mydata):
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        sns.histplot(mydata.obs["total_counts"], kde=False, ax=axs[0])
        sns.histplot(mydata.obs["total_counts"][mydata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
        sns.histplot(mydata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
        sns.histplot(mydata.obs["n_genes_by_counts"][mydata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])

    def filter(self, mydata):
        sc.pp.filter_cells(mydata, min_counts=5000)
        sc.pp.filter_cells(mydata, max_counts=35000)
        mydata = mydata[mydata.obs["pct_counts_mt"] < 20]
        print(f"#cells after MT filter: {mydata.n_obs}")
        sc.pp.filter_genes(mydata, min_cells=10)

    def normalize(self, mydata):
        sc.pp.normalize_total(mydata, inplace=True)
        sc.pp.log1p(mydata)
        sc.pp.highly_variable_genes(mydata, flavor="seurat", n_top_genes=2000)

    def reduce(self, mydata):
        sc.pp.pca(mydata)
        sc.pp.neighbors(mydata)
        sc.tl.umap(mydata)
        sc.tl.leiden(mydata, key_added="clusters")

    def showUMAP(self, mydata):
        plt.rcParams["figure.figsize"] = (4, 4)
        sc.pl.umap(mydata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)

    def show_spatial(self, mydata):
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(mydata, img_key="hires", color=["total_counts", "n_genes_by_counts"])
        sc.pl.spatial(mydata, img_key="hires", color="clusters", size=1.5)

    def show_rankings(self, mydata):
        sc.tl.rank_genes_groups(mydata, "clusters", method="t-test")
        sc.pl.rank_genes_groups_heatmap(mydata, groups="0", n_genes=10, groupby="clusters")
    
    def compute_moran(self, mydata):
        genes = mydata[:, mydata.var.highly_variable].var_names.values[:100]
        sq.gr.spatial_neighbors(mydata)
        sq.gr.spatial_autocorr(
            mydata,
            mode="moran",
            genes=genes,
            n_perms=100,
            n_jobs=1,
        )
        mydata.uns["moranI"].head(10)

    def compute_geary(self, mydata):
        genes = mydata[:, mydata.var.highly_variable].var_names.values[:100]
        sq.gr.spatial_neighbors(mydata)
        sq.gr.spatial_autocorr(
            mydata,
            mode="geary",
            genes=genes,
            n_perms=100,
            n_jobs=1,
        )
        mydata.uns["gearyC"].head(10)


    def run(self) -> None:
        print("Start running")
        self.sc_settings()
        mydata = self.read_data(self.data_folder)
        self.plot_counts(mydata)
        self.filter(mydata)
        print("done filtering!")
        self.normalize(mydata)
        print("done normalizing!")
        self.reduce(mydata)
        print("done dimensionally reducing!")
        self.showUMAP(mydata)
        self.show_spatial(mydata)
        self.show_rankings(mydata)
        self.compute_moran(self, mydata)
        self.compute_geary(self, mydata)
        #finished running processing
        
        return mydata
        print("End running")

    if __name__ ==  '__main__':
        sys.exit(run())