import pandas as pd
import matplotlib.pyplot as plt


def visualize_histograms(df: pd.DataFrame) -> None:
    df.hist(bins=50, figsize=(12, 9))
    plt.show()
