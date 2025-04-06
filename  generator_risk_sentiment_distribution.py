import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_distribution_report(df, output_dir="outputs", table_filename="distribution_table.csv", plot_filename="distribution_plot.png"):
    steps = [
        "Creating output directory",
        "Grouping data",
        "Saving distribution table",
        "Creating pivot table",
        "Generating heatmap plot",
        "Saving heatmap plot"
    ]

    with tqdm(total=len(steps), desc="Generating Distribution Report", unit="step") as pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            pbar.update(1)

            distribution = df.groupby(['risk_level_semantic', 'sentiment']).size().reset_index(name='count')
            pbar.update(1)

            distribution_table_file = f"{output_dir}/{table_filename}"
            distribution.to_csv(distribution_table_file, index=False)
            print(f"✅ Distribution table saved to: {distribution_table_file}")
            pbar.update(1)

            pivot_table = distribution.pivot(index='risk_level_semantic', columns='sentiment', values='count').fillna(0)
            print("✅ Pivot table created:")
            print(pivot_table)
            pbar.update(1)

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Blues", cbar=True)
            plt.title("Distribution of Posts by Sentiment and Risk Level")
            plt.xlabel("Sentiment")
            plt.ylabel("Risk Level")
            plt.tight_layout()
            pbar.update(1)

            distribution_plot_file = f"{output_dir}/{plot_filename}"
            plt.savefig(distribution_plot_file)
            plt.close()
            print(f"✅ Distribution plot saved to: {distribution_plot_file}")
            pbar.update(1)

        except Exception as e:
            print("❌ An error occurred during report generation:", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment-risk distribution report from CSV.")
    parser.add_argument("input_csv", help="Path to the CSV file containing 'risk_level_semantic' and 'sentiment' columns")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save the report outputs")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
        generate_distribution_report(df, output_dir=args.output_dir)
    except Exception as e:
        print("❌ Failed to read input CSV:", str(e))
