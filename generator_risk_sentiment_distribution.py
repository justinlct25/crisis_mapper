import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def categorize_sentiment(score):
    if score < -0.3:
        return "Negative"
    elif score > 0.3:
        return "Positive"
    else:
        return "Neutral"

def generate_distribution_report(input_csv_file, output_dir="outputs/distribution", table_filename=None, plot_filename=None):
    df = pd.read_csv(input_csv_file, comment='#')

    steps = [
        "Creating output directory",
        "Categorizing sentiment",
        "Grouping data",
        "Saving distribution table",
        "Creating pivot table",
        "Generating heatmap plot",
        "Saving heatmap plot",
        "Generating and saving box plot"
    ]

    # Create a specific subfolder for this run
    input_base = os.path.splitext(os.path.basename(input_csv_file))[0]
    output_dir = os.path.join(output_dir, f"distribution_{input_base}")

    with tqdm(total=len(steps), desc="Generating Distribution Report", unit="step") as pbar:
        os.makedirs(output_dir, exist_ok=True)
        pbar.update(1)

        df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
        pbar.update(1)

        distribution = df.groupby(['risk_level_semantic', 'sentiment_category']).size().reset_index(name='count')
        pbar.update(1)

        table_filename = table_filename if table_filename else f"distribution_table_{input_base}.csv"
        distribution_table_file = os.path.join(output_dir, table_filename)
        distribution.to_csv(distribution_table_file, index=False)
        print(f"✅ Distribution table saved to: {distribution_table_file}")
        pbar.update(1)

        pivot_table = distribution.pivot(index='risk_level_semantic', columns='sentiment_category', values='count').fillna(0)
        print("✅ Pivot table created:")
        print(pivot_table)
        pbar.update(1)

        risk_order = []
        for group in ['High', 'Moderate', 'Low']:
            matching = sorted([r for r in pivot_table.index if r.startswith(group)])
            risk_order.extend(matching)
        pivot_table = pivot_table.loc[risk_order]

        plt.figure(figsize=(10, 12))
        ax = sns.heatmap(
            pivot_table,
            annot=False,
            fmt=".0f",
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"size": 9}
        )
        plt.title("Distribution of Posts by Sentiment Category and Risk Level", fontsize=14, pad=12)
        plt.xlabel("Sentiment Category", fontsize=12)
        plt.ylabel("Risk Level", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=9, rotation=0)
        plt.tight_layout()
        plot_filename = plot_filename if plot_filename else f"heatmap_distribution_plot_{input_base}.png"
        distribution_plot_file = os.path.join(output_dir, plot_filename)
        plt.savefig(distribution_plot_file)
        plt.close()
        print(f"✅ Distribution plot saved to: {distribution_plot_file}")
        pbar.update(1)

        # Generate and save box plot
        df['risk_category'] = df['risk_level_semantic'].str.extract(r"^(Low|Moderate|High)")

        # Calculate total counts for each risk category
        risk_counts = df['risk_category'].value_counts()
        y_labels = [f"{category}\n(n={risk_counts.get(category, 0)})" for category in ["Low", "Moderate", "High"]]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="sentiment", y="risk_category", order=["Low", "Moderate", "High"], palette="Set2")
        plt.title("Distribution of Sentiment Scores by Semantic Risk Category")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Risk Category")
        plt.grid(True, axis='x')
        plt.yticks(ticks=range(len(y_labels)), labels=y_labels)  
        plt.tight_layout()

        box_plot_filename = f"boxplot_sentiment_vs_risk_{input_base}.png"
        box_plot_file = os.path.join(output_dir, box_plot_filename)
        plt.savefig(box_plot_file)
        plt.close()
        print(f"✅ Box plot saved to: {box_plot_file}")
        pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment-risk distribution report from CSV.")
    parser.add_argument("input_csv_file", help="Path to the CSV file containing 'risk_level_semantic' and 'sentiment' columns")
    parser.add_argument("--output_dir", default="outputs/distribution", help="Directory to save the report outputs")
    args = parser.parse_args()
    generate_distribution_report(args.input_csv_file, output_dir=args.output_dir)
