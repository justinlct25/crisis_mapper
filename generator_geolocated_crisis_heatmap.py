import folium
from folium.plugins import MarkerCluster
import pandas as pd

def generate_heatmap(input_data, output_file=None):
    """
    Generate a heatmap from geolocated posts.
    
    Args:
        input_data (str or pd.DataFrame): Path to the CSV file containing geolocated posts or a DataFrame.
        output_file (str): Path to save the generated heatmap HTML file. Defaults to "crisis_heatmap_<input_file_name>.html".
    """
    # Load data based on the type of input
    if isinstance(input_data, str):  # If input is a CSV file path
        print("Loading data from CSV file...")
        df = pd.read_csv(input_data, comment='#')
        input_file_name = input_data.split("/")[-1].replace(".csv", "")
    elif isinstance(input_data, pd.DataFrame):  # If input is a DataFrame
        print("Loading data from DataFrame...")
        df = input_data
        input_file_name = "dataframe_input"
    else:
        raise ValueError("Input must be a file path (str) or a pandas DataFrame.")

    # Set default output file name if not provided
    if output_file is None:
        output_file = f"outputs/heatmap/crisis_heatmap_{input_file_name}.html"

    print("Generating heatmap...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)
    cluster = MarkerCluster(
        spiderfy_on_max_zoom=True,
        show_coverage_on_hover=True,
        options={"spiderfyDistanceMultiplier": 2}  
)

    for _, row in df.iterrows():
        color = 'yellow'
        if str(row.get('risk_level_semantic', '')).startswith('High'):
            color = 'red'
        elif str(row.get('risk_level_semantic', '')).startswith('Moderate'):
            color = 'orange'

        popup = folium.Popup(
            html=f"""
            <div style="width: 300px; overflow-x: auto; white-space: nowrap;">
                <b>Location:</b> {row['validated_location']}<br>
                <b>Sentiment:</b> {row.get('sentiment', 'N/A')}<br>
                <b>Risk Level:</b> {row.get('risk_level_semantic', 'N/A')}<br>
                <b>Date:</b> {row.get('timestamp', 'N/A')}<br>
                <b>Post ID:</b> {row['id']}<br>
                <b>URL:</b> <a href="{row.get('url', '#')}" target="_blank">{row.get('url', '#')}</a>
            </div>
            """,
            max_width=300
        )
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(cluster)

    cluster.add_to(map_)
    map_.save(output_file)
    print(f"Heatmap saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a heatmap from geolocated posts.")
    parser.add_argument("input_csv_file", help="Path to the CSV file containing geolocated posts.")
    parser.add_argument("output_file", nargs="?", default=None, help="Path to save the generated heatmap HTML file.")
    args = parser.parse_args()

    generate_heatmap(args.input_csv_file, args.output_file)