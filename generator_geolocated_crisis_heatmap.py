import folium
from folium.plugins import MarkerCluster
import pandas as pd

def generate_heatmap(input_csv_file, output_file=None):
    """
    Generate a heatmap from geolocated posts.
    
    Args:
        input_csv_file (str): Path to the CSV file containing geolocated posts.
        output_file (str): Path to save the generated heatmap HTML file. Defaults to "crisis_heatmap_<input_file_name>.html".
    """
    print("Loading data...")
    df = pd.read_csv(input_csv_file, comment='#')

    # Set default output file name if not provided
    if output_file is None:
        input_file_name = input_csv_file.split("/")[-1].replace(".csv", "")
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