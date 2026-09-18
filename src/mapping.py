def make_map(last_lat, last_lon, centers, priorities=None, route=None):
    """Build a clean CASEFILE map with readable map labels."""

    import folium

    fmap = folium.Map(
        location=[last_lat, last_lon],
        zoom_start=12,
        control_scale=True,
        tiles=None
    )

    # Clean Voyager base map
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="Voyager",
        subdomains="abcd",
        max_zoom=20
    ).add_to(fmap)

    # English-readable geographic labels
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="English Labels",
        subdomains="abcd",
        max_zoom=20,
        overlay=True,
        control=False,
        show=True
    ).add_to(fmap)

    # Last known location
    folium.Marker(
        [last_lat, last_lon],
        popup=folium.Popup(
            "<b>Last Known Location</b><br>"
            "Synthetic case location",
            max_width=300
        ),
        tooltip="Last Known Location",
        icon=folium.Icon(
            color="red",
            icon="info-sign"
        )
    ).add_to(fmap)

    # Priority lookup
    priority_lookup = {}

    if priorities is not None:
        priority_lookup = priorities.set_index(
            "area_id"
        ).to_dict("index")

    # Area markers
    for row in centers.itertuples():

        item = priority_lookup.get(row.area_id, {})

        priority_score = item.get("priority_score", 0)
        priority_label = item.get("priority_label", "N/A")
        probability = item.get("probability", 0)

        popup_html = f"""
        <div style="font-family: Arial; width: 230px;">
            <h4>CASEFILE Area {row.area_id}</h4>

            <b>Visits:</b> {row.visit_frequency}<br>
            <b>Probability:</b> {probability:.2f}%<br>
            <b>Priority Score:</b> {priority_score:.2f}<br>
            <b>Priority:</b> {priority_label}<br>
            <b>Latitude:</b> {row.latitude:.5f}<br>
            <b>Longitude:</b> {row.longitude:.5f}
        </div>
        """

        folium.CircleMarker(
            [row.latitude, row.longitude],
            radius=8,
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=f"CASEFILE Area {row.area_id}",
            color="blue",
            fill=True,
            fill_opacity=0.7
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

    return fmap