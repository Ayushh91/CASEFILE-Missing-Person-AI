# Data source documentation

Primary source: Microsoft GeoLife GPS Trajectory Dataset. Source: <https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/>.

The program writes record counts only after inspecting local `.plt` files. It uses latitude, longitude, altitude where present, timestamps, user folder ID, and trajectory filename. The public dataset is used for movement-pattern research, not identifying people. Consult the official source for licence/usage terms. Limitations include missing modalities, sampling variation, geographic coverage, and lack of missing-person ground truth. When it is absent, CASEFILE produces a clearly labelled synthetic demo only.
