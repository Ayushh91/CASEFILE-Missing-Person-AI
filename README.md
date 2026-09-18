
# CASEFILE — GeoLife Mobility Analysis Simulation

This project uses the public Microsoft GeoLife GPS Trajectory Dataset when it is installed locally, and otherwise uses clearly labelled synthetic demo data.

## Dataset setup

1. Download GeoLife from the official Microsoft page:
   https://www.microsoft.com/en-us/download/details.aspx?id=52367
2. Extract the archive.
3. Copy the `Data` directory into:
   `data/raw/GeoLife/Data/`
   or place the user folders directly under `data/raw/GeoLife/`.

The loader recursively discovers `.plt` files and parses the standard six-line header plus latitude, longitude, altitude, date and time fields.

## Run

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline
pytest
streamlit run app/app.py
```

## Academic and ethical limits

The project does not contain missing-person records or identify real people. Fictional cases are generated from aggregate mobility patterns. Probabilities, anomalies, routes and rankings are uncertain statistical outputs; they must not be used as search instructions, surveillance, or evidence of wrongdoing.
