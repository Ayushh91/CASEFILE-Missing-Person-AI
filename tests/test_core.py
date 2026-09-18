import pandas as pd
from src.data_loader import generate_demo_trajectories
from src.preprocessing import clean_trajectories
from src.feature_engineering import haversine_km,add_movement_features
from src.clustering import cluster_locations
from src.anomaly_detection import detect_anomalies
from src.case_generator import generate_cases
from src.route_prediction import build_transition_matrix,predict_route
from src.priority_scoring import rank_areas
def prepared():
    clean,_=clean_trajectories(generate_demo_trajectories(3,30)); feat=add_movement_features(clean); return cluster_locations(feat)
def test_coordinate_cleaning():
    df=generate_demo_trajectories(1,3); df.loc[0,"latitude"]=200; clean,report=clean_trajectories(df); assert report["invalid_coordinates_or_time"]==1 and len(clean)==2
def test_haversine(): assert 110 < haversine_km(0,0,1,0) < 112
def test_features_clusters_anomalies_cases_routes_priority():
    data,_,centers,_=prepared(); assert "area_id" in data
    anomalies,model,features=detect_anomalies(data); assert set(anomalies.anomaly_label).issubset({-1,1}) and features
    cases=generate_cases(anomalies,8); assert "Target_Area" in cases and cases.Person_ID.str.startswith("FICTIONAL_").all()
    matrix=build_transition_matrix(anomalies)
    for destinations in matrix.values(): assert abs(sum(destinations.values())-1)<1e-9
    route=predict_route(matrix,cases.iloc[0].Previous_Area); probs=pd.DataFrame({"area_id":centers.area_id.astype(str),"probability":[100/len(centers)]*len(centers)})
    ranks=rank_areas(probs,centers,cases.iloc[0].Last_Latitude,cases.iloc[0].Last_Longitude,route); assert ranks.priority_score.between(0,100).all()
