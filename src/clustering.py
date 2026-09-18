import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .config import SEED

def cluster_locations(data: pd.DataFrame, n_clusters: int=4, method: str="kmeans") -> tuple[pd.DataFrame, object, pd.DataFrame, dict]:
    df=data.copy(); coords=df[["latitude","longitude"]].to_numpy(); scaled=StandardScaler().fit_transform(coords)
    model=DBSCAN(eps=.35,min_samples=5).fit(scaled) if method=="dbscan" else KMeans(n_clusters=min(n_clusters,len(df)),random_state=SEED,n_init="auto").fit(scaled)
    df["area_id"]=model.labels_; usable=df.area_id>=0
    centers=df[usable].groupby("area_id")[["latitude","longitude"]].mean().reset_index(); centers["visit_frequency"]=df[usable].groupby("area_id").size().values
    labels=df.loc[usable,"area_id"]; metrics={"cluster_count":int(labels.nunique()),"noise_observations":int((~usable).sum()),"silhouette":float(silhouette_score(scaled[usable],labels)) if labels.nunique()>1 and usable.sum()>labels.nunique() else None}
    return df,model,centers,metrics
