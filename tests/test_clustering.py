import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fx_external_pipeline_full.clustering import run_clustering

def test_clustering_outputs():
    panel = pd.DataFrame({
        "company_id":[f"C{i}" for i in range(20)],
        "export_amt":[1000+i*10 for i in range(20)],
        "import_amt":[800+i*8 for i in range(20)],
        "net_exposure":[(1000+i*10)-(800+i*8) for i in range(20)],
        "region_sido_code":[i%3 for i in range(20)],
        "corp_grade_code":[i%2 for i in range(20)],
    })
    df, km, gmm = run_clustering(panel, ["export_amt","import_amt","net_exposure","region_sido_code","corp_grade_code"], k_kmeans=3, k_gmm=3)
    assert {"company_id","cluster_kmeans","cluster_gmm"}.issubset(set(df.columns))
    assert "cluster_kmeans" in km.columns
    assert "cluster_gmm" in gmm.columns
