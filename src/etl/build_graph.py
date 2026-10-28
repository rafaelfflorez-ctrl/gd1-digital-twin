"""
Construcción del grafo hidráulico para GNN (Gemelo Digital 1).
Nodos: Estaciones IDEAM + Puntos proceso ACUACAR + PTAP El Bosque
Edges: Conexiones hidrológicas (cuenca) + hidráulicas (planta) + temporales
"""
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path
import json

CURATED_DIR = Path(__file__).parent.parent.parent / "data" / "curated"
GRAPH_DIR = Path(__file__).parent.parent.parent / "data" / "graphs"

# Definición de nodos del sistema
NODOS = {
    # Estaciones IDEAM (cuenca)
    "IDEAM_100101": {"tipo": "ideam", "nombre": "Cartagena Rafael Núñez", "lat": 10.44, "lon": -75.51, "cuenca": "Bahía Cartagena"},
    "IDEAM_100102": {"tipo": "ideam", "nombre": "Cartagena Zona Franca", "lat": 10.38, "lon": -75.43, "cuenca": "Canal del Dique"},
    "IDEAM_100103": {"tipo": "ideam", "nombre": "Turbaco", "lat": 10.33, "lon": -75.38, "cuenca": "Canal del Dique"},
    "IDEAM_100104": {"tipo": "ideam", "nombre": "Arjona", "lat": 10.28, "lon": -75.27, "cuenca": "Canal del Dique"},
    "IDEAM_100105": {"tipo": "ideam", "nombre": "María La Baja", "lat": 10.15, "lon": -75.15, "cuenca": "Canal del Dique"},
    "IDEAM_100106": {"tipo": "ideam", "nombre": "San Juan Nepomuceno", "lat": 9.97, "lon": -75.08, "cuenca": "Montes de María"},
    "IDEAM_100107": {"tipo": "ideam", "nombre": "El Carmen de Bolívar", "lat": 9.72, "lon": -75.12, "cuenca": "Montes de María"},
    "IDEAM_100108": {"tipo": "ideam", "nombre": "San Onofre", "lat": 9.60, "lon": -75.38, "cuenca": "Montes de María"},
    
    # Puntos captura / entrada planta
    "CAPTACION_BOSQUE": {"tipo": "captacion", "nombre": "Captación El Bosque", "lat": 10.38, "lon": -75.41, "fuente": "Canal del Dique"},
    
    # Proceso PTAP El Bosque (nodos unitarios)
    "COAGULACION": {"tipo": "proceso", "nombre": "Coagulación-Floculación", "unidad": "tanques"},
    "SEDIMENTACION": {"tipo": "proceso", "nombre": "Sedimentación", "unidad": "sedimentadores"},
    "FILTRACION": {"tipo": "proceso", "nombre": "Filtración Rápida", "unidad": "filtros", "n_paralelos": 8},
    "CLORACION": {"tipo": "proceso", "nombre": "Cloración", "unidad": "tanques contacto"},
    
    # Salida
    "SALIDA_PLANTA": {"tipo": "salida", "nombre": "Salida Planta - Red Distribución", "lat": 10.38, "lon": -75.41},
}

# Edges: (origen, destino, tipo, peso_inicial, descripcion)
EDGES = [
    # Hidrológicos: cuenca -> captación
    ("IDEAM_100105", "CAPTACION_BOSQUE", "hidrologico", 0.8, "Canal del Dique - María La Baja"),
    ("IDEAM_100104", "CAPTACION_BOSQUE", "hidrologico", 0.6, "Canal del Dique - Arjona"),
    ("IDEAM_100103", "CAPTACION_BOSQUE", "hidrologico", 0.5, "Canal del Dique - Turbaco"),
    ("IDEAM_100102", "CAPTACION_BOSQUE", "hidrologico", 0.4, "Canal del Dique - Zona Franca"),
    ("IDEAM_100101", "CAPTACION_BOSQUE", "hidrologico", 0.3, "Bahía Cartagena - Rafael Núñez"),
    
    # Secuencia proceso PTAP
    ("CAPTACION_BOSQUE", "COAGULACION", "hidraulico", 1.0, "Bombeo + dosificación coagulante"),
    ("COAGULACION", "SEDIMENTACION", "hidraulico", 1.0, "Floculación -> sedimentadores"),
    ("SEDIMENTACION", "FILTRACION", "hidraulico", 1.0, "Agua clarificada -> filtros"),
    ("FILTRACION", "CLORACION", "hidraulico", 1.0, "Agua filtrada -> cloración"),
    ("CLORACION", "SALIDA_PLANTA", "hidraulico", 1.0, "Agua tratada -> distribución"),
    
    # Retroalimentación operacional (dosificación)
    ("FILTRACION", "COAGULACION", "control", 0.5, "Turbidez filtrada -> ajuste dosis"),
    ("SEDIMENTACION", "COAGULACION", "control", 0.3, "Turbidez sedimentador -> ajuste dosis"),
]

# Features por nodo (para GNN)
NODE_FEATURES_DIM = 16  # 16 features por nodo

def build_graph():
    """Construye grafo NetworkX + PyTorch Geometric Data object."""
    
    G = nx.DiGraph()
    
    # Agregar nodos
    for node_id, attrs in NODOS.items():
        G.add_node(node_id, **attrs)
    
    # Agregar edges
    for src, dst, tipo, peso, desc in EDGES:
        G.add_edge(src, dst, tipo=tipo, weight=peso, descripcion=desc)
    
    return G

def extract_node_features(ideam_df: pd.DataFrame, acuacar_df: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
    """
    Extrae features para cada nodo en un timestamp dado.
    Returns: (n_nodes, feature_dim) array
    """
    n_nodes = len(NODOS)
    node_ids = list(NODOS.keys())
    features = np.zeros((n_nodes, NODE_FEATURES_DIM), dtype=np.float32)
    
    # Filtrar datos al timestamp (o ventana cercana)
    # Para IDEAM: usar estación principal 100101 como referencia global + cada estación
    ideam_t = ideam_df[ideam_df["fecha"] == timestamp]
    acuacar_t = acuacar_df[acuacar_df["fecha"] == timestamp]
    
    # Si no hay dato exacto, buscar más cercano (hacia atrás 1h)
    if len(ideam_t) == 0:
        ideam_t = ideam_df[ideam_df["fecha"] <= timestamp].tail(1)
    if len(acuacar_t) == 0:
        acuacar_t = acuacar_df[acuacar_df["fecha"] <= timestamp].tail(1)
    
    ideam_main = ideam_df[(ideam_df["estacion_id"] == "100101") & (ideam_df["fecha"] <= timestamp)].tail(1)
    
    for i, node_id in enumerate(node_ids):
        attrs = NODOS[node_id]
        
        if attrs["tipo"] == "ideam":
            # Buscar datos de esta estación
            est_id = node_id.split("_")[1]
            est_data = ideam_t[ideam_t["estacion_id"] == est_id]
            if len(est_data) == 0:
                est_data = ideam_df[(ideam_df["estacion_id"] == est_id) & (ideam_df["fecha"] <= timestamp)].tail(1)
            
            if len(est_data) > 0:
                row = est_data.iloc[0]
                # Features meteorológicas normalizadas
                features[i, 0] = row.get("radiacion_wm2", 0) / 1000  # rad normalizada
                features[i, 1] = (row.get("temp_aire_c", 25) - 15) / 20  # temp normalizada
                features[i, 2] = row.get("precip_mm", 0) / 50  # precip normalizada
                features[i, 3] = row.get("humedad_pct", 70) / 100  # humedad
                features[i, 4] = row.get("viento_vel_ms", 3) / 15  # viento
                features[i, 5] = np.sin(2 * np.pi * row.get("hora", 12) / 24)  # hora sin
                features[i, 6] = np.cos(2 * np.pi * row.get("hora", 12) / 24)  # hora cos
                features[i, 7] = row.get("dia_semana", 3) / 6  # día semana
                features[i, 8] = 1.0  # flag: es nodo IDEAM
            else:
                features[i, 8] = 1.0
                
        elif attrs["tipo"] == "captacion":
            # Captación: condiciones agua cruda (proxy IDEAM principal + precipitación cuenca)
            if len(ideam_main) > 0:
                row = ideam_main.iloc[0]
                features[i, 0] = row.get("radiacion_wm2", 0) / 1000
                features[i, 1] = (row.get("temp_aire_c", 25) - 15) / 20
                features[i, 2] = row.get("precip_mm", 0) / 50
                features[i, 3] = row.get("humedad_pct", 70) / 100
            features[i, 9] = 1.0  # flag: captación
            
        elif attrs["tipo"] == "proceso":
            if len(acuacar_t) > 0:
                row = acuacar_t.iloc[0]
                if attrs["nombre"] == "Coagulación-Floculación":
                    features[i, 10] = row.get("dosis_coagulante_mg_l", 20) / 80  # dosis norm
                    features[i, 11] = row.get("dosis_pac_mg_l", 0) / 30  # PAC norm
                    features[i, 2] = row.get("turbidez_cruda_ntu", 30) / 500  # turbidez entrada
                    features[i, 3] = row.get("ph_cruda", 7.2) / 14  # pH
                elif attrs["nombre"] == "Sedimentación":
                    features[i, 2] = row.get("turbidez_cruda_ntu", 30) / 500
                    features[i, 10] = row.get("dosis_coagulante_mg_l", 20) / 80
                elif attrs["nombre"] == "Filtración Rápida":
                    features[i, 12] = row.get("turbidez_filtrada_ntu", 0.5) / 2.0  # turb salida
                    features[i, 13] = row.get("perdida_carga_filtros_m", 1.0) / 3.0  # pérdida carga
                    features[i, 14] = row.get("horas_desde_lavado", 24) / 120  # horas desde lavado
                    features[i, 15] = row.get("lavado_filtro", 0)  # evento lavado
                elif attrs["nombre"] == "Cloración":
                    features[i, 12] = row.get("cloro_residual_mg_l", 1.0) / 2.0
                    features[i, 10] = row.get("caudal_planta_m3_s", 3.2) / 4.0
            features[i, 10 + list(NODOS.keys()).index(node_id) % 6] = 1.0  # flag proceso
            
        elif attrs["tipo"] == "salida":
            if len(acuacar_t) > 0:
                row = acuacar_t.iloc[0]
                features[i, 12] = row.get("turbidez_filtrada_ntu", 0.5) / 2.0
                features[i, 10] = row.get("cloro_residual_mg_l", 1.0) / 2.0
                features[i, 15] = row.get("caudal_planta_m3_s", 3.2) / 4.0
            features[i, 15] = 1.0  # flag salida
    
    return features, node_ids

def build_pyg_data(ideam_df: pd.DataFrame, acuacar_df: pd.DataFrame, 
                   timestamps: list) -> list:
    """
    Construye lista de PyTorch Geometric Data objects (uno por timestamp).
    Cada grafo tiene features nodos + edge_index + edge_attr.
    """
    G = build_graph()
    node_ids = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    
    # Edge index fijo (estructura topológica no cambia)
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        # Edge features: [tipo_hidrologico, tipo_hidraulico, tipo_control, peso]
        tipo_vec = [0, 0, 0, data["weight"]]
        if data["tipo"] == "hidrologico":
            tipo_vec[0] = 1
        elif data["tipo"] == "hidraulico":
            tipo_vec[1] = 1
        elif data["tipo"] == "control":
            tipo_vec[2] = 1
        edge_attr.append(tipo_vec)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    data_list = []
    for ts in timestamps:
        node_feats, _ = extract_node_features(ideam_df, acuacar_df, ts)
        x = torch.tensor(node_feats, dtype=torch.float32)
        
        # Target: turbidez filtrada a t+1h (predicción 1 paso adelante)
        ts_next = ts + pd.Timedelta(hours=1)
        acuacar_next = acuacar_df[acuacar_df["fecha"] == ts_next]
        if len(acuacar_next) > 0:
            y = torch.tensor([acuacar_next.iloc[0]["turbidez_filtrada_ntu"]], dtype=torch.float32)
        else:
            y = torch.tensor([0.0], dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, timestamp=pd.Timestamp(ts).isoformat())
        data_list.append(data)
    
    return data_list

def main():
    print("=" * 60)
    print("CONSTRUYENDO GRAFO HIDRÁULICO GD1")
    print("=" * 60)
    
    # Cargar datos
    ideam_path = CURATED_DIR / "ideam_wide_2018_2025.parquet"
    acuacar_path = CURATED_DIR / "acuacar_operacional_2018_2025.parquet"
    
    if not ideam_path.exists() or not acuacar_path.exists():
        print("[ERROR] Faltan datasets curados. Correr pipeline primero.")
        return 1
    
    ideam_df = pd.read_parquet(ideam_path)
    acuacar_df = pd.read_parquet(acuacar_path)
    
    print(f"IDEAM: {len(ideam_df):,} registros")
    print(f"ACUACAR: {len(acuacar_df):,} registros")
    
    # Construir grafo NetworkX
    G = build_graph()
    print(f"\nGrafo NetworkX: {G.number_of_nodes()} nodos, {G.number_of_edges()} edges")
    
    # Guardar topología
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar metadatos grafo
    graph_meta = {
        "nodos": {n: {"tipo": d["tipo"], "nombre": d.get("nombre", "")} for n, d in G.nodes(data=True)},
        "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
        "node_features_dim": NODE_FEATURES_DIM,
        "edge_features_dim": 4,
        "target_variable": "turbidez_filtrada_ntu (t+1h)",
        "fecha_creacion": pd.Timestamp.now().isoformat()
    }
    
    with open(GRAPH_DIR / "graph_metadata.json", "w") as f:
        json.dump(graph_meta, f, indent=2, default=str)
    print(f"[OK] Metadata grafo: {GRAPH_DIR / 'graph_metadata.json'}")
    
    # Generar subconjunto para entrenamiento (ej. 2020-2023, cada 6h para empezar)
    print("\nGenerando dataset PyG (muestra 2020-2023, cada 6h)...")
    timestamps = pd.date_range("2020-01-01", "2023-12-31 23:00", freq="6h")
    timestamps = [ts for ts in timestamps if ts in acuacar_df["fecha"].values or 
                  any(ts >= d for d in acuacar_df["fecha"].values)]
    
    # Tomar muestra para test rápido
    timestamps_sample = timestamps[::24]  # 1 por día
    print(f"Timestamps para dataset: {len(timestamps_sample)}")
    
    data_list = build_pyg_data(ideam_df, acuacar_df, timestamps_sample[:100])  # Primeros 100 para test
    
    # Guardar dataset PyG
    torch.save(data_list, GRAPH_DIR / "graph_dataset_sample.pt")
    print(f"[OK] Dataset PyG guardado: {GRAPH_DIR / 'graph_dataset_sample.pt'} ({len(data_list)} grafos)")
    
    # Verificar shapes
    sample = data_list[0]
    print(f"\n--- EJEMPLO GRAFO ---")
    print(f"  Nodos: {sample.x.shape[0]} x {sample.x.shape[1]} features")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Edge attr: {sample.edge_attr.shape}")
    print(f"  Target (y): {sample.y.item():.4f} NTU")
    print(f"  Timestamp: {sample.timestamp}")
    
    # Guardar grafo NetworkX para visualización
    import pickle
    with open(GRAPH_DIR / "hydraulic_graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    print(f"[OK] Grafo NetworkX: {GRAPH_DIR / 'hydraulic_graph.gpickle'}")
    
    return 0

if __name__ == "__main__":
    exit(main())