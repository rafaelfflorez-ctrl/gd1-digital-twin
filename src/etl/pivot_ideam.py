"""
Transforma datos IDEAM de formato largo a ancho (pivot) para modelado.
Une con datos operacionales ACUACAR cuando estén disponibles.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
CURATED_DIR = Path(__file__).parent.parent.parent / "data" / "curated"

def pivot_ideam_data():
    """Convierte datos IDEAM de long a wide format."""
    df = pd.read_parquet(RAW_DIR / "ideam_2018_2025.parquet")
    
    # Pivot: fecha + estacion_id como índice, variables como columnas
    df_wide = df.pivot_table(
        index=["fecha", "estacion_id"],
        columns="variable",
        values="valor",
        aggfunc="mean"
    ).reset_index()
    
    # Aplanar nombres de columnas
    df_wide.columns.name = None
    
    # Renombrar columnas para claridad
    rename_map = {
        "radiacion": "radiacion_wm2",
        "temperatura": "temp_aire_c",
        "precipitacion": "precip_mm",
        "viento_vel": "viento_vel_ms",
        "viento_dir": "viento_dir_grados",
        "humedad": "humedad_pct",
    }
    df_wide = df_wide.rename(columns=rename_map)
    
    # Agregar features temporales
    df_wide["hora"] = df_wide["fecha"].dt.hour
    df_wide["dia_semana"] = df_wide["fecha"].dt.day_of_week
    df_wide["mes"] = df_wide["fecha"].dt.month
    df_wide["dia_año"] = df_wide["fecha"].dt.day_of_year
    df_wide["es_fin_semana"] = df_wide["dia_semana"].isin([5, 6]).astype(int)
    
    # Features cíclicas para hora y día del año
    df_wide["hora_sin"] = np.sin(2 * np.pi * df_wide["hora"] / 24)
    df_wide["hora_cos"] = np.cos(2 * np.pi * df_wide["hora"] / 24)
    df_wide["dia_año_sin"] = np.sin(2 * np.pi * df_wide["dia_año"] / 365)
    df_wide["dia_año_cos"] = np.cos(2 * np.pi * df_wide["dia_año"] / 365)
    
    # Guardar
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CURATED_DIR / "ideam_wide_2018_2025.parquet"
    df_wide.to_parquet(output_file, index=False)
    
    print("[OK] Pivot completado: {:,} registros".format(len(df_wide)))
    print("   Columnas: {}".format(list(df_wide.columns)))
    print("   Guardado en: {}".format(output_file))
    
    return df_wide

def agregar_features_estacionales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de estacionalidad y rezagos para series temporales."""
    df = df.copy()
    df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)
    
    # Variables objetivo para modelado
    target_vars = ["radiacion_wm2", "temp_aire_c", "precip_mm", "humedad_pct"]
    
    for est in df["estacion_id"].unique():
        mask = df["estacion_id"] == est
        for var in target_vars:
            if var in df.columns:
                # Rezagos (lags) 1, 3, 6, 12, 24 horas
                for lag in [1, 3, 6, 12, 24]:
                    df.loc[mask, f"{var}_lag{lag}h"] = df.loc[mask, var].shift(lag)
                
                # Rolling stats 6h, 24h
                for window in [6, 24]:
                    df.loc[mask, f"{var}_roll{window}h_mean"] = df.loc[mask, var].rolling(window, min_periods=1).mean()
                    df.loc[mask, f"{var}_roll{window}h_std"] = df.loc[mask, var].rolling(window, min_periods=1).std()
                    df.loc[mask, f"{var}_roll{window}h_max"] = df.loc[mask, var].rolling(window, min_periods=1).max()
                    df.loc[mask, f"{var}_roll{window}h_min"] = df.loc[mask, var].rolling(window, min_periods=1).min()
    
    # Eliminar filas con NaN en targets (primeros registros por estación)
    df = df.dropna(subset=target_vars).reset_index(drop=True)
    
    return df

def main():
    print("=" * 60)
    print("PIVOT IDEAM + FEATURE ENGINEERING")
    print("=" * 60)
    
    # 1. Pivot
    df_wide = pivot_ideam_data()
    
    # 2. Feature engineering temporal
    print("\n[Agregando features temporales (lags, rolling stats)...]")
    df_features = agregar_features_estacionales(df_wide)
    
    # Guardar dataset con features
    output_file = CURATED_DIR / "ideam_features_2018_2025.parquet"
    df_features.to_parquet(output_file, index=False)
    
    print("\n[OK] Dataset final con features: {:,} registros".format(len(df_features)))
    print("   Columnas: {}".format(len(df_features.columns)))
    print("   Guardado en: {}".format(output_file))
    
    # Resumen de variables
    target_vars = [c for c in df_features.columns if any(t in c for t in ["radiacion", "temp_aire", "precip", "humedad"]) and "lag" not in c and "roll" not in c]
    feature_vars = [c for c in df_features.columns if "lag" in c or "roll" in c]
    print("\n   Variables target: {}".format(len(target_vars)))
    print("   Features temporales: {}".format(len(feature_vars)))
    print("   Total features: {}".format(len(df_features.columns)))

if __name__ == "__main__":
    main()