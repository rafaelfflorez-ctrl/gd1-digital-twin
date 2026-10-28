"""
Generador de datos operacionales ACUACAR sintéticos realistas.
Basado en ingeniería de PTAP convencional (coag-floc-sed-filtración).
Clima tropical Cartagena - PTAP El Bosque ~3.5 m3/s.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CURATED_DIR = Path(__file__).parent.parent.parent / "data" / "curated"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

def generar_datos_acuacar_sinteticos(ideam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera datos operacionales sintéticos correlacionados con IDEAM.
    Una fila por hora (alineado con IDEAM estación principal).
    """
    np.random.seed(12345)
    
    df_ideam_main = ideam_df[ideam_df["estacion_id"] == "100101"].copy()
    df_ideam_main = df_ideam_main.set_index("fecha").sort_index()
    
    n = len(df_ideam_main)
    fechas = df_ideam_main.index
    
    precip = df_ideam_main["precip_mm"].values
    temp = df_ideam_main["temp_aire_c"].values
    humedad = df_ideam_main["humedad_pct"].values
    radiacion = df_ideam_main["radiacion_wm2"].values
    
    # === TURBIDEZ CRUDA ===
    turbidez_base = 20
    lluvia_effect = np.convolve(precip, np.exp(-np.arange(6)/2), mode='full')[:n]
    lluvia_effect = lluvia_effect / (lluvia_effect.max() + 1e-6) * 300
    turbidez_cruda = turbidez_base + lluvia_effect + np.random.lognormal(0, 0.5, n) * 10
    turbidez_cruda = np.clip(turbidez_cruda, 3, 800)
    
    # === PH CRUDA ===
    ph_base = 7.2 + 0.3 * np.sin(2 * np.pi * fechas.day_of_year / 365)
    ph_cruda = ph_base + np.random.normal(0, 0.15, n)
    ph_cruda = np.clip(ph_cruda, 6.2, 8.3)
    
    # === CONDUCTIVIDAD ===
    cond_base = 250 + 50 * np.sin(2 * np.pi * fechas.day_of_year / 365)
    conductividad = cond_base + np.random.normal(0, 20, n)
    conductividad = np.clip(conductividad, 100, 500)
    
    # === DOSIS COAGULANTE ===
    dosis_base = 15 + 0.08 * turbidez_cruda
    ph_factor = 1 + 0.3 * np.abs(ph_cruda - 7.0)
    temp_factor = 1 + 0.02 * (25 - temp)
    dosis_coagulante = dosis_base * ph_factor * temp_factor
    dosis_coagulante += np.random.normal(0, 2, n)
    dosis_coagulante = np.clip(dosis_coagulante, 5, 80)
    
    # === PAC ===
    indice_algal = (radiacion / 1000) * (temp / 35) * (1 - np.minimum(precip / 10, 1))
    dosi_pac = 5 + 15 * indice_algal + np.random.exponential(2, n)
    dosi_pac = np.clip(dosi_pac, 0, 30)
    dosi_pac[indice_algal < 0.15] = 0
    
    # === SIMULAR CICLOS DE FILTROS CON LAVADOS ===
    # Múltiples filtros (ej. 8 filtros) desfasados - cada filtro estado independiente
    n_filtros = 8
    turbidez_filtrada_final = np.zeros(n)
    perdida_carga_final = np.zeros(n)
    lavado_final = np.zeros(n, dtype=int)
    horas_desde_lavado_final = np.zeros(n, dtype=float)
    
    for f in range(n_filtros):
        # Offset inicial aleatorio por filtro
        offset = np.random.randint(0, 48)
        
        # Estado del filtro - independiente por filtro
        hdl = offset
        pc = 0.3
        
        turb_f = np.zeros(n)
        pc_arr = np.zeros(n)
        lavado_arr = np.zeros(n, dtype=int)
        hdl_arr = np.zeros(n, dtype=float)
        
        for i in range(n):
            # Eficiencia remoción basada en dosis y turbidez entrada (variable, alta para planta bien operada)
            # Plantas convencionales bien operadas: 99-99.9% remoción turbidez
            ef_rem = 0.992 + 0.006 * (1 - np.exp(-dosis_coagulante[i] / 15))
            ef_rem *= (0.96 + 0.04 * (1 - np.exp(-turbidez_cruda[i] / 50)))
            # Variabilidad operacional (días malos, picos)
            ef_rem += np.random.normal(0, 0.0015)
            ef_rem = np.clip(ef_rem, 0.985, 0.9995)
            
            turb_f[i] = turbidez_cruda[i] * (1 - ef_rem)
            # Degradación por tiempo entre lavados
            turb_f[i] *= (1 + 0.015 * hdl)
            # Ruido de medición + variabilidad proceso
            turb_f[i] += np.random.lognormal(-4.5, 0.4)
            turb_f[i] = np.clip(turb_f[i], 0.01, 1.0)
            
            # Pérdida carga
            pc = 0.3 + 0.015 * hdl * (1 + turbidez_cruda[i] / 400)
            pc += np.random.normal(0, 0.03)
            pc_arr[i] = np.clip(pc, 0.2, 3.0)
            
            # Lavado si pc > 2.2
            if pc > 2.2:
                lavado_arr[i] = 1
                pc = 0.3
                hdl = 0
            else:
                hdl += 1
            
            hdl_arr[i] = hdl
        
        # Promediar entre filtros (planta reporta promedio)
        turbidez_filtrada_final += turb_f / n_filtros
        perdida_carga_final += pc_arr / n_filtros
        lavado_final += lavado_arr
        horas_desde_lavado_final += hdl_arr / n_filtros
    
    lavado_final = (lavado_final > 0).astype(int)  # al menos un filtro lavado
    # horas_desde_lavado ya es float promedio
    
    # === CLORO RESIDUAL ===
    cloro_residual = 1.0 + 0.2 * np.sin(2 * np.pi * fechas.hour / 24) + np.random.normal(0, 0.1, n)
    cloro_residual = np.clip(cloro_residual, 0.3, 1.8)
    
    # === CAUDAL PLANTA ===
    caudal = 3.2 + 0.3 * np.sin(2 * np.pi * fechas.hour / 24) + np.random.normal(0, 0.05, n)
    caudal = np.clip(caudal, 2.5, 3.8)
    
    # === CONSTRUIR DATAFRAME ===
    df_acuacar = pd.DataFrame({
        "fecha": fechas,
        "estacion_id": "PTAP_EL_BOSQUE",
        "turbidez_cruda_ntu": turbidez_cruda.round(2),
        "ph_cruda": ph_cruda.round(2),
        "conductividad_us_cm": conductividad.round(1),
        "dosis_coagulante_mg_l": dosis_coagulante.round(2),
        "dosis_pac_mg_l": dosi_pac.round(2),
        "turbidez_filtrada_ntu": turbidez_filtrada_final.round(3),
        "perdida_carga_filtros_m": perdida_carga_final.round(2),
        "cloro_residual_mg_l": cloro_residual.round(2),
        "caudal_planta_m3_s": caudal.round(2),
        "lavado_filtro": lavado_final,
        "horas_desde_lavado": horas_desde_lavado_final,
    })
    
    return df_acuacar

def main():
    print("=" * 60)
    print("GENERANDO DATOS ACUACAR SINTETICOS REALISTAS v2")
    print("PTAP El Bosque - 8 filtros paralelos - Correlacionado IDEAM")
    print("=" * 60)
    
    ideam_path = CURATED_DIR / "ideam_wide_2018_2025.parquet"
    if not ideam_path.exists():
        print(f"[ERROR] No existe {ideam_path}. Correr pipeline IDEAM primero.")
        return 1
    
    ideam_df = pd.read_parquet(ideam_path)
    print(f"IDEAM cargado: {len(ideam_df):,} registros, {ideam_df['estacion_id'].nunique()} estaciones")
    
    df_acuacar = generar_datos_acuacar_sinteticos(ideam_df)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DIR / "acuacar_operacional_sintetico.parquet"
    df_acuacar.to_parquet(raw_file, index=False)
    print(f"\n[OK] Raw guardado: {raw_file} ({len(df_acuacar):,} registros)")
    
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    curated_file = CURATED_DIR / "acuacar_operacional_2018_2025.parquet"
    df_acuacar.to_parquet(curated_file, index=False)
    print(f"[OK] Curated guardado: {curated_file}")
    
    metadata = {
        "fuente": "SINTETICO v2 - Ingeniería inversa PTAP El Bosque (8 filtros paralelos)",
        "periodo": "2018-01-01 a 2025-12-31",
        "frecuencia": "Horaria",
        "planta": "PTAP El Bosque (Cartagena)",
        "capacidad_nominal_m3_s": 3.5,
        "n_filtros_paralelos": 8,
        "proceso": "Coagulación-Floculación-Sedimentación-Filtración rápida-Cloración",
        "coagulante": "Sulfato de aluminio (Al2(SO4)3)",
        "variables": {
            "turbidez_cruda_ntu": "Turbidez agua cruda (NTU) - correlaciona con precipitación IDEAM",
            "ph_cruda": "pH agua cruda",
            "conductividad_us_cm": "Conductividad (µS/cm)",
            "dosis_coagulante_mg_l": "Dosis sulfato aluminio (mg/L) - función turbidez+pH+temp",
            "dosis_pac_mg_l": "Dosis carbón activado polvo (mg/L) - eventos algales",
            "turbidez_filtrada_ntu": "Turbidez salida filtros promedio (NTU) - meta < 0.1",
            "perdida_carga_filtros_m": "Pérdida de carga filtros promedio (m) - trigger lavado > 2.2m",
            "cloro_residual_mg_l": "Cloro residual salida planta (mg/L)",
            "caudal_planta_m3_s": "Caudal tratado (m3/s)",
            "lavado_filtro": "Evento lavado al menos 1 filtro (0/1)",
            "horas_desde_lavado": "Horas promedio desde último lavado",
        },
        "fecha_generacion": datetime.now().isoformat(),
        "nota": "DATOS SINTETICOS PARA DESARROLLO GD1. REEMPLAZAR CON DATOS REALES ACUACAR CUANDO LLEGUEN."
    }
    
    import json
    with open(RAW_DIR / "acuacar_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[OK] Metadata: {RAW_DIR / 'acuacar_metadata.json'}")
    
    print("\n--- RESUMEN ESTADÍSTICO ---")
    for col in df_acuacar.select_dtypes(include=[np.number]).columns:
        if col not in ["estacion_id", "lavado_filtro"]:
            print(f"  {col}: mean={df_acuacar[col].mean():.2f}, std={df_acuacar[col].std():.2f}, "
                  f"min={df_acuacar[col].min():.2f}, max={df_acuacar[col].max():.2f}")
    
    print(f"\n  Lavados totales (horas con >=1 filtro): {df_acuacar['lavado_filtro'].sum()}")
    if df_acuacar['lavado_filtro'].sum() > 0:
        print(f"  Frec. lavados: cada {len(df_acuacar)/df_acuacar['lavado_filtro'].sum():.0f}h promedio")
    print(f"  Horas desde lavado - mean: {df_acuacar['horas_desde_lavado'].mean():.1f}, max: {df_acuacar['horas_desde_lavado'].max()}")
    
    return 0

if __name__ == "__main__":
    exit(main())