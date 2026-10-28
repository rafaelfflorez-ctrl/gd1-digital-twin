"""
Descarga de datos históricos IDEAM (radiación, temperatura, precipitación, viento)
Período: 2018-2025
Estaciones: Sistema Juan Gómez-Dolores y alrededores Cartagena
"""
import requests
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

IDEAM_BASE_URL = "https://www.ideam.gov.co/web/tiempo-y-clima/servicios-web"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Estaciones IDEAM relevantes para Cartagena y cuenca Juan Gómez-Dolores
ESTACIONES_CARTAGENA = {
    "100101": {"nombre": "CARTAGENA RAFAEL NUÑEZ", "lat": 10.442, "lon": -75.513, "alt": 2},
    "100102": {"nombre": "CARTAGENA ZONA FRANCA", "lat": 10.383, "lon": -75.433, "alt": 10},
    "100103": {"nombre": "TURBACO", "lat": 10.333, "lon": -75.383, "alt": 100},
    "100104": {"nombre": "ARJONA", "lat": 10.267, "lon": -75.267, "alt": 50},
    "100105": {"nombre": "MARIA LA BAJA", "lat": 10.167, "lon": -75.167, "alt": 80},
    "100106": {"nombre": "SAN JUAN NEPOMUCENO", "lat": 9.967, "lon": -75.083, "alt": 120},
    "100107": {"nombre": "EL CARMEN DE BOLIVAR", "lat": 9.717, "lon": -75.117, "alt": 180},
    "100108": {"nombre": "SAN ONOFRE", "lat": 9.583, "lon": -75.167, "alt": 100},
}

VARIABLES = {
    "radiacion": "Radiación solar global",
    "temperatura": "Temperatura del aire",
    "precipitacion": "Precipitación",
    "viento_vel": "Velocidad del viento",
    "viento_dir": "Dirección del viento",
    "humedad": "Humedad relativa",
}

def descargar_estacion_ideam(estacion_id: str, variable: str, año_inicio: int = 2018, año_fin: int = 2025) -> pd.DataFrame:
    """
    Descarga datos de una estación IDEAM para una variable específica.
    Nota: IDEAM no tiene API pública oficial documentada. 
    Esta función es un template - requiere adaptación al servicio real disponible.
    """
    # PLACEHOLDER: IDEAM requiere solicitud formal o uso de su portal web
    # Para producción: usar IDEAM Open Data o solicitar credenciales
    print(f"  [SIMULADO] Descargando {variable} para estación {estacion_id} ({año_inicio}-{año_fin})")
    
    # Generar datos sintéticos realistas para desarrollo
    fechas = pd.date_range(f"{año_inicio}-01-01", f"{año_fin}-12-31", freq="h")
    n = len(fechas)
    
    if variable == "radiacion":
        # Radiación: ciclo diario + estacional + ruido (no negativo)
        hora = fechas.hour
        dia_año = fechas.day_of_year
        base = 800 * np.sin(2 * np.pi * (hora - 6) / 24) * np.sin(2 * np.pi * dia_año / 365)
        base = np.maximum(base, 0)
        ruido = np.random.normal(0, 50, n)
        valores = np.maximum(base + ruido, 0)
    elif variable == "temperatura":
        # Temperatura: ciclo diario + estacional
        hora = fechas.hour
        dia_año = fechas.day_of_year
        base = 28 + 3 * np.sin(2 * np.pi * (hora - 14) / 24) + 1.5 * np.sin(2 * np.pi * dia_año / 365)
        valores = base + np.random.normal(0, 1.5, n)
    elif variable == "precipitacion":
        # Precipitación: eventos esporádicos
        valores = np.random.exponential(0.5, n) * (np.random.random(n) < 0.05)
    elif variable in ["viento_vel", "viento_dir"]:
        valores = np.random.weibull(2, n) * 5 + np.random.normal(0, 1, n)
    elif variable == "humedad":
        valores = 75 + 15 * np.sin(2 * np.pi * fechas.hour / 24) + np.random.normal(0, 5, n)
        valores = np.clip(valores, 30, 100)
    else:
        valores = np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        "fecha": fechas,
        "estacion_id": estacion_id,
        "variable": variable,
        "valor": valores,
        "unidad": get_unidad(variable)
    })
    return df

def get_unidad(variable: str) -> str:
    unidades = {
        "radiacion": "W/m2",
        "temperatura": "°C",
        "precipitacion": "mm",
        "viento_vel": "m/s",
        "viento_dir": "grados",
        "humedad": "%",
    }
    return unidades.get(variable, "")

def main():
    import numpy as np
    np.random.seed(42)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    print("=" * 60)
    print("DESCARGA DATOS IDEAM - SISTEMA JUAN GÓMEZ-DOLORES")
    print("Período: 2018-2025 | Variables: Radiación, Temp, Prec, Viento, Humedad")
    print("=" * 60)
    
    for est_id, est_info in tqdm(ESTACIONES_CARTAGENA.items(), desc="Estaciones"):
        for var in tqdm(VARIABLES.keys(), desc=f"  {est_info['nombre']}", leave=False):
            df = descargar_estacion_ideam(est_id, var, 2018, 2025)
            all_data.append(df)
            time.sleep(0.1)  # Rate limiting simulado
    
    # Consolidar
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Guardar en formato Parquet (eficiente)
    output_file = DATA_DIR / "ideam_2018_2025.parquet"
    df_final.to_parquet(output_file, index=False)
    
    # Metadata
    metadata = {
        "fuente": "IDEAM (simulado para desarrollo)",
        "periodo": "2018-01-01 a 2025-12-31",
        "frecuencia": "Horaria",
        "estaciones": ESTACIONES_CARTAGENA,
        "variables": VARIABLES,
        "registros_totales": len(df_final),
        "fecha_descarga": datetime.now().isoformat(),
        "nota": "DATOS SINTÉTICOS PARA DESARROLLO - REEMPLAZAR CON DATOS REALES IDEAM"
    }
    
    with open(DATA_DIR / "ideam_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("\n[OK] Completado: {:,} registros guardados en {}".format(len(df_final), output_file))
    print("[INFO] Metadata en: {}".format(DATA_DIR / 'ideam_metadata.json'))
    print("\n[IMPORTANTE] Reemplazar datos sintéticos con datos reales IDEAM")
    print("   Opciones: 1) Portal IDEAM Open Data  2) Solicitud formal via Freddy  3) API SIATA/otros")

if __name__ == "__main__":
    main()