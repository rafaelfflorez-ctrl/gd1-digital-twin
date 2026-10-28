"""
Validación de datos IDEAM con Great Expectations (API v0.18+).
Suite de expectativas para garantizar calidad de datos.
"""
import great_expectations as gx
import pandas as pd
from pathlib import Path

CURATED_DIR = Path(__file__).parent.parent.parent / "data" / "curated"
GX_DIR = Path(__file__).parent.parent.parent / "great_expectations"

def get_or_create_suite(context):
    """Obtiene o crea suite de expectativas para datos IDEAM."""
    
    try:
        suite = context.suites.get("ideam_data_quality")
        print("[OK] Suite 'ideam_data_quality' ya existe, obtenida")
    except:
        suite = context.suites.add(
            gx.ExpectationSuite(name="ideam_data_quality")
        )
        
        expectations = [
            gx.expectations.ExpectTableColumnsToMatchOrderedList(
                column_list=[
                    "fecha", "estacion_id", "humedad_pct", "precip_mm", 
                    "radiacion_wm2", "temp_aire_c", "viento_dir_grados", 
                    "viento_vel_ms", "hora", "dia_semana", "mes", "dia_año",
                    "es_fin_semana", "hora_sin", "hora_cos", "dia_año_sin", "dia_año_cos"
                ]
            ),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="fecha"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="estacion_id"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="radiacion_wm2"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="temp_aire_c"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="precip_mm"),
            gx.expectations.ExpectColumnValuesToBeBetween(column="radiacion_wm2", min_value=0, max_value=1400),
            gx.expectations.ExpectColumnValuesToBeBetween(column="temp_aire_c", min_value=-5, max_value=45),
            gx.expectations.ExpectColumnValuesToBeBetween(column="precip_mm", min_value=0, max_value=500),
            gx.expectations.ExpectColumnValuesToBeBetween(column="humedad_pct", min_value=0, max_value=100),
            gx.expectations.ExpectColumnValuesToBeBetween(column="viento_vel_ms", min_value=-2, max_value=50),
            gx.expectations.ExpectColumnValuesToBeBetween(column="viento_dir_grados", min_value=0, max_value=360),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="estacion_id",
                value_set=["100101", "100102", "100103", "100104", "100105", "100106", "100107", "100108"]
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(column="hora", min_value=0, max_value=23),
            gx.expectations.ExpectColumnValuesToBeBetween(column="mes", min_value=1, max_value=12),
            gx.expectations.ExpectColumnValuesToBeBetween(column="dia_semana", min_value=0, max_value=6),
            gx.expectations.ExpectTableRowCountToBeBetween(min_value=500000, max_value=700000),
            gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["fecha", "estacion_id"]),
        ]
        
        for exp in expectations:
            suite.add_expectation(exp)
        
        context.suites.add(suite)
        print("[OK] Suite 'ideam_data_quality' creada")
    
    return suite

def get_or_create_batch_definition(datasource):
    """Obtiene o crea batch definition."""
    try:
        asset = datasource.get_asset("ideam_wide")
    except:
        asset = datasource.add_dataframe_asset(name="ideam_wide")
    
    try:
        batch_def = asset.get_batch_definition("ideam_batch")
    except:
        batch_def = asset.add_batch_definition_whole_dataframe("ideam_batch")
    
    return batch_def

def run_validation():
    """Ejecuta validación sobre dataset curado."""
    context = gx.get_context(mode="file", project_root_dir=str(GX_DIR.parent))
    
    # Cargar datos
    df = pd.read_parquet(CURATED_DIR / "ideam_wide_2018_2025.parquet")
    
    # Crear/obtener data source
    try:
        datasource = context.data_sources.get("ideam_pandas")
    except:
        datasource = context.data_sources.add_pandas("ideam_pandas")
    
    # Batch definition
    batch_def = get_or_create_batch_definition(datasource)
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    
    # Validar
    suite = context.suites.get("ideam_data_quality")
    validation_result = batch.validate(suite)
    
    print("\n" + "=" * 60)
    print("RESULTADO VALIDACION IDEAM")
    print("=" * 60)
    print("Exito: {}".format(validation_result.success))
    print("Expectativas evaluadas: {}".format(len(validation_result.results)))
    
    failed = [r for r in validation_result.results if not r.success]
    if failed:
        print("\n[FAIL] EXPECTATIVAS FALLADAS ({}):".format(len(failed)))
        for r in failed:
            exp_type = r.expectation_config.type
            kwargs = r.expectation_config.kwargs
            print("  - {}: {}".format(exp_type, kwargs))
            if r.result:
                print("    Detalle: {}".format(r.result))
    else:
        print("\n[OK] TODAS LAS EXPECTATIVAS PASARON")
    
    return validation_result

def main():
    print("=" * 60)
    print("GREAT EXPECTATIONS - VALIDACION IDEAM")
    print("=" * 60)
    
    context = gx.get_context(mode="file", project_root_dir=str(GX_DIR.parent))
    
    # Obtener/crear suite
    get_or_create_suite(context)
    
    # Ejecutar validación
    result = run_validation()
    
    if result.success:
        print("\n[OK] Validacion exitosa - Datos listos para modelado")
        return 0
    else:
        print("\n[ERROR] Validacion fallo - Revisar datos")
        return 1

if __name__ == "__main__":
    exit(main())