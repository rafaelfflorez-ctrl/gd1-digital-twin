"""
Pipeline ETL principal - Orquesta todo el proceso de datos.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

def run_script(script_name: str, description: str) -> bool:
    """Ejecuta un script y reporta resultado."""
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[OK] {description} - EXITOSO")
        if result.stdout:
            print(result.stdout[-500:])  # Últimos 500 chars
        return True
    else:
        print(f"[FAIL] {description} - FALLÓ")
        print(result.stderr[-1000:])
        return False

def main():
    print("=" * 60)
    print("PIPELINE ETL COMPLETO - GEMMELO DIGITAL 1 (GD1)")
    print("Fase 1: Datos IDEAM -> Curated -> Validado")
    print("=" * 60)
    
    steps = [
        ("download_ideam.py", "Descarga datos IDEAM (simulado)"),
        ("pivot_ideam.py", "Pivot + Feature Engineering"),
        ("validate_ideam.py", "Validación Great Expectations"),
    ]
    
    all_ok = True
    for script, desc in steps:
        ok = run_script(script, desc)
        if not ok:
            all_ok = False
            print(f"\n[ERROR] Pipeline detenido en: {desc}")
            break
    
    print("\n" + "=" * 60)
    if all_ok:
        print("PIPELINE ETL COMPLETADO EXITOSAMENTE")
        print("Datos listos en: data/curated/")
        print("  - ideam_wide_2018_2025.parquet (pivot)")
        print("  - ideam_features_2018_2025.parquet (con features temporales)")
        print("Validación Great Expectations: PASÓ")
    else:
        print("PIPELINE FALLÓ - Revisar errores arriba")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())