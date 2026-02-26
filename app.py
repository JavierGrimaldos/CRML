from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import json
import sys
import os
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app=Flask(__name__, static_folder="static")
CORS(app)

with open("modelos.json","r",encoding="UTF-8")as f:
    MODELS_DATA=json.load(f)

try:
    from analizador import dataAnalisis
    from analizador import calcular_consenso_semantico, imprimir_matriz_consenso
    from Ensambladores.ensamblador_LLM import Ensamblador
    MODULES_OK=True
    print("Módulos importados correctamente")
except ImportError as e:
    print(f"Error importando módulos: {e}")
    traceback.print_exc()
    MODULES_OK=False

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/api/health",methods=["GET"])
def health():
    return jsonify({
        'status': 'ok',
        'modules': MODULES_OK,
        'time': datetime.now().isoformat()
    })

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        'success': True,
        'free_models': MODELS_DATA['LLM']['FREE_MODELS'],
        'pay_models': MODELS_DATA['LLM']['PAY_MODELS']
    })

def format_consenso_data(report, results):
    if not report or "error" in report:
        return None
    consensos_individuales=[]
    if "consensos_individuales" in report and report["consensos_individuales"]:
        for consenso in report["consensos_individuales"]:
            consensos_individuales.append({
               'modelo': consenso.get('modelo', ''),
                'consenso_individual': consenso.get('consenso_individual', 0),
                'respuesta_idx': consenso.get('respuesta_idx', 0) 
            })
    elif results and len(results)>0:
        for i, result in enumerate(result):
            consensos_individuales.append({
                'modelo': result.get('model_name', f'Modelo_{i}'),
                'consenso_individual': 0.0,
                'respuesta_idx': i
            })
    
    modelo_mas_consensuado=None
    if consensos_individuales:
        consensos_validos=[c for c in consensos_individuales if c["consenso_individual"]>0]
        if consensos_validos:
            modelo_mas_consensuado=max(consensos_validos, key=lambda x: x["consenso_individual"])
            print(f"Modelo más consensuado identificado: {modelo_mas_consensuado["modelo"]} (consenso: {modelo_mas_consensuado["consenso_individual"]:.3f})")
    
    top3_modelos=[]
    if consensos_individuales:
        consensos_validos=[c for c in consensos_individuales if c["consenso_individual"]>0]
        if consensos_validos:
            consensos_ordenados=sorted(consensos_validos, key=lambda x:x["consenso_individual"], reverse=True)
            top_cuenta=min(3, len(consensos_ordenados))
            top3_modelos=[item["modelo"] for item in consensos_ordenados[:top_cuenta]]
            print(f"Top {top_cuenta} modelos identificados: {top3_modelos}")

    formateado={
        'consenso_global': report.get('consenso_global', 0.0),
        'consensos_individuales': consensos_individuales,
        'nombres_filtrados': report.get('nombres_filtrados', []),
        'top3_modelos': top3_modelos,
        'modelo_mas_consensuado': modelo_mas_consensuado['modelo'] if modelo_mas_consensuado else None
    }

    if 'respuesta_mas_consensuada' in report and report['respuesta_mas_consensuada']:
        formateado['respuesta_mas_consensuada'] = {
            'modelo': report['respuesta_mas_consensuada'].get('modelo', ''),
            'consenso_individual': report['respuesta_mas_consensuada'].get('consenso_individual', 0)
        }
    
    if 'respuesta_fusionada' in report:
        formateado['respuesta_fusionada'] = report['respuesta_fusionada']
        formateado['modelos_base'] = report.get('modelos_base', [])
    
    return formateado

@app.route("/api/run-ensemble", methods=["POST"])
def run_ensamble():
    if not MODULES_OK:
        return jsonify({"success":False, "error":"Módulos de análisis no disponibles"})
    
    try:
        data=request.json
        prompt=data.get("prompt","").strip()
        model_names=data.get("models",[])
        model_type=data.get("modelType","free")

        print(f"\t\tEJECUTANDO ENSAMBLE\nPrompt: {prompt}\nModelos seleccionados: {len(model_names)}\nTipo: {model_type}")

        if not prompt: return jsonify({"success":False,"Error":"El prompt no puede estar vacío"})
        if not model_names: return jsonify({"succes":False,"Error":"Debe seleccionar al menos un modelo"})
        if model_type == "free": source_models=MODELS_DATA["LLM"]["FREE_MODELS"]
        else: source_models=MODELS_DATA["LLM"]["PAY_MODELS"]
        
        modelos=[]
        for nombre_modelo in model_names:
            for modelo in source_models:
                if modelo["name"]==nombre_modelo:
                    modelos.append(modelo)
                    break
        if not modelos: return jsonify({"success":False, "error":"No se encontraron los modelos seleccionados"})

        async def ejecutar_ensamble():
            print("1. Iniciando consulta a modelos.")
            try:
                ensamble=Ensamblador(modelos=modelos)
                resultados=await ensamble.run(prompt)
                print(f"2.{len(resultados)} respuestas recibidas.")
                if hasattr(ensamble, "save_results"): ensamble.save_results(resultados)
                respuestas_texto=[]
                nombres_modelos=[]
                resultados_formateados=[]

                for i,r in enumerate(resultados):
                    nombre_modelo=r.get("model_name", f"Modelo_{i}")
                    respuesta=r.get("response", "")
                    timestamp=r.get("timestamp",datetime.now().isoformat())

                    respuestas_texto.append(respuesta)
                    nombres_modelos.append(nombre_modelo)
                    resultados_formateados.append({
                        'model_name': nombre_modelo,
                        'response': respuesta,
                        'timestamp': timestamp,
                        'index': i
                    })
                
                print("3. Ejecutando análisis de consenso...")
                resultado_consenso=calcular_consenso_semantico(respuestas_texto, nombres_modelos)
                if "matriz_consenso" in resultado_consenso:
                    imprimir_matriz_consenso(resultado_consenso["matriz_consenso"],resultado_consenso.get("nombres_filtrados", nombres_modelos))
                
                print("4. Ejecutando análisis completo...")
                reporte=dataAnalisis(resultados)
                if "consensos_individuales" not in reporte and "consensos_individuales" in resultado_consenso: reporte["consensos_individuales"]=resultado_consenso["consensos_individuales"]
                if "consenso_global" not in reporte and "consenso_global" in resultado_consenso: reporte["consenso_global"]=resultado_consenso["consenso_global"]
                if "mayores_consensos" in resultado_consenso: reporte["mayores_consensos"]=resultado_consenso["mayores_consensos"]

                consenso_data=format_consenso_data(reporte, resultados)
                if not consenso_data or not consenso_data.get("consensos_individuales"): consenso_data=format_consenso_data(resultado_consenso, resultados)

                print("5. Análisis completado")
                consenso_global=reporte.get("consenso_global") or resultado_consenso.get("consenso_global", 0)
                print(f"Consenso global: {consenso_global:.3f}")

                if consenso_data and consenso_data.get("modelo_mas_consensuado"): print(f"Modelo más consensuado: {consenso_data["modelo_mas_consensuado"]}")
                if consenso_data and consenso_data.get("top3_modelos"): print(f"Top{len(consenso_data["top3_modelos"])} modelos: {consenso_data["top3_modelos"]}")

                return {
                    'success': True,
                    'results': resultados_formateados,
                    'report': reporte,
                    'consenso_data': consenso_data,
                    'prompt': prompt,
                    'models_count': len(modelos),
                    'timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                print(f"Error en ejecutar_ensamble: {e}")
                traceback.print_exc()
                raise
            
        resultado=asyncio.run(ejecutar_ensamble())
        return jsonify(resultado)
    
    except Exception as e:
        print(f"Error general: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error en el servidor: {str(e)}'
        })
    
if __name__=="__main__":
    print("\t\t CONSENSUADOR DE RESPUESTAS DE MODELOS DE LENGUAJE")
    print(f"Directorio actual: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Módulos cargados: {MODULES_OK}")
    print(f"URL: http://localhost:8282")
    print(f"\nServidor listo. Presione Ctrl+C para detenerlo.")

    app.run(
        host="0.0.0.0",
        port=8282,
        debug=True,
        threaded=True,
        use_reloader=False
    )