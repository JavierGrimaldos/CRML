import aiohttp
import asyncio
import json
from datetime import datetime
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = //COMPLETAR CON LA DE USURIO

OUTPUT_PATH = //AÑADIR PATH PARA GUARDAR LOS ARCHIVOS

class Ensamblador:
    def __init__(self, modelos=None):
        self.modelos= modelos or []
    
    async def query_modelo(self, sesion, modelo, prompt:str)-> dict:
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        payload={
            "model":modelo["name"],
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.4
        }
        try:
            async with sesion.post(API_URL, headers=headers, json=payload) as rep:
                data=await rep.json()
                if "choices" in data:
                    texto_respuesta=data["choices"][0]["message"]["content"]
                else:
                    texto_respuesta=f"Error {data}"
        except Exception as e:
            texto_respuesta=f"Request failed: {e}"

        return{
            "model_name":modelo["name"],
            "response":texto_respuesta,
            "timestamp":datetime.now().isoformat()
        }
    
    async def run(self, prompt:str):
        async with aiohttp.ClientSession() as sesion:
            tasks=[self.query_modelo(sesion, modelo, prompt) for modelo in self.modelos]
            resultados= await asyncio.gather(*tasks)
        resultados_validos = []
        for r in resultados:
            respuesta_texto = r["response"]
            if (not str(respuesta_texto).startswith("Error") and not str(respuesta_texto).startswith("Request failed:")):
                resultados_validos.append(r)
            else:
                print(f"Filtrado error de modelo {r['model_name']}: {respuesta_texto[:50]}...")

        print(f"Respuestas válidas: {len(resultados_validos)}/{len(self.modelos)}")

        return resultados_validos
    
    def guardar_resultados(self, respuestas, output_dir=OUTPUT_PATH):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        filename=f"ensamble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"
        filepath=Path(output_dir)/filename
        with open(filepath,"w",encoding="UTF-8") as file:

            json.dump(respuestas, file, indent=2, ensure_ascii=False)
