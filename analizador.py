import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from textblob import TextBlob
import json
import requests

API_KEY="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"
MODELO="gpt-3.5-turbo"

def obtener_matriz_consenso_completa(respuestas):
    if len(respuestas)<2:
        return np.array([[1.0]]), 1.0, [0]
    
    try:
        respuestas_limpias=[]
        indices_validos=[]

        for i, respuesta in enumerate(respuestas):
            if respuesta and len(respuesta.strip())>10:
                respuestas_limpias.append(respuesta.strip())
                indices_validos.append(i)

        if len(respuestas_limpias)<2:
            print("advertancia: Muy pocas respuestas válidas para calcular consenso")
            n=len(respuestas)
            return np.identity(n), 0.5, list(range(n))
        
        vectorizer=TfidfVectorizer(
            min_df=1,
            max_features=1000,
            strip_accents='unicode'
        )
        tfidf_matrix=vectorizer.fit_transform(respuestas_limpias)
        if tfidf_matrix.shape[1]==0:
            print("Advertencia: Matriz vacía, usando identidad")
            n=len(respuestas_limpias)
            matriz_similitud=np.identity(n)
        else:
            matriz_similitud=cosine_similarity(tfidf_matrix)
        matriz_completa=np.zeros((len(respuestas), len(respuestas)))
        for i, idx_i in enumerate(indices_validos):
            for j, idx_j in enumerate(indices_validos):
                matriz_completa[idx_i,idx_j]=matriz_similitud[i,j]
        
        for i in range(len(respuestas)):
            if i not in indices_validos:
                matriz_completa[i,i]=1.0
        
        indices_a_mantener=[]
        for i in range(len(matriz_completa)):
            tiene_similitudes_validas=any(matriz_completa[i][j]>0 for j in range(len(matriz_completa))if j!=i)
            if tiene_similitudes_validas or i in indices_validos:
                indices_a_mantener.append(i)
        
        if not indices_a_mantener:
            print("Advertencia: Ningún modelo tiene similitudes válidas")
            n=len(respuestas)
            return np.identity(n), 0.5, list(range(n))
        
        matriz_filtrada=matriz_completa[np.ix_(indices_a_mantener, indices_a_mantener)]
        indices_originales_filtrados=indices_a_mantener
        indices_superiores=np.triu_indices_from(matriz_filtrada, k=1)
        similitudes_filtradas=[]
        for i,j in zip(indices_superiores[0], indices_superiores[1]):
            idx_i_orig=indices_originales_filtrados[i]
            idx_j_orig=indices_originales_filtrados[j]
            if idx_i_orig in indices_validos and idx_j_orig in indices_validos:
                similitudes_filtradas.append(matriz_filtrada[i,j])
        if similitudes_filtradas:
            consenso_general=np.mean(similitudes_filtradas)
        else:
            consenso_general=0.5
        
        return matriz_filtrada, consenso_general, indices_originales_filtrados
            
    except Exception as e:
        print(f"Error calculando matriz de consenso: {e}")
        n=len(respuestas)
        return np.identity(n), 0.5, list(range(n))
    
def calcular_consensos_individuales(matriz_similitud, nombres_modelos, indices_originales):
    n=len(matriz_similitud)
    consensos_individuales=[]

    for i in range(n):
        idx_original=indices_originales[i]
        similitudes_con_otros=[matriz_similitud[i][j] for j in range(n) if j!=i]
        consenso_individual=np.mean(similitudes_con_otros)if similitudes_con_otros else 0.0
        consensos_individuales.append({
            "respuesta_idx":idx_original,
            "indice_filtrado":i,
            "modelo":nombres_modelos[idx_original],
            "consenso_individual": consenso_individual,
        })
    return consensos_individuales

def encontrar_mayores_consensos_individuales(consensos_individuales):
    consensos_validos=[c for c in consensos_individuales if c["consenso_individual"]>0]
    if not consensos_validos: return[]

    consensos_ordenados=sorted(consensos_validos, key=lambda x: x["consenso_individual"], reverse=True)
    top_k=max(1,len(consensos_ordenados)*2//3)
    mayores_consensos=consensos_ordenados[:top_k]

    return mayores_consensos

def calcular_consenso_semantico(respuestas, nombres_modelos):
    matriz_consenso, consenso_global, indices_filtrados=obtener_matriz_consenso_completa(respuestas)
    consensos_individuales=calcular_consensos_individuales(matriz_consenso,nombres_modelos,indices_filtrados)
    mayores_consensos=encontrar_mayores_consensos_individuales(consensos_individuales)
    consensos_validos=[c for c in consensos_individuales if c["consenso_individual"]>0]
    respuesta_mas_consensuada=max(consensos_validos, key=lambda x: x["consenso_individual"]) if consensos_validos else None

    return{
        'matriz_consenso': matriz_consenso,
        'consenso_global': consenso_global,
        'consensos_individuales': consensos_individuales,
        'mayores_consensos': mayores_consensos,
        'respuesta_mas_consensuada': respuesta_mas_consensuada,
        'indices_filtrados': indices_filtrados,
        'nombres_filtrados': [nombres_modelos[i] for i in indices_filtrados]
    }

def analizar_consenso_con_determinante(matriz_consenso):
    try:
        indices_validos=[]
        for i in range(len(matriz_consenso)):
            tiene_similitudes=any(matriz_consenso[i][j]>0 for j in range(len(matriz_consenso)) if j!=i)
            if tiene_similitudes:
                indices_validos.append(i)
        
        if len(indices_validos)<2:
            return None
        
        submatriz=matriz_consenso[np.ix_(indices_validos, indices_validos)]
        det=np.linalg.det(submatriz)

        if det>0.8: interpretacion="Alta diferencia entre respuestas"
        if det>0.5: interpretacion="Alguna diferencia entre respuestas"
        if det>0.25: interpretacion="Alta similitud"
        else: interpretacion="Baja similitud"

        return {
            'determinante': 1-det,
            'interpretacion': interpretacion,
            'rango_matriz': np.linalg.matrix_rank(submatriz),
            'autovalores': np.linalg.eigvals(submatriz),
            'indices_validos': indices_validos
        }
    except Exception as e:
        print(f"Error en análisis con determinante: {e}")
        return None
    
def imprimir_matriz_consenso(matriz_consenso, nombres_modelos_filtrados):
    if not nombres_modelos_filtrados:
        print("No hay modelos válidos para mostrar en la matriz")
        return
    
    nombres_cortos=[]
    for nombre in nombres_modelos_filtrados:
        if len(nombre)>20:
            partes=nombre.split("/")
            if len(partes)>1: nombres_cortos.append(partes[-1][:20])
            else: nombres_cortos.append(nombre[:20])
        else: nombres_cortos.append(nombre)

    print("\t\tMATRIZ DE CONSENSO - SIMILITUDES COSENO ENTRE MODELOS VÁLIDOS")
    print(f"Modelos mostrados: {len(nombres_modelos_filtrados)}")

    header=" "*25
    for nombre in nombres_cortos:
        header+=f"{nombre:>12}"
    print(header)

    for i in range(len(matriz_consenso)):
        fila=f"{nombres_cortos[i]:<25}"
        for j in range(len(matriz_consenso[i])):
            valor=matriz_consenso[i][j]
            if i==j: fila += f"{"1.000":>12}"
            else: fila += f"{valor:>12.3f}"
        print(fila)

    es_simetrica=np.allclose(matriz_consenso, matriz_consenso.T)
    print(f"\nLa matriz es simétrica: {es_simetrica}")
    total_elementos=len(matriz_consenso)*len(matriz_consenso)
    elementos_cero=np.sum(matriz_consenso==0)
    elementos_diagonal=len(matriz_consenso)
    elementos_cero_fuera_diagonal=elementos_cero-elementos_diagonal
    print(f"Elementos cero (fuera de la diagonal): {elementos_cero_fuera_diagonal}/{total_elementos-elementos_diagonal}")

def llamar_chatgpt(mensajes):
    try:
        url="https://api.openai.com/v1/chat/completions"
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload={
            "model": MODELO,
            "messages": mensajes,
            "max_tokens": 1500,
            "temperature":0.2
        }

        print("Llamando a ChatGPT...")
        response=requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code==200: return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error API: {response.status_code}-{response.text}")
            return None
        
    except Exception as e:
        print(f"Error llamando a ChatGPT: {e}")
        return None

def generar_fusion(respuestas_top3):
    try:
        contexto_fusion = f"""
        Combina estas 3 respuestas en una única respuesta coherente y de alta calidad:

        RESPUESTA 1 ({respuestas_top3[0]['model_name']}):
        {respuestas_top3[0]['response']}

        RESPUESTA 2 ({respuestas_top3[1]['model_name']}):
        {respuestas_top3[1]['response']}

        RESPUESTA 3 ({respuestas_top3[2]['model_name']}):
        {respuestas_top3[2]['response']}

        Instrucciones:
        - Combina lo mejor de cada respuesta
        - Elimina redundancias
        - Mantén coherencia y flujo lógico
        - Conserva los puntos más importantes

        Responde ÚNICAMENTE con la respuesta fusionada:
        """
        
        mensajes=[
            {
                "role":"system",
                "content": "Eres un experto en fusionar respuestas de IA de forma coherente"
            },
            {
                "role":"user",
                "content":contexto_fusion
            }
        ]

        respuesta_fusionada=llamar_chatgpt(mensajes)
        if respuesta_fusionada: return respuesta_fusionada.strip()
        else: return f"Fusión falló. Respuesta más consensuada:\n\n{respuestas_top3[0]["response"]}"
        
    except Exception as e:
        print(f"Error en fusión: {e}")
        return respuestas_top3[0]["response"]

def dataAnalisis(resultados_ensamblador):
    print("\t\tANÁLISIS DE CONSENSO CON CHATGPT")

    if not resultados_ensamblador or len(resultados_ensamblador) <3:
        print("Se necesitan al menos 3 respuestas para el análisis")
        return {"Error":"Se necesitan al menos 3 respuestas para el análisis"}
    try:
        respuestas_texto=[r["response"] for r in resultados_ensamblador]
        nombres_modelos=[r["model_name"] for r in resultados_ensamblador]
        resultado_consenso=calcular_consenso_semantico(respuestas_texto,nombres_modelos)
        imprimir_matriz_consenso(resultado_consenso["matriz_consenso"], resultado_consenso["nombres_filtrados"])
        reporte = {
            'consenso_global': round(resultado_consenso['consenso_global'], 3),
            'respuesta_mas_consensuada': {
                'modelo': resultado_consenso['respuesta_mas_consensuada']['modelo'],
                'consenso_individual': round(resultado_consenso['respuesta_mas_consensuada']['consenso_individual'], 3)
            } if resultado_consenso['respuesta_mas_consensuada'] else None
        }
        print(f"\nCONSENSO GLOBAL: {resultado_consenso["consenso_global"]:.3f}")
        if resultado_consenso["respuesta_mas_consensuada"]:
            mejor=resultado_consenso["respuesta_mas_consensuada"]
            print(f"RESPUESTA MÁS CONSENSUADA: {mejor["modelo"]} (consenso: {mejor["consenso_individual"]:.3f})")

        if len(resultado_consenso["mayores_consensos"])>=3:
            print(f"\nTOP 3 CONSENSOS:")
            for i, consenso in enumerate(resultado_consenso["mayores_consensos"][:3]):
                print(f" {i+1}. {consenso["modelo"]}: {consenso["consenso_individual"]:.3f}")
            print(f"\nGENERANDO FUSIÓN CON CHATGPT...")

            top3_respuestas=[]
            for consenso in resultado_consenso["mayores_consensos"][:3]:
                idx=consenso["respuesta_idx"]
                respuesta_original=resultados_ensamblador[idx]
                top3_respuestas.append({
                    'model_name': respuesta_original['model_name'],
                    'response': respuesta_original['response'],
                    'consenso_individual': consenso['consenso_individual']
                })
            respuesta_fusionada=generar_fusion(top3_respuestas)
            print(f"\t\tRESPUESTA FUSIONADA:")
            print(respuesta_fusionada)

            reporte["respuesta_fusionada"]=respuesta_fusionada
            reporte["modelos_base"]=[r["model_name"]for r in top3_respuestas]

        return reporte
    except Exception as e:
        print(f"Error en el análisis: {e}")
        return {"Error":f"Error: {str(e)}"}