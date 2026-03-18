# 🎤 GUIÓN DE PRESENTACIÓN — GLUCOVIBES
## Predicción del Impacto Glucémico Postprandial
**Duración total: 20 minutos | Dos presentadores: A y B**

---

> **Distribución de tiempo:**
> - Portada + intro: 1 min
> - El problema: 2 min
> - Datos: 1.5 min
> - Metodología + features: 3.5 min
> - Resultados peak_value: 3 min
> - Resultados TTP: 2.5 min
> - Resumen: 1.5 min
> - Trabajo futuro: 2.5 min
> - Demo + cierre: 2.5 min
> **Total: 20 min**

---

## 🖼️ SLIDE 1 — PORTADA
**⏱️ ~1 min | Presentador: A**

*[Con la slide de portada visible]*

**A:** Buenos días a todos. Somos el equipo de Glucovibes, y hoy os venimos a contar cómo conseguimos predecir el pico glucémico de una persona *antes de que coma*.

No es magia ni es ciencia ficción. Son 721.000 lecturas de glucosa continua, 15.000 comidas registradas y un pipeline de machine learning que construimos desde cero.

En los próximos 20 minutos os vamos a contar la historia completa: el problema que resolvemos, cómo lo atacamos, qué conseguimos y, sobre todo, qué significa esto para un usuario real de la app.

*[Señalar los tres pills de datos en la parte baja]*

Y estos son los números que lo sustentan: 15.066 comidas, 92 usuarios con sensor de glucosa continua, y más de 720.000 lecturas. Un dataset real, con toda la complejidad que eso implica.

---

## 🖼️ SLIDE 2 — EL PROBLEMA
**⏱️ ~2 min | Presentador: A**

**A:** Antes de hablar de modelos y métricas, necesitamos entender por qué esto importa.

*[Señalar el panel izquierdo]*

Cuando comes, tu glucosa en sangre sube. Sube hasta un pico y luego baja. Este ciclo ocurre varias veces al día, y la forma de esa curva — cuánto sube y cuándo llega al máximo — tiene consecuencias directas sobre tu salud.

*[Señalar las tres tarjetas de la derecha]*

Primero, riesgo cardiovascular. Los picos altos y frecuentes dañan el endotelio vascular. Es un factor de riesgo independiente para diabetes tipo 2, incluso en personas sin diagnóstico previo.

Segundo, rendimiento cognitivo. El "crash" de glucosa después del pico — esa caída brusca — es lo que provoca la somnolencia y la niebla mental de después de comer. Si predices el pico, puedes evitar el crash.

Y tercero, recuperación deportiva. El timing del pico respecto al ejercicio condiciona si tu cuerpo oxida grasa o glucosa, y la síntesis muscular posterior.

El valor diferencial de Glucovibes es exactamente este: pasar de reaccionar a lo que ya pasó, a anticipar lo que va a pasar.

---

## 🖼️ SLIDE 3 — LOS DATOS
**⏱️ ~1.5 min | Presentador: B**

**B:** Gracias. Antes de entrar en el modelo, os cuento con qué trabajamos.

*[Señalar las tarjetas de estadísticas]*

El dataset del reto combina seis fuentes de datos distintas: las lecturas CGM continuas del sensor de glucosa, los registros de comidas con sus alimentos individuales, la composición nutricional de cada alimento, los registros de ejercicio, y los cuestionarios de mañana y noche con datos de sueño, frecuencia cardíaca, HRV y estado de ánimo.

*[Señalar la tabla de fuentes]*

Lo más valioso — y lo más complejo — es el CGM: readings cada 5 minutos durante más de dos años, para 92 usuarios. Eso nos da la ventana de glucosa preprandial que se convierte en uno de los predictores más potentes del modelo.

El rango temporal va de julio de 2021 a enero de 2023. Son datos reales, con todo lo que eso supone: huecos, sensores fallidos, comidas mal registradas. El 60% del trabajo del proyecto fue la limpieza y validación de estas fuentes.

---

## 🖼️ SLIDE 4 — METODOLOGÍA / PIPELINE
**⏱️ ~1.5 min | Presentador: B**

**B:** El pipeline que construimos tiene cinco fases.

*[Señalar las cinco tarjetas del pipeline]*

La fase 2 es limpieza científica: eliminamos comidas con ventana CGM insuficiente, sincronizamos timestamps entre tablas y validamos los targets glucémicos.

La fase 3 construye las 57 features base: macronutrientes, actividad física, sueño, datos de cuestionario.

La fase 3b es donde añadimos 28 features especializadas para predecir el pico. Las más importantes son las features CGM preprandiales — la pendiente de glucosa en los 90 minutos antes de comer — y las rolling stats del usuario sin leakage.

*[Señalar el pill "features rolling SIN leakage"]*

Esta última decisión es crítica: en el pipeline original, las estadísticas del usuario se calculaban con todos sus datos, incluyendo el futuro. Eso es data leakage. Nosotros las reconstruimos como medias rodantes de los 14 días anteriores a cada comida.

Las fases 5 y 5b son el modelado y la versión de producción, que os explicamos ahora.

*[Señalar los tres pills de configuración del pipeline]*

Todo con GroupKFold de 5 folds para garantizar que el mismo usuario no aparece en entrenamiento y test, y 80 trials de Optuna para la búsqueda de hiperparámetros.

---

## 🖼️ SLIDE 5 — FEATURE ENGINEERING
**⏱️ ~2 min | Presentador: A**

**A:** El corazón del proyecto son estas 64 variables, organizadas en seis grupos.

*[Señalar las seis tarjetas]*

Las 17 features nutricionales incluyen macros, fibra, índice glucémico, proporción de ultraprocesados y grupos de alimentos de cada comida.

Las 8 features CGM preprandiales son las que más nos diferenciaron del baseline. Calculamos la pendiente de glucosa en los últimos 30, 60 y 90 minutos antes de comer, su variabilidad y el delta total. Si la glucosa ya está subiendo antes de la comida, el pico va a ser sistemáticamente más alto. Esto es fisiología, y el modelo lo aprende.

Las features de actividad tienen ventanas de 2, 6, 24 y 48 horas, con una puntuación de intensidad ponderada. El ejercicio de las últimas 2 horas tiene un efecto muy diferente al de ayer.

Y las 5 features de perfil rolling del usuario son las que más impacto tienen en el modelo, junto con el pico previo del mismo día. Cada persona tiene una respuesta glucémica característica, y estas features la capturan de forma prospectiva, sin leakage.

*[Señalar el subtítulo en coral]*

Esta decisión de no usar estadísticas históricas con datos futuros es lo que hace que el modelo sea desplegable en producción. No es un detalle técnico menor: es la diferencia entre un modelo que funciona en investigación y uno que funciona en una app real.

---

## 🖼️ SLIDE 6 — RESULTADOS PEAK VALUE
**⏱️ ~3 min | Presentador: B**

**B:** Entramos en resultados. Empezamos con el target principal: predecir el valor máximo de glucosa en mg/dL.

*[Señalar la tabla de comparativa]*

Partimos de un baseline de 13.79 mg/dL de MAE con el pipeline original. Nuestro modelo final, combinando Optuna, sample weights y calibración isotónica, llega a 13.32 mg/dL con un R² de 0.54.

*[Señalar los cuatro bloques de distribución del error a la derecha]*

Pero la métrica más relevante para el producto no es el MAE global, sino la distribución del error. El 47.9% de nuestras predicciones tienen un error menor a 10 mg/dL. El 79.8% tienen un error menor a 20 mg/dL. Y el 92% tienen un error menor a 30 mg/dL.

Para un usuario que tiene un rango objetivo de, digamos, 70 a 140 mg/dL, un error de 10 mg/dL es clínicamente aceptable para generar recomendaciones de ajuste de la dieta.

*[Señalar las barras de feature importance]*

Los predictores más importantes son los que más nos sorprendieron: el pico previo del mismo día —si ya tuviste un pico alto esta mañana, el siguiente será más alto— y el perfil rolling del usuario. La glucosa preprandial está en segundo lugar.

Destacar que cuatro de los cinco top features son features nuevas que construimos en la fase 3b. Eso valida la hipótesis de que la ventana CGM preprandial y el perfil personal del usuario son las dos fuentes de información más potentes que teníamos sin explotar.

---

## 🖼️ SLIDE 7 — CALIBRACIÓN ISOTÓNICA
**⏱️ ~1.5 min | Presentador: B**

**B:** El modelo base tenía un sesgo sistemático que detectamos analizando las predicciones por rango de valor real.

*[Señalar el gráfico de barras]*

Para picos bajos, menores de 100 mg/dL, el modelo sobreestimaba en 18 mg/dL de media. Para picos altos, mayores de 200 mg/dL, subestimaba en 44 mg/dL. Es el fenómeno clásico de regresión a la media en modelos de machine learning.

*[Señalar el panel derecho de calibración]*

La solución es elegante: entrenamos un regresor isotónico sobre las predicciones out-of-fold versus los valores reales. Esta función monótona aprende a corregir el sesgo sin tocar el modelo principal. En producción, cada predicción pasa primero por el modelo base y luego por el calibrador.

El impacto directo en MAE es de 0.15 mg/dL, pero el impacto en la fiabilidad de las predicciones extremas es mucho mayor: el sesgo en picos mayores de 200 se reduce de -44 a -40 mg/dL.

---

## 🖼️ SLIDE 8 — RESULTADOS TTP
**⏱️ ~2.5 min | Presentador: A**

**A:** El segundo target es el tiempo hasta el pico. Y aquí la historia es diferente, y más interesante.

*[Señalar el panel de "Problema inicial"]*

Empezamos con regresión directa. R² de 0.06. Prácticamente inútil: el modelo apenas mejora sobre predecir siempre la media.

Pasamos a clasificación de 3 clases: rápido, moderado y tardío. F1-macro de 0.41. Mejor, pero aún insuficiente. El análisis de la matriz de confusión reveló el problema: la clase "moderado", entre 45 y 75 minutos, tenía un recall de solo el 16%. Actúa como zona de ambigüedad que contamina el aprendizaje de las otras dos clases.

*[Señalar el panel de "Solución"]*

La solución fue reformular el problema. Descartamos la clase moderada del entrenamiento y entrenamos solo con los casos claros: rápido, menos de 45 minutos, y tardío, más de 75 minutos. En producción, las comidas que el modelo clasifica con baja confianza se reportan directamente como "incierto".

*[Señalar la matriz de confusión]*

El resultado: F1 de 0.616, un aumento del 49% sobre la clasificación de 3 clases. Precision del 60.5%, recall del 62.9%.

*[Señalar las métricas de la derecha]*

¿Qué significa esto en la práctica? Cuando el modelo predice que el pico va a ser tardío, acierta el 60% de las veces. Y de todos los picos que realmente llegan tarde, detectamos el 63%.

Para una app que quiere recomendar "espera 15 minutos antes de hacer ejercicio después de comer", este nivel de precisión es suficiente para generar valor real sin generar alertas falsas excesivas.

---

## 🖼️ SLIDE 9 — RESUMEN DE MEJORAS
**⏱️ ~1.5 min | Presentador: A**

**A:** Aquí está la visión global de lo que conseguimos.

*[Señalar panel izquierdo peak_value]*

Para el pico glucémico: de un MAE de 13.79 al inicio a 13.32 en producción. El R² pasó de 0.48 a 0.54, una mejora del 12.7%. Esto con LightGBM optimizado con Optuna, sample weights que enfatizan los usuarios de perfil extremo, y calibración isotónica.

*[Señalar panel derecho TTP]*

Para el tiempo al pico: de un modelo que no funcionaba — regresión con R² de 0.03 — a una clasificación binaria con F1 de 0.616. Una mejora del 49% sobre el mejor intento anterior.

Dos modelos listos para producción, serializados y documentados.

---

## 🖼️ SLIDE 10 — TRABAJO FUTURO
**⏱️ ~2.5 min | Presentador: B**

**B:** Somos honestos sobre el techo del modelo actual. El R² de 0.54 refleja que aproximadamente el 46% de la varianza en la respuesta glucémica viene de factores que no observamos: genética, microbioma, variabilidad biológica intrínseca.

Pero hay cinco líneas de trabajo concretas que pueden mover la aguja.

*[Señalar la primera tarjeta en rojo]*

La más urgente: modelos personalizados para usuarios de alto error. Los usuarios 401 y 919 tienen MAE de 35 y 31 mg/dL. Son perfiles glucémicos extremos con picos medios de más de 200 mg/dL. Con un esquema de dos niveles — modelo global más un ajuste diferencial por usuario — estimamos reducir su error a menos de 15 mg/dL. Tienen 80-120 comidas cada uno, suficiente para entrenar con regularización fuerte.

*[Señalar la segunda tarjeta]*

Para el tiempo al pico, el predictor que falta es la composición de carbohidratos. El índice glucémico binario que usamos actualmente no distingue entre carbohidratos simples y complejos dentro del mismo alimento. Los simples producen picos en 20-30 minutos; los complejos en 45-90. Extraer esta información de food_composition podría ser la feature discriminante que necesita el modelo de TTP.

*[Señalar la tarjeta de LSTM]*

Más ambicioso: usar la secuencia completa de los últimos 18 puntos CGM —los 90 minutos previos— como entrada a un LSTM ligero. Los escalares que calculamos ahora pierden información sobre patrones temporales, como oscilaciones en dientes de sierra antes de comer.

*[Señalar microbioma y iAUC]*

Y a más largo plazo: datos de microbioma en el onboarding — que explican un 15-20% de la varianza glucémica según la literatura — y reentrenar el modelo de iAUC con las 64 features actuales, que debería reducir su MAE de 1.095 a menos de 950.

---

## 🖼️ SLIDE 11 — DEMO
**⏱️ ~2 min | Presentador: A**

**A:** Os mostramos cómo funciona en producción.

*[Señalar el bloque de código]*

El flujo de inferencia es simple: se cargan los tres modelos una sola vez al iniciar el servicio. Para cada nueva comida, el modelo base produce una predicción de pico, que pasa por el calibrador isotónico. El modelo de TTP produce una probabilidad de pico tardío, y con un threshold de 0.40 —que optimiza el F1— clasifica entre rápido, tardío o incierto.

*[Señalar las tarjetas de salida a la derecha]*

El output para el usuario es directo: pico esperado de 147 mg/dL, tiempo tardío con confianza del 72%, y una recomendación accionable. En este caso, añadir fibra para reducir la carga glucémica efectiva.

*[Señalar los paquetes de archivos en la parte baja]*

Los cinco artefactos generados están documentados y listos: el modelo de pico, el calibrador, los feature sets y el clasificador de TTP.

---

## 🖼️ SLIDE 12 — CIERRE
**⏱️ ~0.5 min | Presentadores: A y B**

**A:** Para cerrar.

**B:** Construimos un pipeline completo desde datos crudos hasta modelos de producción. MAE de 13.32 mg/dL en el pico glucémico, F1 de 0.62 en el tiempo al pico, 64 features sin leakage.

**A:** Identificamos cinco líneas de trabajo futuro con impacto cuantificado. El siguiente paso más impactante son los modelos personalizados por usuario, que pueden atacar directamente los casos de mayor error sin afectar al rendimiento global.

**B:** Muchas gracias. Quedamos a vuestra disposición para preguntas.

---

## 📋 PREGUNTAS FRECUENTES Y RESPUESTAS PREPARADAS

**P: ¿Por qué no usáis redes neuronales desde el principio?**
> R: Con 15.000 muestras y alta varianza por usuario, los modelos de gradiente boosting generalizan mejor que redes profundas. El R² de 0.54 con LightGBM probablemente no mejora con una red densa con los mismos datos tabulares. El LSTM es interesante pero solo para la componente CGM, no para todo el pipeline.

**P: ¿El modelo funciona en tiempo real?**
> R: Sí. La inferencia con joblib tarda menos de 50ms por predicción. El cuello de botella es el cálculo de las features rolling, que requiere acceso al historial del usuario. Con una base de datos en memoria para los últimos 30 días, es completamente viable en tiempo real.

**P: ¿No es un MAE de 13 mg/dL demasiado alto para ser útil clínicamente?**
> R: Depende del uso. Para diagnóstico clínico, sí. Para recomendaciones de ajuste de dieta en una app de bienestar, el 66% de predicciones con error menor a 15 mg/dL es suficiente para generar valor. La clave es comunicar la incertidumbre al usuario, no presentar el número como exacto.

**P: ¿Cómo manejáis la privacidad de los datos de glucosa?**
> R: El pipeline trabaja con IDs anonimizados desde la fase de limpieza. Los modelos personalizados por usuario requieren un modelo por ID, nunca se comparten datos entre usuarios. En producción, el pipeline de features se ejecutaría en el dispositivo del usuario o en un servidor con cifrado de extremo a extremo.

**P: ¿Qué pasa con usuarios nuevos sin historial?**
> R: Las features rolling tienen fallback a las medias globales del dataset para los primeros 14 días. Después de 5-10 comidas registradas, el modelo ya empieza a tener un perfil personal del usuario con información más útil que el global.

---

*Guión elaborado para Glucovibes Challenge 2026 · Máster en Data Science*
