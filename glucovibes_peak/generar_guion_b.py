from fpdf import FPDF

OUTPUT = r"C:\Users\enriq\Desktop\Master\Reto Empresarial\glucovibes_peak\guion_B_v3.pdf"
ARIAL     = r"C:\Windows\Fonts\arial.ttf"
ARIAL_B   = r"C:\Windows\Fonts\arialbd.ttf"
ARIAL_I   = r"C:\Windows\Fonts\ariali.ttf"

SECTIONS = [
    {
        "title": "SLIDE 3 — Los datos",
        "content": (
            "Gracias. Os cuento con qué datos trabajamos.\n\n"
            "Combinamos seis fuentes: lecturas de glucosa continua del sensor CGM, registros de comidas, "
            "composición nutricional de cada alimento, ejercicio, y cuestionarios diarios con sueño, "
            "frecuencia cardíaca y HRV.\n\n"
            "El volumen es relevante: 92 usuarios, más de dos años de datos, una lectura cada 5 minutos. "
            "Eso nos da la ventana de glucosa justo antes de cada comida, que es uno de los predictores "
            "más potentes del modelo.\n\n"
            "Y es importante decirlo: son datos reales, con todos los problemas que eso conlleva. "
            "Limpiar y validar las seis fuentes fue una parte enorme del trabajo."
        ),
    },
    {
        "title": "SLIDE 4 — Pipeline",
        "content": (
            "El pipeline que construimos tiene cinco fases.\n\n"
            "Primero limpieza: eliminamos comidas sin ventana CGM suficiente y sincronizamos timestamps entre tablas.\n\n"
            "Luego construimos las features. En total 85 variables: macros, actividad física con distintas "
            "ventanas temporales, sueño, cuestionario, y las features CGM preprandiales — "
            "la pendiente de glucosa en los 30, 60 y 90 minutos antes de comer.\n\n"
            "Una decisión clave: las estadísticas de cada usuario se calculan solo con sus datos anteriores "
            "a esa comida. Si usas datos futuros para calcularlas, el modelo parece funcionar bien "
            "en entrenamiento pero falla en producción. Nosotros lo hicimos bien desde el principio.\n\n"
            "El entrenamiento usa GroupKFold de 5 folds — mismo usuario nunca en train y test a la vez — "
            "y 80 iteraciones de Optuna para hiperparámetros."
        ),
    },
    {
        "title": "SLIDE 6 — Resultados: valor del pico",
        "content": (
            "El primer modelo predice el valor máximo de glucosa en mg/dL. Resultado: MAE de 13.32 y R² de 0.54.\n\n"
            "Pero más que el error medio, lo que importa es cuántas predicciones son útiles en la práctica: "
            "el 48% tienen error menor de 10 mg/dL, el 80% menor de 20, y el 92% menor de 30. "
            "Para generar recomendaciones de dieta, eso es suficiente.\n\n"
            "Lo más interesante fue la feature importance. El predictor número uno es el pico previo del mismo día: "
            "si ya tuviste un pico alto esta mañana, el siguiente va a ser más alto. "
            "El segundo es el perfil personal del usuario. "
            "La glucosa preprandial entra en tercer lugar.\n\n"
            "Eso confirma que la respuesta glucémica es muy individual, y que el modelo la captura."
        ),
    },
    {
        "title": "SLIDE 7 — Calibración isotónica",
        "content": (
            "Detectamos que el modelo tenía un sesgo en los extremos: sobreestimaba picos bajos "
            "y subestimaba picos altos. Es algo habitual en regresión — el modelo tiende hacia la media.\n\n"
            "La solución: un regresor isotónico entrenado sobre las predicciones fuera de muestra. "
            "Aprende a corregir ese sesgo de forma monótona, sin tocar el modelo principal. "
            "En producción, cada predicción pasa primero por el modelo y luego por el calibrador.\n\n"
            "El impacto en MAE global es pequeño, pero en los casos extremos — "
            "que son precisamente los más relevantes clínicamente — la corrección es significativa."
        ),
    },
    {
        "title": "SLIDE 10 — Trabajo futuro",
        "content": (
            "El R² de 0.54 es honesto: casi la mitad de la varianza glucémica viene de cosas que no observamos "
            "— genética, microbioma, variabilidad biológica. Eso tiene techo sin datos adicionales.\n\n"
            "Hay tres líneas que más nos interesan.\n\n"
            "La primera son modelos personalizados por usuario. Hay perfiles con error muy alto porque su respuesta "
            "glucémica es atípica. Con un ajuste individual sobre el modelo global podríamos reducir ese error "
            "a la mitad para esos usuarios.\n\n"
            "La segunda es mejorar el modelo de tiempo al pico incorporando la composición de carbohidratos: "
            "simples versus complejos. Los simples generan pico en 20-30 minutos, los complejos en 45-90. "
            "Es la variable que más nos falta.\n\n"
            "Y la más ambiciosa: sustituir los escalares CGM por la secuencia completa de los 90 minutos previos "
            "como entrada a un LSTM. Los escalares pierden patrones temporales que pueden ser muy informativos."
        ),
    },
    {
        "title": "SLIDE 12 — Cierre",
        "content": (
            "Pipeline completo desde datos crudos hasta modelos en producción. "
            "MAE de 13.32 en el pico glucémico, F1 de 0.62 en el tiempo al pico, 85 features sin leakage.\n\n"
            "[esperar a que A hable]\n\n"
            "Muchas gracias. Quedamos para preguntas."
        ),
    },
]

BRAND = (0, 160, 120)
DARK  = (240, 240, 240)
LIGHT = (30, 30, 30)
CARD  = (225, 245, 240)


class PDF(FPDF):
    def header(self):
        self.set_fill_color(*BRAND)
        self.rect(0, 0, 210, 20, "F")
        self.set_xy(0, 4)
        self.set_font("Arial-B", size=14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "GLUCOVIBES  —  Guión personal  (Speaker B)", align="C")
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", size=8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")


pdf = PDF()
pdf.add_font("Arial",   fname=ARIAL,   uni=True)
pdf.add_font("Arial-B", fname=ARIAL_B, uni=True)
pdf.add_font("Arial-I", fname=ARIAL_I, uni=True)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# Subtitle
pdf.set_font("Arial-I", size=10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 6, "Slides 3, 4, 6, 7, 10 y cierre  |  ~10 minutos", ln=True)
pdf.ln(4)

for sec in SECTIONS:
    # Title bar with left accent
    y = pdf.get_y()
    pdf.set_fill_color(*BRAND)
    pdf.rect(10, y, 3, 9, "F")
    pdf.set_fill_color(*CARD)
    pdf.rect(13, y, 187, 9, "F")
    pdf.set_xy(17, y + 1)
    pdf.set_font("Arial-B", size=11)
    pdf.set_text_color(0, 90, 70)   # dark green, readable on light bg
    pdf.cell(183, 7, sec["title"], ln=True)
    pdf.ln(2)

    # Body
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(20, 20, 20)  # near-black
    pdf.set_left_margin(14)
    pdf.set_right_margin(14)
    pdf.multi_cell(0, 5.8, sec["content"])
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.ln(6)

pdf.output(OUTPUT)
print(f"PDF generado: {OUTPUT}")
