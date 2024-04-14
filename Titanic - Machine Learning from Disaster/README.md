# Titanic - Machine Learning from Disaster

Este es un proyecto para el desafío del Titanic en Kaggle, donde el objetivo es predecir si un pasajero sobrevivió o no al naufragio del Titanic utilizando datos de pasajeros como edad, género, clase socioeconómica, etc.

## Descripción del proyecto

El objetivo de este proyecto es aplicar técnicas de aprendizaje automático para predecir la supervivencia de los pasajeros del Titanic. Se utilizan datos históricos de pasajeros que incluyen características como edad, género, clase socioeconómica, etc. El proyecto se estructura en los siguientes pasos:

1. **Exploración de datos**: Análisis exploratorio de los datos para comprender las características de los pasajeros y su relación con la supervivencia.

2. **Preprocesamiento de datos**: Limpieza y preparación de los datos para su posterior análisis y modelado. Esto incluye la imputación de valores faltantes, la codificación de variables categóricas, etc.

3. **Modelado de datos**: Entrenamiento y evaluación de modelos de aprendizaje automático para predecir la supervivencia de los pasajeros.

4. **Validación del modelo**: Validación del rendimiento del modelo utilizando métricas de evaluación y técnicas de validación cruzada.

## Estructura del repositorio

El repositorio está estructurado de la siguiente manera:

- `data/`: Carpeta que contiene los datos del desafío Titanic (train.csv, test.csv).
- `notebooks/`: Cuadernos Jupyter utilizados para la exploración de datos, el preprocesamiento, el modelado y la validación del modelo.
- `models.py`: Archivo Python que contiene las definiciones de los modelos de aprendizaje automático utilizados en el proyecto.
- `train_and_evaluate.py`: Archivo Python que contiene funciones para entrenar y evaluar los modelos.
- `requirements.txt`: Archivo que enumera todas las dependencias del proyecto para replicar el entorno de desarrollo.

## Requisitos de instalación

Para ejecutar los cuadernos y scripts de Python en este repositorio, se requiere Python 3 y las bibliotecas enumeradas en `requirements.txt`. Puedes instalar las dependencias utilizando pip:

`pip install -r requirements.txt`


## Uso

1. Clona este repositorio en tu máquina local:

`git clone https://github.com/tu_usuario/titanic-project.git `

2. Instala las dependencias necesarias:

`pip install -r requirements.txt`

3. Explora los cuadernos Jupyter en la carpeta `notebooks/` para comprender el proceso de análisis y modelado.

4. Ejecuta los scripts Python en la carpeta principal para entrenar y evaluar modelos.

## Contribución

Si quieres contribuir a este proyecto, sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una rama para tu nueva funcionalidad: `git checkout -b nueva-funcionalidad`.
3. Realiza tus cambios y haz commit de ellos: `git commit -am 'Agrega nueva funcionalidad'`.
4. Haz push a la rama: `git push origin nueva-funcionalidad`.
5. Crea un pull request en GitHub.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más detalles.



