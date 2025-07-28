# Métodos Matemáticos

Este repositorio agrupa los **trabajos** de la asignatura **Métodos Matemáticos** del programa de Maestría en Inteligencia Artificial.

---

## 📁 Estructura del repositorio

Cada trabajo se organiza en su propia carpeta numerada:

```
metodos-matematicos/
├─ trabajo-01/         ← Carpeta del Trabajo 01
│   ├─ Dockerfile
│   ├─ docker-compose.yml
│   ├─ Makefile         ← Comandos básicos para usuarios no familiarizados con Docker
│   ├─ src/            ← Código fuente (scripts, módulos)
│   ├─ notebooks/      ← Notebooks Jupyter
│   └─ README.md       ← Documentación específica del trabajo
├─ trabajo-02/
│   └─ …
└─ README.md           ← Índice y guía general (este archivo)
```

* **trabajo-XX/**: Carpeta del trabajo número XX.

  * **Dockerfile**: Imagen base y dependencias.
  * **docker-compose.yml**: Servicios necesarios (Jupyter, bases de datos, etc.).
  * **Makefile**: Atajos `make` que simplifican la ejecución de comandos Docker y levantamiento del entorno, pensado para quienes no manejan Docker directamente.
  * **src/**: Scripts y módulos Python u otros lenguajes.
  * **notebooks/**: Análisis y visualizaciones en Jupyter.
  * **README.md**: Instrucciones específicas (objetivo, instalación, ejemplos de uso).

---

## 🚀 Cómo entregar un trabajo

1. Copiar la carpeta de un trabajo existente como plantilla o crear `trabajo-XX` desde cero.
2. Añadir los archivos necesarios (`Dockerfile`, `docker-compose.yml`, `Makefile`, `src/`, `notebooks/`, etc.).
3. Completar el `README.md` de la carpeta con:

   * **Descripción** del objetivo y alcance.
   * **Requisitos** previos (Docker, Python, etc.).
   * **Pasos de instalación** y ejecución.
   * **Ejemplos de comandos** (entrenamiento, evaluación, etc.).
4. Guardar y commitear:

   ```bash
   git add trabajo-XX/
   git commit -m "[ADD] trabajo-XX: Título breve"
   ```
5. Push al repositorio:

   ```bash
   git push origin main
   ```

---

## ⚙️ Uso de Docker para aislamiento

Para evitar conflictos de librerías entre trabajos, cada carpeta define:

* **Dockerfile** con la imagen base (p.ej. `python:3.10`).
* **docker-compose.yml** para orquestar servicios.
* **Makefile** para simplificar el levantamiento del entorno y facilitar a usuarios no familiarizados con Docker.

Ejemplo de ejecución:

```bash
cd metodos-matematicos/trabajo-01
make up
```

*(equivalente a `docker-compose up --build`)*

Luego accede a Jupyter Lab en `http://localhost:8888` (o al puerto configurado).

---

## 🛠 Buenas prácticas de Git

* Trabaja siempre en la rama `main` para entregas finales.
* Emplea mensajes de commit claros:

  * `[ADD]`: nueva entrega o módulo.
  * `[FIX]`: corrección de errores.
  * `[IMP]`: mejoras o refactor.
* Mantén un historial limpio; usa `git mv` si renombras carpetas.

---

## 🤝 Contribuciones y soporte

Si detectas errores o tienes sugerencias:

1. Abre un *issue* en GitHub.
2. Propón un *pull request* siguiendo las buenas prácticas.

---

**Autor:** Juan Camilo Sandoval Garcia
**GitHub:** @CamiloGarcia06
**Fecha de creación:** 28 Jul 2025
