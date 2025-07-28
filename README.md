# Métodos Matemáticos

Este repositorio agrupa los **trabajos** de la asignatura **Métodos Matemáticos** del programa de Maestría en Inteligencia Artificial.

---

## 📁 Estructura del repositorio

Cada trabajo se organiza en su propia carpeta con un nombre **descriptivo** que refleje su contenido:

```
metodos-matematicos/
├─ clasificador_mnist/  ← Carpeta del trabajo “Clasificador MNIST”
│   ├─ Dockerfile
│   ├─ docker-compose.yml
│   ├─ Makefile         ← Atajos para usuarios no familiarizados con Docker
│   ├─ src/            ← Código fuente (scripts, módulos)
│   ├─ notebooks/      ← Notebooks Jupyter
│   └─ README.md       ← Documentación específica del trabajo
├─ regresion_lineal/    ← Carpeta del trabajo “Regresión Lineal”
│   └─ …
└─ README.md           ← Índice y guía general (este archivo)
```

* **\<nombre\_del\_trabajo>/**: Carpeta con nombre claro y conciso.

  * **Dockerfile**: Imagen base y dependencias.
  * **docker-compose.yml**: Servicios necesarios (Jupyter, bases de datos, etc.).
  * **Makefile**: Atajos `make` para simplificar el levantamiento del entorno.
  * **src/**: Scripts y módulos de código.
  * **notebooks/**: Exploraciones en Jupyter.
  * **README.md**: Instrucciones específicas (objetivo, instalación, ejemplos).

---

## 🚀 Cómo entregar un trabajo

1. Copia una carpeta de trabajo existente como plantilla o crea una nueva con un nombre descriptivo, por ejemplo `clasificador_mnist/`.
2. Añade los archivos necesarios (`Dockerfile`, `docker-compose.yml`, `Makefile`, `src/`, `notebooks/`, etc.).
3. Completa el `README.md` de la carpeta con:

   * **Título y descripción** del objetivo.
   * **Requisitos** previos (Docker, Python, etc.).
   * **Pasos de instalación** y ejecución.
   * **Ejemplos de comandos** (entrenamiento, evaluación, etc.).
4. Crea una rama nueva basada en `main` con el patrón `<nombre_del_trabajo>/<tu_usuario>`, por ejemplo:

   ```bash
   git checkout -b clasificador_mnist/camiloGarcia
   ```
5. Guarda y commitea en tu rama:

   ```bash
   git add <nombre_del_trabajo>/
   git commit -m "[ADD] <nombre_del_trabajo>: Descripción breve"
   ```
6. Sube tu rama al repositorio remoto:

   ```bash
   git push origin <nombre_del_trabajo>/<tu_usuario>
   ```
7. Abre un **Pull Request (PR)** desde tu rama hacia `main` para revisión y merge.

---

## ⚙️ Uso de Docker para aislamiento

Para evitar conflictos de librerías entre trabajos, cada carpeta define:

* **Dockerfile** con la imagen base.
* **docker-compose.yml** para orquestar servicios.
* **Makefile** para simplificar comandos Docker.

Ejemplo de ejecución:

```bash
cd metodos-matematicos/clasificador_mnist
make up
```

*(equivalente a `docker-compose up --build`)*

Luego accede a Jupyter Lab en `http://localhost:8888` (o al puerto configurado).

---

## 🛠 Buenas prácticas de Git

* **No hacer push directo** a la rama `main`.
* **Crea siempre** una rama nueva para tu trabajo siguiendo el patrón `<nombre_del_trabajo>/<tu_usuario>`.
* Trabaja y commitea tus cambios solo en tu rama.
* Abre un PR para integrar tus cambios a `main` una vez listos.
* Usa mensajes de commit claros:

  * `[ADD]`: nuevo trabajo o módulo.
  * `[FIX]`: corrección de errores.
  * `[IMP]`: mejoras o refactorización.
* Si renombras carpetas o ficheros, utiliza `git mv` para preservar historial.

---

## 🤝 Contribuciones y soporte

Si detectas errores o sugerencias:

1. Abre un *issue* en GitHub.
2. Propón un *pull request* siguiendo las buenas prácticas.

---

**Autor:** Juan Camilo Sandoval Garcia
**GitHub:** @CamiloGarcia06
**Fecha de creación:** 28 Jul 2025

---

🏗️ Generar plantilla de trabajo desde consola

El script create_work.sh ya está incluido en la raíz del repositorio. Para ejecutarlo, sigue estos pasos:

    Otorga permisos de ejecución:
    chmod +x create_work.sh

    Ejecuta el script indicando el nombre del nuevo trabajo:
    ./create_work.sh <nombre_del_trabajo>

Por ejemplo, para crear una carpeta llamada clasificador_mnist:
./create_work.sh clasificador_mnist

Esto generará la estructura completa con los archivos base (Dockerfile, docker-compose.yml, Makefile, README.md, src/, notebooks/).


