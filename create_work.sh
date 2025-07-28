if [ -z "$1" ]; then
  echo "Uso: $0 <nombre_del_trabajo>"
  exit 1
fi

WORKDIR="$1"

echo "Creando carpeta de trabajo '$WORKDIR'..."
mkdir -p "$WORKDIR"/{src,notebooks}

echo "Generando archivos base..."
touch "$WORKDIR"/{Dockerfile,docker-compose.yml,Makefile,README.md}

cat > "$WORKDIR/README.md" <<EOF
# $WORKDIR

Descripción: [Breve descripción del trabajo]

## 🚀 Levantar el entorno

dd \${PWD}/$WORKDIR && make up
EOF

echo "Plantilla creada en '$WORKDIR'"