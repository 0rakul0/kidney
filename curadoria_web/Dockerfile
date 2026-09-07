# syntax=docker/dockerfile:1.7
# As wheels sao produzidas fora da imagem final. Assim, ferramentas de build,
# cache do pip e artefatos temporarios nao acompanham o runtime.
FROM python:3.11-slim-bookworm AS wheels

WORKDIR /build
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CURATORIA_DATA_ROOT=/data

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    python -m pip install --no-compile --no-index --find-links=/wheels -r requirements.txt \
    && groupadd --system --gid 10001 curadoria \
    && useradd --system --gid curadoria --uid 10001 --create-home curadoria

COPY --chown=10001:10001 app.py ./curadoria_web/app.py
COPY --chown=10001:10001 static ./curadoria_web/static

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["python", "/app/curadoria_web/app.py"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--manifest", "/data/dataset_aumentado/curadoria/miniaturas_completas/manifest.csv", "--database", "/data/dataset_aumentado/curadoria/respostas/curadoria.sqlite3", "--cache-dir", "/data/dataset_aumentado/curadoria/respostas/overlay_cache"]
