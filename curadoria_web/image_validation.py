"""Validation of uploaded and locally recovered raster images."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError


MAX_RAW_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def decode_image(raw: bytes, *, max_raw_bytes: int = MAX_RAW_BYTES) -> Image.Image:
    """Decode a complete image after checking format, dimensions and pixels.

    The file is opened twice deliberately: ``verify`` checks the container
    without decoding all pixels, then ``load`` forces decompression so the
    pixel budget is applied to the actual raster rather than only the upload
    byte size.
    """
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("Envie uma imagem raster válida.")
    if len(raw) > max_raw_bytes:
        raise ValueError("A imagem excede o limite de 20 MB.")
    try:
        with Image.open(BytesIO(raw)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise ValueError("Use PNG, JPEG ou WebP.")
            width, height = probe.size
            if width < 8 or height < 8:
                raise ValueError("A imagem possui dimensões insuficientes.")
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ValueError("A imagem excede a dimensão máxima permitida.")
            if width * height > MAX_PIXELS:
                raise ValueError("A imagem descomprimida excede o limite de pixels.")
            probe.verify()
        with Image.open(BytesIO(raw)) as source:
            source.load()
            return source.convert("L")
    except ValueError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("Não foi possível decodificar a imagem enviada.") from error
