from io import BytesIO

import pytest
from PIL import Image

from curadoria_web.image_validation import decode_image


def png_bytes(size=(32, 24)):
    buffer = BytesIO()
    Image.new('L', size, 128).save(buffer, format='PNG')
    return buffer.getvalue()


def test_decode_image_validates_and_loads_pixels():
    image = decode_image(png_bytes())
    assert image.mode == 'L'
    assert image.size == (32, 24)


def test_decode_image_rejects_invalid_container():
    with pytest.raises(ValueError):
        decode_image(b'not-an-image')


def test_decode_image_rejects_excessive_dimensions():
    with pytest.raises(ValueError):
        decode_image(png_bytes((9000, 1)))
