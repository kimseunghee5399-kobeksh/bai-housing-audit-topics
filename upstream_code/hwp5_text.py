"""Minimal HWP 5.0 text extractor (OLE compound file -> BodyText/ViewText sections -> PARA_TEXT records).

Handles ordinary documents (BodyText, zlib-compressed) and distribution documents (배포용 문서: ViewText
sections encrypted with AES-128-ECB, key derived from the HWPTAG_DISTRIBUTE_DOC_DATA record, as in pyhwp).
Sufficient for text mining; keeps table cell text, drops controls/inline objects.
Usage: python3 hwp5_text.py in.hwp > out.txt
"""
import sys, re, zlib, struct
import olefile

HWPTAG_BEGIN = 0x10
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51      # 67
HWPTAG_DISTRIBUTE_DOC_DATA = HWPTAG_BEGIN + 12  # 28


def records(data):
    pos, n = 0, len(data)
    while pos + 4 <= n:
        hdr = struct.unpack('<I', data[pos:pos + 4])[0]
        tag = hdr & 0x3FF
        level = (hdr >> 10) & 0x3FF
        size = (hdr >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            size = struct.unpack('<I', data[pos:pos + 4])[0]
            pos += 4
        yield tag, level, data[pos:pos + size]
        pos += size


def para_text(payload):
    out = []
    i, n = 0, len(payload)
    while i + 1 < n:
        ch = struct.unpack('<H', payload[i:i + 2])[0]
        i += 2
        if ch < 32:
            if ch in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                i += 14  # extended control: 7 more WCHARs
            if ch in (10, 13):
                out.append('\n')
            elif ch in (9, 24, 25, 30, 31):
                out.append(' ')
            continue
        out.append(chr(ch))
    return ''.join(out)


class MSRand:
    def __init__(self, seed):
        self.seed = seed & 0xFFFFFFFF

    def rand(self):
        self.seed = (self.seed * 214013 + 2531011) & 0xFFFFFFFF
        return (self.seed >> 16) & 0x7FFF


def distdoc_key(record_payload):
    data = bytearray(record_payload[:256])
    seed = struct.unpack('<I', bytes(data[0:4]))[0]
    rnd = MSRand(seed)
    n = 0
    key = 0
    for i in range(256):
        if n == 0:
            key = rnd.rand() & 0xFF
            n = (rnd.rand() & 0xF) + 1
        if i >= 4:
            data[i] ^= key
        n -= 1
    offset = 4 + (seed & 0xF)
    sha1 = bytes(data[offset:offset + 80])
    return sha1[:16]


def decrypt_distdoc(stream):
    from Crypto.Cipher import AES
    hdr = struct.unpack('<I', stream[:4])[0]
    tag = hdr & 0x3FF
    size = (hdr >> 20) & 0xFFF
    if tag != HWPTAG_DISTRIBUTE_DOC_DATA or size != 256:
        raise RuntimeError('not a distribution section')
    key = distdoc_key(stream[4:260])
    body = stream[260:]
    body = body[:len(body) - (len(body) % 16)]
    dec = AES.new(key, AES.MODE_ECB).decrypt(body)
    return dec


def extract(path):
    ole = olefile.OleFileIO(path)
    hdr = ole.openstream('FileHeader').read()
    flags = struct.unpack('<I', hdr[36:40])[0]
    compressed = flags & 1
    if flags & 2:
        raise RuntimeError('encrypted HWP')
    distribution = (flags >> 2) & 1
    root = 'ViewText' if distribution and ole.exists('ViewText') else 'BodyText'
    sections = sorted([e for e in ole.listdir() if e[0] == root], key=lambda e: int(e[1].replace('Section', '')))
    texts = []
    for ent in sections:
        raw = ole.openstream(ent).read()
        if root == 'ViewText':
            raw = decrypt_distdoc(raw)
        if compressed:
            try:
                raw = zlib.decompressobj(-15).decompress(raw)
            except zlib.error:
                raw = zlib.decompressobj(-15).decompress(raw[:len(raw) - 16]) if len(raw) > 16 else b''
        for tag, level, payload in records(raw):
            if tag == HWPTAG_PARA_TEXT:
                texts.append(para_text(payload))
    txt = '\n'.join(texts)
    return re.sub(r'[\ud800-\udfff\x00-\x08\x0b\x0c\x0e-\x1f]', '', txt)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(extract(sys.argv[1]))
