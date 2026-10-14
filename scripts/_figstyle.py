"""Pengaturan figure bersama. Figure digambar langsung pada ukuran akhir di naskah (satu kolom
IJIES, sekitar 8,3 cm), sehingga huruf 10 pt tetap 10 pt di Word dan tidak perlu diperkecil.
Pasang gambar di Word pada 100% (jangan diperbesar atau diperkecil)."""
import textwrap

FIG_W = 3.3      # lebar figure dalam inci (sekitar 8,4 cm = lebar satu kolom)
FONT = 10        # ukuran huruf (pt), minimal 10 pt
TITLE_FONT = 10
DPI = 600


def wrap(title: str, width: int = 30) -> str:
    """Pecah judul menjadi beberapa baris agar muat dalam lebar satu kolom."""
    return "\n".join(textwrap.wrap(title, width))