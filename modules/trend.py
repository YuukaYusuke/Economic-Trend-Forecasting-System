def get_direction(current: float, reference: float) -> str:
    """Kembalikan arah pergerakan: NAIK, TURUN, atau DATAR."""
    if current > reference:
        return "NAIK"
    if current < reference:
        return "TURUN"
    return "DATAR"


def direction_label(direction: str) -> str:
    icons = {"NAIK": "📈", "TURUN": "📉", "DATAR": "➡️"}
    return f"{direction} {icons.get(direction, '')}"


def match_status(predicted: str, actual: str) -> tuple[str, bool]:
    """Bandingkan arah prediksi dengan arah data aktual (API)."""
    if predicted == "DATAR" or actual == "DATAR":
        return "Arah belum jelas (nilai hampir sama)", False
    if predicted == actual:
        return "Prediksi sesuai arah kurs aktual (API)", True
    return "Prediksi tidak sesuai arah kurs aktual (API)", False
