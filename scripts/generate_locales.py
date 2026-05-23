"""Generate locale JSON files from English base + per-language overrides."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "locales"


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Terjemahan penuh di locales/ar.json (jangan ditimpa generator)
MANUAL_LOCALES = frozenset({"ar"})

# Ringkasan override per bahasa (sisanya fallback ke en.json)
OVERRIDES = {
    "ja": {
        "lang_label": "表示言語",
        "language": "言語",
        "sidebar_title": "為替予測",
        "nav_hint": "下のメニューから選択。番号は推奨順序です。",
        "nav": {
            "home": "ホーム",
            "dataset": "データセット",
            "preprocessing": "前処理",
            "visualisasi": "可視化",
            "lstm": "LSTMモデル",
            "prediksi": "予測",
            "evaluasi": "評価",
            "trend": "トレンド",
            "insight": "インサイト",
            "realtime": "リアルタイム",
            "bandingkan": "比較",
            "simulasi": "シミュレーション",
        },
        "dir_up": "上昇",
        "dir_down": "下落",
        "dir_flat": "横ばい",
        "live": "リアルタイム",
        "prediction": "予測",
        "direction": "方向",
        "refresh": "更新",
        "all": "すべて",
        "yes": "はい",
        "no": "いいえ",
        "search": "検索",
        "loading": "読み込み中…",
        "dark_mode": "ダークモード",
        "models_ready": "モデル準備完了",
        "confidence": "信頼度",
        "strong": "強",
        "medium": "中",
        "weak": "弱",
        "disclaimer": "投資助言ではありません。予測は参考値です。",
        "pages": {
            "home": {
                "title": "経済トレンド予測",
                "subtitle": "為替トレンド分析と方向予測 — わかりやすく視覚的。",
                "about": "概要",
                "exec_kpi": "エグゼクティブサマリー",
                "fav_currency": "監視通貨",
                "pred_today": "予測方向",
            },
            "prediksi": {
                "title": "為替方向予測",
                "subtitle": "学習済みLSTMとリアルタイムAPIで方向を推定。",
            },
            "realtime": {"title": "リアルタイム為替", "subtitle": "APIの全通貨とヒートマップ。"},
        },
    },
    "zh": {
        "lang_label": "显示语言",
        "language": "语言",
        "nav_hint": "请使用下方菜单，数字为建议顺序。",
        "nav": {
            "home": "首页",
            "dataset": "数据集",
            "preprocessing": "预处理",
            "visualisasi": "可视化",
            "lstm": "LSTM模型",
            "prediksi": "预测",
            "evaluasi": "评估",
            "trend": "趋势",
            "insight": "洞察",
            "realtime": "实时汇率",
            "bandingkan": "对比",
            "simulasi": "模拟",
        },
        "dir_up": "上涨",
        "dir_down": "下跌",
        "dir_flat": "持平",
        "pages": {
            "home": {"title": "经济趋势预测", "subtitle": "汇率趋势分析与方向预测。"},
            "prediksi": {"title": "汇率方向预测", "subtitle": "LSTM模型结合实时API。"},
        },
    },
    "ko": {
        "lang_label": "표시 언어",
        "language": "언어",
        "nav_hint": "아래 메뉴를 사용하세요. 숫자는 권장 순서입니다.",
        "nav": {
            "home": "홈",
            "dataset": "데이터셋",
            "preprocessing": "전처리",
            "visualisasi": "시각화",
            "lstm": "LSTM 모델",
            "prediksi": "예측",
            "evaluasi": "평가",
            "trend": "추세",
            "insight": "인사이트",
            "realtime": "실시간",
            "bandingkan": "비교",
            "simulasi": "시뮬레이션",
        },
        "dir_up": "상승",
        "dir_down": "하락",
        "dir_flat": "보합",
        "pages": {
            "home": {"title": "경제 트렌드 예측", "subtitle": "환율 추세 및 방향 예측."},
            "prediksi": {"title": "환율 방향 예측", "subtitle": "LSTM과 실시간 API."},
        },
    },
    "ms": {
        "lang_label": "Bahasa paparan",
        "language": "Bahasa",
        "nav_hint": "Gunakan menu di bawah. Nombor menunjukkan urutan cadangan.",
        "nav": {
            "home": "Utama",
            "dataset": "Set data",
            "preprocessing": "Prapemprosesan",
            "visualisasi": "Visualisasi",
            "lstm": "Model LSTM",
            "prediksi": "Ramalan",
            "evaluasi": "Penilaian",
            "trend": "Trend",
            "insight": "Insight",
            "realtime": "Masa nyata",
            "bandingkan": "Bandingkan",
            "simulasi": "Simulasi",
        },
        "dir_up": "NAIK",
        "dir_down": "TURUN",
        "pages": {
            "home": {"title": "Ramalan Trend Ekonomi", "subtitle": "Analisis arah kadar pertukaran."},
        },
    },
    "es": {
        "lang_label": "Idioma",
        "language": "Idioma",
        "nav_hint": "Use el menú inferior. Los números indican el orden sugerido.",
        "nav": {
            "home": "Inicio",
            "dataset": "Dataset",
            "preprocessing": "Preproceso",
            "visualisasi": "Visualización",
            "lstm": "Modelo LSTM",
            "prediksi": "Pronóstico",
            "evaluasi": "Evaluación",
            "trend": "Tendencia",
            "insight": "Insight",
            "realtime": "Tiempo real",
            "bandingkan": "Comparar",
            "simulasi": "Simulación",
        },
        "dir_up": "SUBE",
        "dir_down": "BAJA",
        "pages": {
            "home": {"title": "Pronóstico de tendencia", "subtitle": "Análisis y pronóstico de divisas."},
        },
    },
    "fr": {
        "lang_label": "Langue",
        "language": "Langue",
        "nav_hint": "Utilisez le menu ci-dessous. Les numéros indiquent l'ordre suggéré.",
        "nav": {
            "home": "Accueil",
            "dataset": "Jeu de données",
            "preprocessing": "Prétraitement",
            "visualisasi": "Visualisation",
            "lstm": "Modèle LSTM",
            "prediksi": "Prévision",
            "evaluasi": "Évaluation",
            "trend": "Tendance",
            "insight": "Insight",
            "realtime": "Temps réel",
            "bandingkan": "Comparer",
            "simulasi": "Simulation",
        },
        "dir_up": "HAUSSE",
        "dir_down": "BAISSE",
        "pages": {
            "home": {"title": "Prévision de tendance", "subtitle": "Analyse et prévision des devises."},
        },
    },
    "de": {
        "lang_label": "Anzeigesprache",
        "language": "Sprache",
        "nav_hint": "Menü unten verwenden. Zahlen zeigen die empfohlene Reihenfolge.",
        "nav": {
            "home": "Start",
            "dataset": "Datensatz",
            "preprocessing": "Vorverarbeitung",
            "visualisasi": "Visualisierung",
            "lstm": "LSTM-Modell",
            "prediksi": "Prognose",
            "evaluasi": "Bewertung",
            "trend": "Trend",
            "insight": "Insight",
            "realtime": "Echtzeit",
            "bandingkan": "Vergleich",
            "simulasi": "Simulation",
        },
        "dir_up": "AUF",
        "dir_down": "AB",
        "pages": {
            "home": {"title": "Wirtschaftsprognose", "subtitle": "Wechselkursanalyse und Richtungsprognose."},
        },
    },
    "pt": {
        "lang_label": "Idioma",
        "language": "Idioma",
        "nav_hint": "Use o menu abaixo. Os números indicam a ordem sugerida.",
        "nav": {
            "home": "Início",
            "dataset": "Dataset",
            "preprocessing": "Pré-processamento",
            "visualisasi": "Visualização",
            "lstm": "Modelo LSTM",
            "prediksi": "Previsão",
            "evaluasi": "Avaliação",
            "trend": "Tendência",
            "insight": "Insight",
            "realtime": "Tempo real",
            "bandingkan": "Comparar",
            "simulasi": "Simulação",
        },
        "dir_up": "ALTA",
        "dir_down": "BAIXA",
        "pages": {
            "home": {"title": "Previsão de tendência", "subtitle": "Análise e previsão cambial."},
        },
    },
    "th": {
        "lang_label": "ภาษาที่แสดง",
        "language": "ภาษา",
        "nav_hint": "ใช้เมนูด้านล่าง ตัวเลขคือลำดับที่แนะนำ",
        "nav": {
            "home": "หน้าแรก",
            "dataset": "ชุดข้อมูล",
            "preprocessing": "การประมวลผล",
            "visualisasi": "การแสดงผล",
            "lstm": "โมเดล LSTM",
            "prediksi": "พยากรณ์",
            "evaluasi": "ประเมินผล",
            "trend": "แนวโน้ม",
            "insight": "ข้อมูลเชิงลึก",
            "realtime": "เรียลไทม์",
            "bandingkan": "เปรียบเทียบ",
            "simulasi": "จำลอง",
        },
        "dir_up": "ขึ้น",
        "dir_down": "ลง",
        "dir_flat": "คงที่",
        "pages": {
            "home": {"title": "พยากรณ์เทรนด์เศรษฐกิจ", "subtitle": "วิเคราะห์และพยากรณ์อัตราแลกเปลี่ยน"},
        },
    },
}


def main():
    base = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))
    for code, patch in OVERRIDES.items():
        if code in MANUAL_LOCALES:
            print("skip (manual)", code)
            continue
        merged = deep_merge(base, patch)
        (LOCALES / f"{code}.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("wrote", code)


if __name__ == "__main__":
    main()
