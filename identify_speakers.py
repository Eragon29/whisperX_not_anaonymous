"""
scripts/identify_speakers.py

Usage :
    python scripts/identify_speakers.py --audio reunion.wav --speakers-db speakers.json

Entrées :
  --audio        : le fichier audio de la réunion
  --speakers-db  : JSON des voiceprints déjà connus, potentiellement plus
                   large que les personnes réellement présentes dans CET
                   audio précis -- ex. {"Bob": [0.12, -0.4, ...], "Alice": [...]}

Sortie :
  Un JSON (par défaut <audio>.identified.json) contenant le transcript aligné,
  avec les labels de locuteur remplacés par les vrais noms quand reconnus
  (sinon "Inconnu_1", "Inconnu_2", ...).

Avec --enroll-unknown : les locuteurs non reconnus sont ajoutés au fichier
--speakers-db avec leur embedding détecté, pour être reconnus la prochaine fois.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


def load_speakers_db(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_speakers_db(path, db):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def match_speakers(detected_embeddings, speakers_db, seuil=0.3):
    """
    detected_embeddings : {"SPEAKER_00": [...], ...}         -- sortie DiarizationPipeline
    speakers_db         : {"Bob": [...], "Alice": [...]}  -- base connue (peut être plus large)

    Assignation bipartite (scipy.linear_sum_assignment) pour éviter que deux
    locuteurs détectés différents matchent accidentellement le même nom connu.

    Retourne (mapping, nouveaux) :
      mapping  : {"SPEAKER_00": "Bob", "SPEAKER_01": "Inconnu_1"}
      nouveaux : {"Inconnu_1": [...]}  -- embeddings des non-reconnus, à enrôler si besoin
    """
    detected_labels = list(detected_embeddings.keys())
    known_names = list(speakers_db.keys())

    mapping = {}
    assigned_rows = set()

    if known_names and detected_labels:
        det_matrix = [detected_embeddings[l] for l in detected_labels]
        db_matrix = [speakers_db[n] for n in known_names]
        dist_matrix = np.array([[cosine(d, k) for k in db_matrix] for d in det_matrix])
        rows, cols = linear_sum_assignment(dist_matrix)
        for r, c in zip(rows, cols):
            if dist_matrix[r, c] < seuil:
                mapping[detected_labels[r]] = known_names[c]
                assigned_rows.add(r)

    nouveaux = {}
    compteur = 1
    for i, label in enumerate(detected_labels):
        if i in assigned_rows:
            continue
        nom = f"Inconnu_{compteur}"
        while nom in speakers_db or nom in nouveaux:
            compteur += 1
            nom = f"Inconnu_{compteur}"
        mapping[label] = nom
        nouveaux[nom] = detected_embeddings[label]
        compteur += 1

    return mapping, nouveaux


def apply_mapping(result, mapping):
    for seg in result.get("segments", []):
        if seg.get("speaker") in mapping:
            seg["speaker"] = mapping[seg["speaker"]]
        for word in seg.get("words", []):
            if word.get("speaker") in mapping:
                word["speaker"] = mapping[word["speaker"]]
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcrit un audio et identifie les locuteurs via voiceprints.")
    parser.add_argument("--audio", required=True, help="Chemin du fichier audio")
    parser.add_argument("--speakers-db", required=True, help="Chemin du JSON de voiceprints connus")
    parser.add_argument("--out", default=None, help="Chemin de sortie JSON (défaut: <audio>.identified.json)")
    parser.add_argument("--model", default="small", help="Modèle Whisper (small, medium, large-v3...)")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")
    parser.add_argument("--hf-token", default=None, help="Token HuggingFace (modèle de diarisation)")
    parser.add_argument("--seuil", type=float, default=0.3, help="Seuil de distance cosinus pour le matching")
    parser.add_argument("--enroll-unknown", action="store_true",
                        help="Ajoute les locuteurs non reconnus au fichier --speakers-db")
    args = parser.parse_args()

    out_path = args.out or str(Path(args.audio).with_suffix("")) + ".identified.json"
    compute_type = "int8" if args.device == "cpu" else "float16"

    print(f"[1/4] Transcription ({args.model})...")
    model = whisperx.load_model(args.model, args.device, compute_type=compute_type)
    audio = whisperx.load_audio(args.audio)
    result = model.transcribe(audio, batch_size=8)

    print("[2/4] Alignement mot-par-mot...")
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=args.device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, args.device, return_char_alignments=False)

    print("[3/4] Diarisation + embeddings...")
    diarize_model = DiarizationPipeline(token=args.hf_token, device=args.device)
    diarize_df, speaker_embeddings = diarize_model(audio, return_embeddings=True)
    result = assign_word_speakers(diarize_df, result, speaker_embeddings)

    print("[4/4] Identification des locuteurs...")
    speakers_db = load_speakers_db(args.speakers_db)
    mapping, nouveaux = match_speakers(speaker_embeddings or {}, speakers_db, seuil=args.seuil)
    print("Correspondance :", mapping)

    result = apply_mapping(result, mapping)

    if args.enroll_unknown and nouveaux:
        speakers_db.update(nouveaux)
        save_speakers_db(args.speakers_db, speakers_db)
        print(f"{len(nouveaux)} nouveau(x) enrôlé(s) dans {args.speakers_db} : {list(nouveaux.keys())}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Résultat écrit dans {out_path}")


if __name__ == "__main__":
    main()