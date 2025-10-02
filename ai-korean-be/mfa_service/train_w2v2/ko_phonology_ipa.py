# ko_phonology_ipa.py (updated)
# Korean phonology (surface realization) for IPA phoneme sequences.
# Operates on a *list* of IPA phones (e.g., ["k", "o", "t", "i", ...]).
# Mirrors key decisions in ko_phonology.py:
#  - n-insertion BEFORE liaison with specific skips (approximate at phone level)
#  - liaison does NOT move ŋ; complex coda with h (…n+h, …l+h): drop h and move the first unit (n/l)
#  - palatalization (d/t + i) ONLY when liaison occurred at that boundary
#  - coda d/t + "hi" -> onset ʧ (aspirated affricate) (approximate)
#  - ㅎ deletion and aspiration interactions
#  - tensification A/B as usual; tensification C (l + {k,t,s,t͡ɕ}) is BLOCKED when liaison fired at that boundary
#
# API:
#   apply_korean_phonology(ipa_list) -> (surface_ipa_list, rule_tags)
#   apply_korean_phonology_str("k o t i") -> ("k o t͡ɕ i", rule_tags)
#
from typing import List, Tuple, Optional

# --- Symbol sets (extend to match your G2P) ---
VOWELS = {
    "a","e","i","o","u",
    "ɐ","ʌ","ɛ","ɯ","ø","y","ɨ","ə","ɤ","ʊ","œ","ɜ","æ","ʉ","ɒ"
}
GLIDES = {"j","w"}  # allow 'j' (y) and 'w' as onglides
H_LIKE = {"h","ɦ"}

# Obstruents / candidates for tensification and aspiration
LAX_ONSET = {"k","t","p","s","t͡ɕ"}                        # ㄱ,ㄷ,ㅂ,ㅅ,ㅈ
TENSE_MAP = {"k":"k͈","t":"t͈","p":"p͈","s":"s͈","t͡ɕ":"t͡ɕ͈"}  # ㄲ,ㄸ,ㅃ,ㅆ,ㅉ
ASP_MAP   = {"k":"kʰ","t":"tʰ","p":"pʰ","t͡ɕ":"t͡ɕʰ"}            # ㅋ,ㅌ,ㅍ, ㅊ

NASALS = {"m","n","ŋ"}
LIQUID = {"l"}

def _is_vowel(x: Optional[str]) -> bool:
    return x in VOWELS if x else False

def _segment_syllables(ipa: List[str]) -> List[Tuple[List[str], List[str]]]:
    """
    Rough syllabification:
      - vowel starts a syllable; preceding consonants become onset
      - consonants after the first vowel in a syllable go to coda
    Returns: list of (onset, rime), where rime=[nucleus-vowel, optional coda...]
    """
    syllables: List[Tuple[List[str], List[str]]] = []
    cur_onset: List[str] = []
    cur_rime: List[str] = []
    seen_vowel = False
    for ph in ipa:
        if _is_vowel(ph):
            if seen_vowel:
                syllables.append((cur_onset, cur_rime))
                cur_onset, cur_rime = [], []
            seen_vowel = True
            cur_rime.append(ph)
        else:
            if not seen_vowel:
                cur_onset.append(ph)
            else:
                cur_rime.append(ph)
    if seen_vowel:
        syllables.append((cur_onset, cur_rime))
    elif cur_onset:  # no vowel at all (rare)
        syllables.append((cur_onset, []))
    return syllables

def _join_syllables(syllables: List[Tuple[List[str], List[str]]]) -> List[str]:
    out: List[str] = []
    for onset, rime in syllables:
        out.extend(onset)
        out.extend(rime)
    return out

def _apply_eui_reading_basic(sylls: List[Tuple[List[str], List[str]]]) -> None:
    """
    Very rough handling for '의' at IPA level:
      - If nucleus looks like [ɯ i], compress to [i] after the first syllable.
    """
    for idx, (onset, rime) in enumerate(sylls):
        if len(rime) >= 2 and rime[0] == "ɯ" and rime[1] == "i":
            if idx > 0:
                sylls[idx] = (onset, ["i"] + rime[2:])

def _last_coda(rime: List[str]) -> Optional[str]:
    # return the last non-vowel element of rime (if any)
    last = None
    seen_vowel = False
    for ph in rime:
        if _is_vowel(ph):
            seen_vowel = True
            last = None
        else:
            if seen_vowel:
                last = ph
    return last

def _coda_list(rime: List[str]) -> List[str]:
    # return list of consonants after the first vowel in rime
    out: List[str] = []
    seen_vowel = False
    for ph in rime:
        if _is_vowel(ph):
            seen_vowel = True
            out.clear()
        else:
            if seen_vowel:
                out.append(ph)
    return out

def _rewrite_rime_with_coda(rime: List[str], new_coda: List[str]) -> List[str]:
    out: List[str] = []
    seen_vowel = False
    for ph in rime:
        if _is_vowel(ph) and not seen_vowel:
            out.append(ph)
            seen_vowel = True
        elif seen_vowel:
            # skip old coda; will add new later
            pass
        else:
            # pre-vowel consonants shouldn't exist in rime by construction, but keep safe
            out.append(ph)
    out.extend(new_coda)
    return out

def apply_korean_phonology(ipa: List[str]) -> Tuple[List[str], List[str]]:
    """
    Apply Korean phonology rules (surface) to a base IPA sequence:
      1) 연음화 (liaison), incl. complex coda with ㅎ: drop ㅎ and move first unit (≈ n/l) to onset
      2) ㄴ 첨가 (n-insertion) before i / j{a,ʌ,o,u} when onset empty (approx.)
      3) 비음화 (nasalization)
      4) 유음비화 (liquidization n/l -> l/l)
      5) 구개음화 (palatalization) t/d/s + (j or i)  — ONLY when liaison fired at boundary (for t/d+i)
      6) ㅎ 탈락 (h deletion) when coda h + next starts with vowel
      7) 격음화 (aspiration) with ㅎ
      8) 경음화 (tensification) A/B/C, with **C disabled** if liaison fired at that boundary
      9) 의 reading basic (compress ɯ i → i after first syllable)
    Returns (surface_ipa_list, rule_tags)
    """
    sylls = _segment_syllables(ipa)
    rule_tags: List[str] = []
    liaison_boundaries = set()  # indices i where liaison happened between i and i+1

    # 9) '의' reading
    _apply_eui_reading_basic(sylls)

    # iterate boundaries i|i+1
    for i in range(len(sylls)-1):
        onset1, rime1 = sylls[i]
        onset2, rime2 = sylls[i+1]
        if not rime1:
            continue

        coda = _coda_list(rime1)
        onset_next = onset2[0] if onset2 else None
        next_starts_with_vowel = (not onset2) and (rime2 and _is_vowel(rime2[0]))

        # --- Helper to write back rime1 after coda edits
        def _commit_coda():
            nonlocal rime1, sylls
            rime1 = _rewrite_rime_with_coda(rime1, coda)
            sylls[i] = (onset1, rime1)

        # 1) Liaison
        if coda and next_starts_with_vowel:
            # Special: if coda ends with h and there is another coda before it (≈ 겹받침 ...+h)
            #   drop h and move the PREVIOUS coda (≈ n/l) to onset.
            if coda[-1] in H_LIKE and len(coda) >= 2 and coda[-2] in {"n","l"}:
                moved = coda[-2]
                # remove both ... n/l and h from coda
                coda = coda[:-2]
                onset2.insert(0, moved)
                rule_tags.append(f"liaison:complex-h(drop h, move {moved})@{i+1}")
                liaison_boundaries.add(i)
                _commit_coda()
            else:
                moved = coda[-1]
                # do not liaise ŋ (no onset ŋ in Korean phonotactics)
                if moved != "ŋ":
                    coda = coda[:-1]
                    onset2.insert(0, moved)
                    rule_tags.append(f"liaison:{i}->{i+1}")
                    liaison_boundaries.add(i)
                    _commit_coda()

        # refresh after liaison
        onset_next = onset2[0] if onset2 else None
        coda_last = coda[-1] if coda else None

        # 2) n-insertion (approximate):
        # If any coda exists, and onset2 empty, and next nucleus is i OR (glide j/w + {a,ʌ,o,u}), insert n.
        if coda and (not onset2) and rime2:
            first = rime2[0]
            second = rime2[1] if len(rime2) >= 2 else None
            need_n = False
            if first == "i":
                # Skip when we expect palatalization case handled by liaison (rough: if liaison occurred and prior coda looked like t/d)
                if (i in liaison_boundaries) and (coda_last in {"t","d"}):
                    need_n = False
                else:
                    need_n = True
            elif first in GLIDES and (second in {"a","ʌ","o","u"}):
                need_n = True
            if need_n:
                onset2.insert(0, "n")
                rule_tags.append(f"n-insertion@{i+1}")
                onset_next = onset2[0]

        # 3) Nasalization
        if coda_last in {"k","ɡ"} and (onset_next in {"n","m"}):
            coda[-1] = "ŋ"; rule_tags.append(f"nasalization:k->ŋ@{i}"); _commit_coda()
        elif coda_last in {"t","d"} and (onset_next in {"n","m"}):
            coda[-1] = "n"; rule_tags.append(f"nasalization:t/d->n@{i}"); _commit_coda()
        elif coda_last in {"p","b"} and (onset_next in {"n","m"}):
            coda[-1] = "m"; rule_tags.append(f"nasalization:p/b->m@{i}"); _commit_coda()

        # m, ŋ + l -> onset n
        onset_next = onset2[0] if onset2 else None
        if coda_last in {"m","ŋ"} and onset_next == "l":
            onset2[0] = "n"; rule_tags.append(f"nasalization:m/ŋ + l -> n@{i+1}")

        # k/p/ɡ/b + l -> onset n
        if coda_last in {"k","p","ɡ","b"} and onset_next == "l":
            onset2[0] = "n"; rule_tags.append(f"nasalization:k/p + l -> n@{i+1}")

        # 4) Liquidization
        onset_next = onset2[0] if onset2 else None
        if coda_last == "n" and onset_next == "l":
            coda[-1] = "l"; onset2[0] = "l"; rule_tags.append(f"liquidization:n+l->l l@{i}"); _commit_coda()
        elif coda_last == "l" and onset_next == "n":
            onset2[0] = "l"; rule_tags.append(f"liquidization:l+n->l l@{i+1}")

        # 5) Palatalization
        # (a) d/t + i at LIAISON boundary only
        onset_next = onset2[0] if onset2 else None
        if (i in liaison_boundaries) and onset_next in {"t","d"}:
            # if next has nucleus i (or onset glide j + i), convert to affricate
            if (rime2 and rime2[0] == "i") or (len(rime2) >= 2 and rime2[0] in GLIDES and rime2[1] == "i"):
                onset2[0] = "t͡ɕ" if onset_next == "t" else "d͡ʑ"
                rule_tags.append(f"palatalization:{onset_next}+i@{i+1}")

        # (b) s + (j or i) anywhere
        onset_next = onset2[0] if onset2 else None
        if onset_next == "s":
            nxt2 = onset2[1] if len(onset2) >= 2 else (rime2[1] if (rime2 and len(rime2)>=2) else None)
            if (rime2 and rime2[0] == "i") or (nxt2 in GLIDES) or (len(rime2)>=2 and rime2[0] in GLIDES and rime2[1]=="i"):
                onset2[0] = "ɕ"; rule_tags.append(f"palatalization:s->ɕ@{i+1}")

        # (c) coda d/t + "hi" -> onset ʧ (approx to t͡ɕʰ)
        if coda_last in {"t","d"} and onset2 and onset2[0] in H_LIKE and rime2 and rime2[0]=="i":
            onset2[0] = "t͡ɕʰ"; rule_tags.append(f"palatalization:(d/t)+hi->t͡ɕʰ@{i+1}")

        # 6) ㅎ deletion
        if coda_last in H_LIKE and (not onset2) and (rime2 and _is_vowel(rime2[0])):
            # drop coda h
            coda = coda[:-1]; rule_tags.append(f"h-deletion@{i}"); _commit_coda()

        # 7) Aspiration with ㅎ
        onset_next = onset2[0] if onset2 else None
        if coda_last in {"k","t"} and onset_next in H_LIKE:
            onset2[0] = ASP_MAP["k"] if coda_last == "k" else ASP_MAP["t"]
            rule_tags.append(f"aspiration:{coda_last}+h->{onset2[0]}@{i+1}")
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None
        if coda_last in H_LIKE and onset_next in {"k","t"}:
            onset2[0] = ASP_MAP[onset_next]
            rule_tags.append(f"aspiration:h+{onset_next}->{onset2[0]}@{i+1}")

        # 8) Tensification
        onset_next = onset2[0] if onset2 else None
        if onset_next in LAX_ONSET:
            # A) coda in {k,t,p} -> tense
            if coda_last in {"k","t","p"}:
                t = TENSE_MAP.get(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t; rule_tags.append(f"tensification:A {coda_last}+{onset_next}->{t}@{i+1}")
            # B) coda in {n,m} and onset in {k,t,t͡ɕ}
            if coda_last in {"n","m"} and onset_next in {"k","t","t͡ɕ"}:
                t = TENSE_MAP.get(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t; rule_tags.append(f"tensification:B {coda_last}+{onset_next}->{t}@{i+1}")
            # C) coda l + onset in {k,t,s,t͡ɕ} -> tense (BUT skip if liaison happened here)
            if (i not in liaison_boundaries) and coda_last == "l" and onset_next in {"k","t","s","t͡ɕ"}:
                t = TENSE_MAP.get(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t; rule_tags.append(f"tensification:C l+{onset_next}->{t}@{i+1}")

        # write back boundary syllables
        sylls[i+1] = (onset2, rime2)
        _commit_coda()

    ipa_surface = _join_syllables(sylls)
    return ipa_surface, rule_tags

# Convenience wrapper for space-separated strings
def apply_korean_phonology_str(ipa_str: str) -> Tuple[str, List[str]]:
    toks = [t for t in ipa_str.split() if t]
    out, tags = apply_korean_phonology(toks)
    return " ".join(out), tags
