# ko_phonology.py
# Korean phonology (surface realization) for IPA phoneme sequences.
# Implements 9 rules as requested (연음화, 경음화, 비음화, 유음비화, 구개음화, ㅎ-탈락, 격음화, 의-발음, ㄴ첨가).
from typing import List, Tuple, Optional

# --- Symbol sets (extend to match your G2P) ---
VOWELS = {
    "a","e","i","o","u",
    "ɐ","ʌ","ɛ","ɯ","ø","y","ɨ","ə","ɤ","ʊ","œ","ɜ","æ","ʉ","ɒ"
}
# sometimes 의 ~ ɯ + i ; you may also encounter a glide ɰ (not treated as vowel here)
GLIDES = {"j"}  # 'y' sound
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

# ---- Rule 8: 의 (very approximate without morphology) -------------------------
def _apply_eui_reading_basic(sylls: List[Tuple[List[str], List[str]]]) -> None:
    """
    Basic treatment for '의':
      - If a syllable's rime starts with [ɯ, i] sequence, keep at syllable 0; else compress to [i].
      - Reading as [e] when '의' = genitive 'of' cannot be reliably detected without morphology.
    This operates in-place on syllables.
    """
    for idx, (onset, rime) in enumerate(sylls):
        if len(rime) >= 2 and rime[0] == "ɯ" and rime[1] == "i":
            if idx == 0:
                # keep [ɯ i]
                pass
            else:
                # compress to [i]
                sylls[idx] = (onset, ["i"] + rime[2:])

# ---- Core rules across syllable boundaries ------------------------------------
def apply_korean_phonology(ipa: List[str]) -> Tuple[List[str], List[str]]:
    """
    Apply Korean phonology rules (surface) to a base IPA sequence:
      1) 연음화 (liaison), including double-final (multi-consonant coda -> move last to next onset if next begins with vowel)
      2) ㄴ 첨가 (n-insertion) before /i, ja, jʌ, jo, ju/ when previous ends with consonant and next onset empty
      3) 비음화 (nasalization)
      4) 유음비화 (liquidization n/l -> l/l)
      5) 구개음화 (palatalization) t/d/s + (j|i)
      6) ㅎ 탈락 (h deletion) when h + vowel
      7) 격음화 (aspiration) with ㅎ (both directions)
      8) 경음화 (tensification) with three contexts: coda∈{k,t,p} | {n,m} | {l}
    Returns (surface_ipa_list, rule_tags)
    """
    sylls = _segment_syllables(ipa)
    rule_tags: List[str] = []

    # Rule 8 first (의) - local-intra-syllable
    _apply_eui_reading_basic(sylls)

    # iterate boundaries i|i+1
    for i in range(len(sylls)-1):
        onset1, rime1 = sylls[i]
        onset2, rime2 = sylls[i+1]
        if not rime1:
            continue

        # split rime1: nucleus + coda
        try:
            nuc_idx = next(j for j,ph in enumerate(rime1) if _is_vowel(ph))
        except StopIteration:
            nuc_idx = -1
        coda = rime1[nuc_idx+1:] if nuc_idx >= 0 else rime1[:]

        def _rewrite_current():
            if nuc_idx >= 0:
                sylls[i] = (onset1, rime1[:nuc_idx+1] + coda)
            else:
                sylls[i] = (onset1, coda)

        # ------- Rule 1: Liaison (연음화) -------
        # If next syllable starts with a vowel (onset empty & rime2 begins with vowel),
        # move the LAST consonant from coda to onset2.
        if (not onset2) and coda and (rime2 and _is_vowel(rime2[0])):
            moved = coda.pop()
            onset2.insert(0, moved)
            rule_tags.append(f"liaison:{i}->{i+1}")
            _rewrite_current()

        # At this point, refresh local boundary vars
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None

        # ------- Rule 9: ㄴ insertion (ㄴ 첨가) -------
        # If previous ends with any consonant (has coda), next onset empty, and next nucleus is i or glide j+{a,ʌ,o,u},
        # insert 'n' as onset.
        if coda and (not onset2) and rime2:
            # rime2[0] is vowel at syllable start
            # target cases: i, (j a), (j ʌ), (j o), (j u)
            first = rime2[0]
            second = rime2[1] if len(rime2) >= 2 else None
            need_insert_n = False
            if first == "i":
                need_insert_n = True
            elif first in GLIDES and (second in {"a","ʌ","o","u"}):
                need_insert_n = True
            if need_insert_n:
                onset2.insert(0, "n")
                rule_tags.append(f"n-insertion:{i+1}")
                # refresh onset_next for subsequent rules
                onset_next = onset2[0]

        # ------- Rule 3: Nasalization (비음화) -------
        # [k,t,p] + n/m => [ŋ,n,m] in coda (assimilation)
        if coda_last in {"k","ɡ"} and (onset_next in {"n","m"}):
            coda[-1] = "ŋ"; rule_tags.append(f"nasalization:k->ŋ@{i}")
            _rewrite_current()
        elif coda_last in {"t","d"} and (onset_next in {"n","m"}):
            coda[-1] = "n"; rule_tags.append(f"nasalization:t/d->n@{i}")
            _rewrite_current()
        elif coda_last in {"p","b"} and (onset_next in {"n","m"}):
            coda[-1] = "m"; rule_tags.append(f"nasalization:p/b->m@{i}")
            _rewrite_current()

        # [m, ŋ] + l  => onset l -> n
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None
        if coda_last in {"m","ŋ"} and onset_next == "l":
            onset2[0] = "n"; rule_tags.append(f"nasalization:m/ŋ + l -> n@{i+1}")

        # [k,b,p,ɡ] + l => onset l -> n
        if coda_last in {"k","ɡ","p","b"} and onset_next == "l":
            onset2[0] = "n"; rule_tags.append(f"nasalization:k/p + l -> n@{i+1}")

        # ------- Rule 4: Liquidization (유음비화) -------
        # n + l -> l l ; l + n -> l l
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None
        if coda_last == "n" and onset_next == "l":
            coda[-1] = "l"; onset2[0] = "l"
            rule_tags.append(f"liquidization:n+l->l l@{i}")
            _rewrite_current()
        elif coda_last == "l" and onset_next == "n":
            onset2[0] = "l"
            rule_tags.append(f"liquidization:l+n->l l@{i+1}")

        # ------- Rule 5: Palatalization (구개음화) -------
        # Onset t/d/s + (j or nucleus=i) -> t͡ɕ / d͡ʑ / ɕ
        onset_next = onset2[0] if onset2 else None
        if onset_next in {"t","d","s"}:
            # lookahead to see j or i
            nxt2 = onset2[1] if len(onset2) >= 2 else (rime2[1] if (rime2 and len(rime2)>=2) else None)
            if (len(rime2)>=1 and rime2[0]=="i") or (nxt2 in GLIDES) or (len(rime2)>=2 and rime2[0] in GLIDES and rime2[1]=="i"):
                if onset2[0] == "t": onset2[0] = "t͡ɕ"; rule_tags.append(f"palatalization:t->t͡ɕ@{i+1}")
                elif onset2[0] == "d": onset2[0] = "d͡ʑ"; rule_tags.append(f"palatalization:d->d͡ʑ@{i+1}")
                elif onset2[0] == "s": onset2[0] = "ɕ";   rule_tags.append(f"palatalization:s->ɕ@{i+1}")

        # ------- Rule 6: ㅎ deletion (ㅎ 탈락) -------
        # If coda h + next begins with vowel (onset empty), delete coda h.
        coda_last = coda[-1] if coda else None
        if coda_last in H_LIKE and (not onset2) and (rime2 and _is_vowel(rime2[0])):
            coda.pop()
            rule_tags.append(f"h-deletion@{i}")
            _rewrite_current()

        # ------- Rule 7: Aspiration (격음화 with ㅎ) -------
        # (a) coda in {k,t} + onset h => onset becomes aspirated: kʰ/tʰ
        onset_next = onset2[0] if onset2 else None
        if coda_last in {"k","t"} and onset_next in H_LIKE:
            # replace onset h by aspirated version of the coda place (use k/t)
            onset2[0] = ASP_MAP["k"] if coda_last == "k" else ASP_MAP["t"]
            rule_tags.append(f"aspiration:{coda_last}+h->{onset2[0]}@{i+1}")
            # keep coda as is (common simplification)

        # (b) coda h + onset in {k,t} => onset aspirated
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None
        if coda_last in H_LIKE and onset_next in {"k","t"}:
            onset2[0] = ASP_MAP[onset_next]
            rule_tags.append(f"aspiration:h+{onset_next}->{onset2[0]}@{i+1}")

        # ------- Rule 2: Tensification (경음화) -------
        # Cases:
        #  A) [k,t,p] + (k,t,p,s,t͡ɕ)  -> tense onset
        #  B) [n,m]   + (k,t,   t͡ɕ)   -> tense onset
        #  C) [l]     + (k,t, s,t͡ɕ)   -> tense onset
        coda_last = coda[-1] if coda else None
        onset_next = onset2[0] if onset2 else None

        def _tense_onset(target: str) -> Optional[str]:
            return TENSE_MAP.get(target)

        if onset_next in LAX_ONSET:
            # A
            if coda_last in {"k","t","p"}:
                t = _tense_onset(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t
                    rule_tags.append(f"tensification:A {coda_last}+{onset_next}->{t}@{i+1}")
            # B
            if coda_last in {"n","m"} and onset_next in {"k","t","t͡ɕ"}:
                t = _tense_onset(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t
                    rule_tags.append(f"tensification:B {coda_last}+{onset_next}->{t}@{i+1}")
            # C
            if coda_last == "l" and onset_next in {"k","t","s","t͡ɕ"}:
                t = _tense_onset(onset_next)
                if t and onset2[0] != t:
                    onset2[0] = t
                    rule_tags.append(f"tensification:C l+{onset_next}->{t}@{i+1}")

        # write back updated boundary syllables
        sylls[i+1] = (onset2, rime2)
        _rewrite_current()

    ipa_surface = _join_syllables(sylls)
    return ipa_surface, rule_tags
