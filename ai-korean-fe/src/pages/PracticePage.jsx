// src/pages/PracticePage.jsx
import React, { useState, useRef, useEffect, useMemo } from "react";
import axios from "axios";

const PracticePage = () => {
  const [level, setLevel] = useState("topik1");
  const [type, setType] = useState(null);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [questionContent, setQuestionContent] = useState(null);
  const [loadingQuestion, setLoadingQuestion] = useState(false);

  // single mode
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  // group mode
  const [selectedAnswers, setSelectedAnswers] = useState({});

  const [feedback, setFeedback] = useState(null);
  const backendUrl = import.meta.env.VITE_API_URL;

  // Audio (chỉ dùng cho nghe)
  const audioRef = useRef(null);
  const [audioLocked, setAudioLocked] = useState(false);
  const [sequenceActive, setSequenceActive] = useState(false);
  const [replaysRemaining, setReplaysRemaining] = useState(0);
  const replayTimerRef = useRef(null);

  // Countdown
  const [remainingSec, setRemainingSec] = useState(null);
  const [timeUp, setTimeUp] = useState(false);
  const [timeLimit, setTimeLimit] = useState(120);
  const timerRef = useRef(null);

  // Timer config (toggle + minutes)
  const [timerEnabled, setTimerEnabled] = useState(false); // mặc định OFF
  const [minutesPerItem, setMinutesPerItem] = useState(2); // phút cho mỗi câu

  // Auto-scroll target
  const answerSectionRef = useRef(null);

  // Dislike panel
  const [showDislikePanel, setShowDislikePanel] = useState(false);
  const [reasonTags, setReasonTags] = useState([]);
  const [freeText, setFreeText] = useState("");

  const TRACK_URL = `${backendUrl}/track/event`;
  const [sessionId, setSessionId] = useState("");
  const [userIdHash, setUserIdHash] = useState("");

  const startTimeRef = useRef(null);
  const audioPlayCountRef = useRef(0);

  useEffect(() => {
    let uid = localStorage.getItem("uid_hash");
    if (!uid) {
      uid = crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
      localStorage.setItem("uid_hash", uid);
    }
    setUserIdHash(uid);

    const day = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
    setSessionId(`S-${day}-${rand}`);
  }, []);

  async function postTrackEvent(payload) {
    try {
      const event_id =
        crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
      await fetch(TRACK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_id, ...payload }),
        keepalive: true,
      });
    } catch {}
  }

  const REASON_OPTIONS = [
    { code: "audio_bad", label: "Âm thanh kém/khó nghe" },
    { code: "off_topic", label: "Lệch chủ đề" },
    { code: "too_easy", label: "Quá dễ" },
    { code: "too_hard", label: "Quá khó" },
    { code: "typo", label: "Lỗi chính tả" },
    { code: "ambiguous", label: "Mơ hồ/khó hiểu" },
  ];

  const toggleReason = (code) => {
    setReasonTags((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
    );
  };

  // ---------- Helpers ----------
  const isGroup =
    Array.isArray(questionContent?.items) && questionContent.items.length > 0;

  const groupQnos = useMemo(() => {
    if (Array.isArray(questionContent?.question_no)) {
      return new Set(questionContent.question_no);
    }
    return new Set();
  }, [questionContent]);

  // Số câu trong bài (1 nếu single)
  const itemsCount = useMemo(() => {
    return Array.isArray(questionContent?.items)
      ? questionContent.items.length
      : 1;
  }, [questionContent]);

  // Tổng phút hiển thị/nhập (đồng bộ theo itemsCount)
  const totalMinutes = useMemo(
    () => Number((minutesPerItem * itemsCount).toFixed(2)),
    [minutesPerItem, itemsCount]
  );

  // Định dạng mm:ss (nếu s == null → dùng timeLimit hiện hành)
  const formatTime = (s) => {
    const total = s == null ? timeLimit : s;
    const mm = String(Math.floor(total / 60)).padStart(2, "0");
    const ss = String(total % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  // ---- Data URI helpers ----
  const isDataUri = (s) => typeof s === "string" && s.startsWith("data:");
  const isImageDataUri = (s) => isDataUri(s) && /^data:image\//i.test(s);

  // Cho phép URL ảnh thường / đường dẫn tĩnh
  // Cho phép URL ảnh thường / đường dẫn tĩnh
  const isImageLike = (s) => {
    if (typeof s !== "string") return false;
    if (isImageDataUri(s)) return true;

    // http/https, //cdn, /absolute, ./relative
    if (
      /^(?:https?:\/\/|\/\/|\/|\.{1,2}\/)[^?\s]+\.(png|jpe?g|webp|gif|svg)(\?.*)?$/i.test(
        s
      )
    ) {
      return true;
    }
    // thư mục tương đối phổ biến: images/, img/, icons/
    if (
      /^(images|img|icons)\/[^?\s]+\.(png|jpe?g|webp|gif|svg)(\?.*)?$/i.test(s)
    ) {
      return true;
    }
    return false;
  };

  // ---- Grid helpers ----
  const isMostlyImageOptions = (choices) => {
    if (!choices || typeof choices !== "object") return false;
    const vals = Object.values(choices).filter((v) => v != null);
    if (!vals.length) return false;
    const imgCount = vals.filter((v) => isImageLike(v)).length;
    return imgCount / vals.length >= 0.75; // ≥ 75% là ảnh
  };

  const grid2x2Styles = {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: "0.75rem",
  };

  const tileButtonStyle = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    minHeight: 160, // 160–240 tuỳ ý
    width: "100%",
    whiteSpace: "normal",
  };

  // Render passage: text hoặc ảnh; nếu là mảng -> thụt đầu dòng từng đoạn
  const renderPassage = (content) => {
    if (!content) return null;

    // component nhỏ để thụt đầu dòng
    const Para = ({ children }) => (
      <p style={{ textIndent: "1.5rem", marginBottom: "0.5rem" }}>{children}</p>
    );

    // Ảnh đơn
    if (isImageLike(content)) {
      return (
        <figure className="mb-0 text-center">
          <img
            src={content}
            alt="passage"
            loading="lazy"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </figure>
      );
    }

    // MẢNG: mỗi phần tử là 1 đoạn (string) hoặc ảnh
    if (Array.isArray(content)) {
      return (
        <div>
          {content.map((part, idx) => {
            // ảnh
            if (isImageLike(part)) {
              return (
                <figure key={idx} className="mb-2 text-center">
                  <img
                    src={part}
                    alt={`passage-${idx}`}
                    loading="lazy"
                    style={{ maxWidth: "100%", height: "auto" }}
                  />
                </figure>
              );
            }
            // object có .text
            if (
              part &&
              typeof part === "object" &&
              typeof part.text === "string"
            ) {
              return (
                <Para key={idx}>{renderWithUnderlineMarkers(part.text)}</Para>
              );
            }
            // string thường
            if (typeof part === "string") {
              return <Para key={idx}>{renderWithUnderlineMarkers(part)}</Para>;
            }
            return null;
          })}
        </div>
      );
    }

    // STRING: nếu có xuống dòng -> tách đoạn để mỗi đoạn thụt đầu dòng
    if (typeof content === "string") {
      const lines = content.split(/\r?\n/).filter(Boolean);
      if (lines.length > 1) {
        return (
          <div>
            {lines.map((ln, i) => (
              <Para key={i}>{renderWithUnderlineMarkers(ln)}</Para>
            ))}
          </div>
        );
      }
      return <Para>{renderWithUnderlineMarkers(content)}</Para>;
    }

    return null;
  };

  // Render option: value có thể là text hoặc ảnh
  const renderOptionContent = (
    value,
    { compactCaption } = { compactCaption: false }
  ) => {
    if (isImageLike(value)) {
      return (
        <div className="w-100 d-flex flex-column align-items-center">
          <img
            src={value}
            alt="option"
            loading="lazy"
            style={{
              maxWidth: "100%",
              height: "auto",
              maxHeight: 320,
              objectFit: "contain",
            }}
          />
          {!compactCaption && (
            <div className="small text-muted mt-1">(ảnh phương án)</div>
          )}
        </div>
      );
    }
    return <span>{value}</span>;
  };

  // Timer helpers
  const startCountdown = (seconds) => {
    if (timerRef.current || remainingSec !== null) return;
    setTimeLimit(seconds);
    setRemainingSec(seconds);
    timerRef.current = setInterval(() => {
      setRemainingSec((prev) => {
        if (prev === null) return prev;
        if (prev <= 1) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          setTimeUp(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const resetTimer = () => {
    stopTimer();
    setRemainingSec(null);
    setTimeUp(false);
    setTimeLimit(120);
  };

  const resetAllPlayback = () => {
    setAudioLocked(false);
    setSequenceActive(false);
    setReplaysRemaining(0);
    if (replayTimerRef.current) {
      clearTimeout(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const resetFeedbackUI = () => {
    setFeedback(null);
    setShowDislikePanel(false);
    setReasonTags([]);
    setFreeText("");
  };

  // Reset khi đổi câu hỏi hoặc audio
  useEffect(() => {
    resetAllPlayback();
    resetTimer();
    resetFeedbackUI();
    setSelectedAnswer(null);
    setSelectedAnswers({});
    return () => {
      if (replayTimerRef.current) {
        clearTimeout(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [questionContent?.question_audio, selectedQuestion]);

  // Hết giờ → dọn playback
  useEffect(() => {
    if (timeUp) {
      if (audioRef.current) audioRef.current.pause();
      if (replayTimerRef.current) {
        clearTimeout(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      setSequenceActive(false);
      setReplaysRemaining(0);
      setAudioLocked(false);
    }
  }, [timeUp]);

  // Auto-scroll khi hết giờ hoặc trả lời xong
  const allItemsAnswered = useMemo(() => {
    if (!isGroup) return !!selectedAnswer;
    const expected = new Set(questionContent?.question_no || []);
    for (const qno of expected) {
      if (!selectedAnswers[qno]) return false;
    }
    return expected.size > 0;
  }, [isGroup, selectedAnswer, selectedAnswers, questionContent]);

  useEffect(() => {
    if ((timeUp || allItemsAnswered) && answerSectionRef.current) {
      answerSectionRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [timeUp, allItemsAnswered]);

  // ==== NGHE: bắt đầu timer khi audio được phát lần đầu (nếu bật timer) ====
  const handleAudioPlay = () => {
    if (!startTimeRef.current) startTimeRef.current = performance.now();
    audioPlayCountRef.current += 1;

    if (allItemsAnswered || timeUp) return;

    if (!sequenceActive && !audioLocked) {
      setAudioLocked(true);
      setSequenceActive(true);
      setReplaysRemaining(1);
      if (timerEnabled) {
        const secs = Math.max(
          5,
          Math.round((minutesPerItem || 2) * itemsCount * 60)
        );
        startCountdown(secs);
      }
    }
  };

  const handleAudioEnded = () => {
    if (sequenceActive && replaysRemaining > 0 && audioRef.current) {
      replayTimerRef.current = setTimeout(() => {
        if (!audioRef.current) return;
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
        setReplaysRemaining((n) => Math.max(0, n - 1));
      }, 5000);
    } else {
      setSequenceActive(false);
      setReplaysRemaining(0);
    }
  };

  // ---------- Question list ----------
  const getQuestions = () => {
    let start = 1;
    let count = 0;

    if (level === "topik1") {
      if (type === "listening") {
        start = 1;
        count = 30;
      } else if (type === "reading") {
        start = 31;
        count = 40; // 31..70 (40 câu)
      }
    } else if (level === "topik2") {
      if (type === "listening" || type === "reading") {
        start = 1;
        count = 50;
      }
    }

    return Array.from({ length: count }, (_, i) => start + i);
  };

  const questions = type ? getQuestions() : [];

  // ---------- API ----------
  const fetchQuestion = async (q) => {
    setLoadingQuestion(true);
    setSelectedQuestion(q);
    setSelectedAnswer(null);
    setSelectedAnswers({});
    resetFeedbackUI();
    resetAllPlayback();
    resetTimer();

    startTimeRef.current = null;
    audioPlayCountRef.current = 0;

    const categoryParam = type === "listening" ? "listening" : "reading";

    try {
      const res = await axios.post(`${backendUrl}/generate_question`, {
        level: level,
        category: categoryParam,
        cau: parseInt(q.toString(), 10),
      });
      setQuestionContent(res.data);
    } catch (err) {
      console.error("Error fetching question:", err);
      setQuestionContent({ error: "Không thể lấy câu hỏi" });
    } finally {
      setLoadingQuestion(false);
    }
  };

  // ==== ĐỌC: chỉ bật timer NGAY khi dữ liệu đọc đã sẵn sàng & timer bật ====
  useEffect(() => {
    if (type !== "reading") return;
    if (!questionContent || questionContent.error) return;
    if (remainingSec !== null || timeUp) return; // đã chạy rồi
    if (!timerEnabled) return;

    // bắt đầu đo thời gian phản hồi + countdown
    startTimeRef.current = performance.now();
    const secs = Math.max(
      5,
      Math.round((minutesPerItem || 2) * itemsCount * 60)
    );
    startCountdown(secs);
  }, [
    type,
    questionContent,
    remainingSec,
    timeUp,
    timerEnabled,
    minutesPerItem,
    itemsCount,
  ]);

  const buildFeedbackPayload = (reaction, extra = {}) => {
    const isGroupLocal =
      Array.isArray(questionContent?.items) && questionContent.items.length > 0;

    const snapshot = {
      id: questionContent?.id ?? null,
      title: questionContent?.title ?? null,
      question: questionContent?.question ?? null,
      dialog: questionContent?.dialog ?? [],
      passage: questionContent?.passage ?? null,
      choices: questionContent?.choices ?? {},
      items: questionContent?.items ?? null,
      answer: questionContent?.answer ?? null,
      score: questionContent?.score ?? null,
      question_no: questionContent?.question_no ?? null,
      level,
      type,
      cau: selectedQuestion,
    };

    const is_correct_single =
      !isGroupLocal && selectedAnswer != null && questionContent?.answer != null
        ? selectedAnswer === questionContent.answer
        : null;

    return {
      level,
      type,
      cau: selectedQuestion,
      questionId: questionContent?.id ?? null,
      reaction,
      reason_tags: extra.reason_tags ?? null,
      free_text: extra.free_text ?? null,
      answer_selected: isGroupLocal ? selectedAnswers : selectedAnswer ?? null,
      is_correct: is_correct_single,
      timer_seconds_left: remainingSec,
      time_limit: timeLimit,
      question_snapshot: snapshot,
    };
  };

  const sendFeedbackLike = async () => {
    try {
      setFeedback("like");
      setShowDislikePanel(false);
      const payload = buildFeedbackPayload("like");
      await axios.post(`${backendUrl}/feedback`, payload);
    } catch (e) {
      console.error("Feedback error:", e);
    }
  };

  const submitDislike = async () => {
    try {
      setFeedback("dislike");
      const payload = buildFeedbackPayload("dislike", {
        reason_tags: reasonTags.length ? reasonTags : null,
        free_text: freeText?.trim() ? freeText.trim() : null,
      });
      await axios.post(`${backendUrl}/feedback`, payload);
      setShowDislikePanel(false);
    } catch (e) {
      console.error("Feedback error:", e);
    }
  };

  const disableChoicesSingle = !!selectedAnswer || timeUp;
  const choiceVariantSingle = (key) => {
    const isCorrect = questionContent?.answer === key;

    if (selectedAnswer) {
      const isSelected = selectedAnswer === key;
      return isSelected
        ? isCorrect
          ? "btn-success"
          : "btn-danger"
        : isCorrect
        ? "btn-success"
        : "btn-outline-secondary";
    }

    if (timeUp) {
      return isCorrect ? "btn-success" : "btn-outline-secondary";
    }

    return "btn-outline-secondary";
  };

  const disableChoicesFor = (qno) => !!selectedAnswers[qno] || timeUp;
  const choiceVariantFor = (qno, key, answerKey) => {
    const isCorrect = answerKey === key;
    const sel = selectedAnswers[qno];

    if (sel) {
      const isSelected = sel === key;
      return isSelected
        ? isCorrect
          ? "btn-success"
          : "btn-danger"
        : isCorrect
        ? "btn-success"
        : "btn-outline-secondary";
    }

    if (timeUp) {
      return isCorrect ? "btn-success" : "btn-outline-secondary";
    }
    return "btn-outline-secondary";
  };

  const onChooseAnswer = (key) => {
    if (disableChoicesSingle) return;
    setSelectedAnswer(key);
    stopTimer();
    if (replayTimerRef.current) {
      clearTimeout(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    setSequenceActive(false);
    setReplaysRemaining(0);
    setAudioLocked(false);
    if (audioRef.current) {
      audioRef.current.pause();
    }

    const t0 = startTimeRef.current;
    const responseTime =
      typeof t0 === "number"
        ? Math.max(0, Math.round(performance.now() - t0))
        : null;

    const is_correct =
      questionContent?.answer != null ? key === questionContent.answer : null;

    const item_id =
      questionContent?.id ??
      `${level}-${type}-Q${String(selectedQuestion).padStart(3, "0")}`;

    postTrackEvent({
      user_id_hash: userIdHash,
      session_id: sessionId,
      event_type: "topik_answered",
      item_id,
      is_correct,
      duration_ms: responseTime,
      meta: {
        level,
        type,
        cau: selectedQuestion,
        audio_play_count: audioPlayCountRef.current || 0,
        time_limit: timeLimit,
        question_no: selectedQuestion,
      },
    });
  };

  const onChooseAnswerGroup = (qno, key, answerKey) => {
    if (disableChoicesFor(qno)) return;
    setSelectedAnswers((prev) => ({ ...prev, [qno]: key }));

    const expected = new Set(questionContent?.question_no || []);
    const next = { ...selectedAnswers, [qno]: key };
    const done = Array.from(expected).every((n) => !!next[n]);

    if (done) {
      stopTimer();
      if (replayTimerRef.current) {
        clearTimeout(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      setSequenceActive(false);
      setReplaysRemaining(0);
      setAudioLocked(false);
      if (audioRef.current) {
        audioRef.current.pause();
      }
    }

    const t0 = startTimeRef.current;
    const responseTime =
      typeof t0 === "number"
        ? Math.max(0, Math.round(performance.now() - t0))
        : null;

    const is_correct = answerKey ? key === answerKey : null;

    const item_id = `${level}-${type}-Q${String(qno).padStart(3, "0")}`;
    postTrackEvent({
      user_id_hash: userIdHash,
      session_id: sessionId,
      event_type: "topik_answered",
      item_id,
      is_correct,
      duration_ms: responseTime,
      meta: {
        level,
        type,
        cau: selectedQuestion,
        question_no: qno,
        audio_play_count: audioPlayCountRef.current || 0,
        time_limit: timeLimit,
      },
    });
  };

  // Chuyển "abc __gạch chân__ xyz" -> React nodes với <u>gạch chân</u>
  const renderWithUnderlineMarkers = (input) => {
    if (typeof input !== "string" || !input.includes("__")) return input;

    const nodes = [];
    const regex = /__([\s\S]+?)__/g; // khớp ngắn nhất giữa hai cặp __ __
    let last = 0;
    let m;
    let i = 0;

    while ((m = regex.exec(input)) !== null) {
      if (m.index > last) {
        nodes.push(
          <React.Fragment key={`txt-${i++}`}>
            {input.slice(last, m.index)}
          </React.Fragment>
        );
      }
      nodes.push(<u key={`u-${i++}`}>{m[1]}</u>);
      last = regex.lastIndex;
    }
    if (last < input.length) {
      nodes.push(
        <React.Fragment key={`txt-${i++}`}>{input.slice(last)}</React.Fragment>
      );
    }
    return nodes;
  };

  // ==== EXPLAIN: state & endpoint ====
  const EXPLAIN_URL = `${backendUrl}/explain`;

  const [showExplain, setShowExplain] = useState(false);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState(null);
  // single: object { explanation, is_correct, explain_model, updated, qid, points[] }
  // group:  array of { qno, explanation, is_correct, explain_model, updated, qid, points[] }
  const [explainData, setExplainData] = useState(null);

  // Ngôn ngữ lời giải (nếu muốn đổi về sau)
  const [explainLang, setExplainLang] = useState("vi");

  // ==== EXPLAIN: build payload(s) ====
  const buildExplainPayloadSingle = () => {
    if (!questionContent) return null;
    return {
      level,
      category: type === "listening" ? "listening" : "reading",
      cau: selectedQuestion,
      type: questionContent?.type || (type === "listening" ? "Listen" : "Read"),
      section:
        questionContent?.section || (type === "listening" ? "Nghe" : "Đọc"),
      title: questionContent?.title || null,
      question: questionContent?.question || null,
      dialogue: Array.isArray(questionContent?.dialog)
        ? questionContent.dialog
        : null,
      passage: questionContent?.passage ?? null,
      options: questionContent?.choices || {},
      answer: questionContent?.answer ?? null,
      user_answer: selectedAnswer ?? null,
      language: explainLang,
      use_sidecar: true,
      source_jsonl: questionContent?.source_jsonl || null,
    };
  };

  const buildExplainPayloadGroup = () => {
    if (!questionContent?.items) return [];
    const sharedPassage =
      type === "reading" ? questionContent?.passage ?? null : null;
    const sharedDialogue =
      type === "listening" && Array.isArray(questionContent?.dialog)
        ? questionContent.dialog
        : null;

    if (!questionContent?.items) return [];
    return questionContent.items.map((it) => ({
      level,
      category: type === "listening" ? "listening" : "reading",
      cau: it.question_no ?? selectedQuestion, // “cau” hiển thị theo từng câu con
      type: it.type || questionContent?.type || "ReadGroup",
      section:
        questionContent?.section || (type === "listening" ? "Nghe" : "Đọc"),
      title: questionContent?.title || null,
      question: it.question || null,
      dialogue: sharedDialogue,
      passage: sharedPassage,
      options: it.choices || {},
      answer: it.answer ?? null,
      user_answer: selectedAnswers?.[it.question_no] ?? null,
      language: explainLang,
      use_sidecar: true,
      source_jsonl: questionContent?.source_jsonl || null,
    }));
  };

  // ==== EXPLAIN: caller ====
  const callExplain = async (mode = "auto") => {
    try {
      setExplainLoading(true);
      setExplainError(null);
      debugger;
      if (!isGroup) {
        const payload = buildExplainPayloadSingle();
        if (!payload) throw new Error("Thiếu dữ liệu câu hỏi.");
        const res = await axios.post(`${EXPLAIN_URL}?mode=${mode}`, payload);
        const d = res.data;
        setExplainData({
          explanation: d.explanation,
          points: Array.isArray(d.points) ? d.points : [],
          is_correct: d.is_correct,
          explain_model: d.explain_model,
          updated: d.updated,
          qid: d.qid,
          updated_path: d.updated_path || null,
        });
      } else {
        const payloads = buildExplainPayloadGroup();
        if (!payloads.length) throw new Error("Thiếu dữ liệu câu hỏi nhóm.");
        const results = await Promise.all(
          payloads.map((p) =>
            axios
              .post(`${EXPLAIN_URL}?mode=${mode}`, p)
              .then((r) => r.data)
              .catch((e) => ({ error: e?.message || "ERR" }))
          )
        );
        const merged = results.map((d, idx) => ({
          qno: questionContent.items[idx]?.question_no ?? idx + 1,
          explanation: d?.explanation || "(Không có giải thích.)",
          points: Array.isArray(d?.points) ? d.points : [],
          is_correct: d?.is_correct ?? null,
          explain_model: d?.explain_model || null,
          updated: !!d?.updated,
          qid: d?.qid || null,
          error: d?.error || null,
        }));
        setExplainData(merged);
      }

      setShowExplain(true);
    } catch (err) {
      console.error("Explain error:", err);
      setExplainError(err?.message || "Không thể lấy lời giải.");
      setShowExplain(true); // vẫn mở modal để hiện lỗi
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div className="container py-5">
      <h1 className="text-center mb-4">Practice TOPIK</h1>

      {/* Tabs TOPIK */}
      <div className="d-flex justify-content-center mb-4 gap-2">
        {["topik1", "topik2"].map((lvl) => (
          <button
            key={lvl}
            className={`btn btn-lg ${
              level === lvl ? "btn-primary shadow" : "btn-outline-secondary"
            }`}
            onClick={() => {
              setLevel(lvl);
              setType(null);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
              setSelectedAnswers({});
              resetFeedbackUI();
              resetAllPlayback();
              resetTimer();
            }}
          >
            {lvl.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Chọn dạng đề */}
      <div className="d-flex justify-content-center mb-4 gap-2">
        {["reading", "listening"].map((t) => (
          <button
            key={t}
            className={`btn btn-md ${
              type === t ? "btn-success shadow" : "btn-outline-secondary"
            }`}
            onClick={() => {
              setType(t);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
              setSelectedAnswers({});
              resetFeedbackUI();
              resetAllPlayback();
              resetTimer();
            }}
          >
            {t === "reading" ? "Read" : "Listen"}
          </button>
        ))}
      </div>

      {/* Chọn câu hỏi */}
      {type && questions.length > 0 && (
        <div className="mb-4">
          <h5 className="text-center mb-2">
            Chọn câu hỏi ({questions.length} câu)
          </h5>
          <div
            className="d-flex flex-wrap justify-content-center gap-2 border rounded p-2 bg-light"
            style={{ maxHeight: "400px", overflowY: "auto" }}
          >
            {questions.map((q) => {
              const disabledByGroup = groupQnos.has(q);
              return (
                <button
                  key={q}
                  className={`btn btn-sm ${
                    selectedQuestion === q
                      ? "btn-primary"
                      : "btn-outline-secondary"
                  }`}
                  style={{
                    width: "40px",
                    height: "40px",
                    position: "relative",
                  }}
                  disabled={loadingQuestion || disabledByGroup}
                  onClick={() => fetchQuestion(q)}
                  title={disabledByGroup ? "Đang hiển thị theo cặp" : undefined}
                >
                  {q}
                  {loadingQuestion && selectedQuestion === q && (
                    <span
                      className="spinner-border spinner-border-sm position-absolute top-50 start-50 translate-middle"
                      role="status"
                      aria-hidden="true"
                    ></span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Hiển thị thông báo câu đang phát triển */}
      {questionContent?.message ? (
        <div
          className="card shadow p-4 mt-4 mx-auto text-center"
          style={{ maxWidth: "600px" }}
        >
          <h5 className="card-title mb-3">{questionContent.title}</h5>
          <p className="text-warning fst-italic">{questionContent.message}</p>
        </div>
      ) : (
        selectedQuestion &&
        questionContent &&
        !questionContent.error && (
          <div
            className="card shadow p-4 mt-4 mx-auto"
            style={{ maxWidth: "600px" }}
          >
            {/* Title row with countdown at right */}
            <div className="d-flex align-items-center justify-content-center position-relative mb-2">
              <h5 className="card-title text-center m-0">
                {level.toUpperCase()} -{" "}
                {type === "reading" ? "reading" : "listening"} - Câu{" "}
                {selectedQuestion}
              </h5>
              {/* Hiển thị countdown khi đang chạy */}
              {remainingSec !== null && (
                <span
                  className="badge bg-dark position-absolute end-0"
                  title="Thời gian còn lại"
                  style={{ fontSize: "0.9rem" }}
                >
                  {formatTime(remainingSec)}
                </span>
              )}
            </div>

            {/* Row điều khiển Timer */}
            <div className="d-flex align-items-center justify-content-between mb-3">
              <div className="d-flex align-items-center gap-2">
                <button
                  type="button"
                  className={`btn btn-sm ${
                    timerEnabled ? "btn-warning" : "btn-outline-secondary"
                  }`}
                  onClick={() => {
                    const next = !timerEnabled;
                    setTimerEnabled(next);
                    if (!next) {
                      // Tắt → dừng & reset
                      resetTimer();
                    } else {
                      // Bật → nếu đã có câu hỏi & timer chưa chạy, khởi động ngay
                      if (!timeUp && remainingSec === null) {
                        const secs = Math.max(
                          5,
                          Math.round((minutesPerItem || 2) * itemsCount * 60)
                        );
                        startTimeRef.current = performance.now();
                        startCountdown(secs);
                      }
                    }
                  }}
                  title={timerEnabled ? "Tắt đồng hồ" : "Bật đồng hồ"}
                >
                  {timerEnabled ? "Timer: ON" : "Timer: OFF"}
                </button>

                <div
                  className="input-group input-group-sm"
                  style={{ width: 190 }}
                >
                  <span className="input-group-text">Thời gian</span>
                  <input
                    type="number"
                    min={0.5}
                    step={0.5}
                    className="form-control"
                    value={totalMinutes}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      const safe = Number.isFinite(v) ? v : 2;
                      // cập nhật minutesPerItem theo tổng/số câu
                      const divisor = itemsCount > 0 ? itemsCount : 1;
                      setMinutesPerItem(Math.max(0.1, safe / divisor));
                    }}
                    disabled={remainingSec !== null} // đang chạy thì khoá input
                  />
                  <span className="input-group-text">ph (tổng)</span>
                </div>
              </div>

              <button
                className="btn btn-sm btn-outline-primary"
                onClick={() => {
                  if (timerEnabled) {
                    stopTimer();
                    const secs = Math.max(
                      5,
                      Math.round((minutesPerItem || 2) * itemsCount * 60)
                    );
                    setTimeLimit(secs);
                    setRemainingSec(null);
                    setTimeUp(false);
                  }
                }}
              >
                Đặt lại
              </button>
            </div>

            {/* Gợi ý nhỏ hiển thị phút mỗi câu */}
            <div className="text-muted mb-2" style={{ fontSize: "0.85rem" }}>
              {itemsCount > 1
                ? `~ ${minutesPerItem.toFixed(
                    2
                  )} phút/câu × ${itemsCount} câu = ${totalMinutes.toFixed(
                    2
                  )} phút`
                : `~ ${minutesPerItem.toFixed(2)} phút`}
            </div>

            {/* Title + feedback */}
            <div className="d-flex align-items-start justify-content-between">
              <p className="fw-bold mb-2 me-3">{questionContent.title}</p>
              <div className="position-relative">
                <div className="btn-group" role="group" aria-label="Feedback">
                  <button
                    type="button"
                    className={`btn btn-sm ${
                      feedback === "like"
                        ? "btn-success"
                        : "btn-outline-success"
                    }`}
                    onClick={sendFeedbackLike}
                    disabled={!selectedQuestion}
                    title="Hữu ích"
                  >
                    👍
                  </button>
                  <button
                    type="button"
                    className={`btn btn-sm ${
                      feedback === "dislike"
                        ? "btn-danger"
                        : "btn-outline-danger"
                    }`}
                    onClick={() => setShowDislikePanel((v) => !v)}
                    disabled={!selectedQuestion}
                    title="Chưa ổn"
                  >
                    👎
                  </button>
                </div>

                {showDislikePanel && (
                  <div
                    className="card shadow-sm p-2 mt-2"
                    style={{
                      position: "absolute",
                      right: 0,
                      zIndex: 20,
                      width: "280px",
                    }}
                  >
                    <div className="small fw-semibold mb-2">
                      Vì sao bạn không thích?
                    </div>
                    <div className="d-flex flex-column gap-1 mb-2">
                      {REASON_OPTIONS.map((opt) => (
                        <label
                          key={opt.code}
                          className="form-check d-flex align-items-center gap-2"
                        >
                          <input
                            type="checkbox"
                            className="form-check-input"
                            checked={reasonTags.includes(opt.code)}
                            onChange={() => toggleReason(opt.code)}
                          />
                          <span className="form-check-label">{opt.label}</span>
                        </label>
                      ))}
                    </div>
                    <div className="mb-2">
                      <textarea
                        className="form-control form-control-sm"
                        rows={2}
                        placeholder="Phản ánh thêm (tuỳ chọn)"
                        value={freeText}
                        onChange={(e) => setFreeText(e.target.value)}
                      />
                    </div>
                    <div className="d-flex justify-content-end gap-2">
                      <button
                        className="btn btn-sm btn-light"
                        onClick={() => setShowDislikePanel(false)}
                      >
                        Đóng
                      </button>
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={submitDislike}
                      >
                        Gửi
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Nội dung câu hỏi */}
            {!isGroup ? (
              <p>
                {questionContent?.cau ? `${questionContent.cau}. ` : ""}
                {questionContent?.question ?? ""}
              </p>
            ) : (
              <div className="mb-2">
                <p className="text-muted mb-1">
                  Bài gồm{" "}
                  {Array.isArray(questionContent?.items)
                    ? questionContent.items.length
                    : 0}{" "}
                  câu:{" "}
                  {Array.isArray(questionContent?.question_no)
                    ? questionContent.question_no.join(", ")
                    : ""}
                </p>
              </div>
            )}

            {/* PASSAGE (ĐỌC) */}
            {type === "reading" && questionContent?.passage && (
              <div className="mb-3 p-3 bg-light border rounded">
                {renderPassage(questionContent.passage)}
              </div>
            )}

            {/* Audio (NGHE) */}
            {type === "listening" && questionContent?.question_audio && (
              <audio
                ref={audioRef}
                controls={!audioLocked || timeUp || allItemsAnswered}
                className="my-2 w-100"
                src={questionContent.question_audio}
                onPlay={handleAudioPlay}
                onEnded={handleAudioEnded}
                preload="auto"
              />
            )}

            {/* Phần đáp án / dialog */}
            <div ref={answerSectionRef}>
              {/* Dialog chỉ hiện với nghe khi xong/hết giờ */}
              {type === "listening" &&
                (allItemsAnswered || timeUp) &&
                Array.isArray(questionContent?.dialog) &&
                questionContent.dialog.length > 0 && (
                  <div className="mb-2 p-2 bg-light border rounded">
                    {questionContent.dialog.map((d, idx) => (
                      <div key={idx} className="mb-1">
                        <strong className="me-1">{d.speaker}:</strong>
                        {isImageLike(d.text) ? (
                          <img
                            src={d.text}
                            alt={`${d.speaker}-utterance`}
                            loading="lazy"
                            style={{
                              maxWidth: "100%",
                              height: "auto",
                              verticalAlign: "middle",
                            }}
                          />
                        ) : (
                          <span>{d.text}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

              {/* Choices (single) */}
              {!isGroup &&
                selectedQuestion &&
                questionContent?.choices &&
                (() => {
                  const isGrid = isMostlyImageOptions(questionContent.choices);

                  if (!isGrid) {
                    // Dọc
                    return (
                      <div className="d-flex flex-column gap-2 mt-3">
                        {Object.entries(questionContent.choices).map(
                          ([key, value]) => (
                            <button
                              key={key}
                              className={`btn text-start ${choiceVariantSingle(
                                key
                              )}`}
                              onClick={() => onChooseAnswer(key)}
                              disabled={disableChoicesSingle}
                              style={{
                                display: "flex",
                                gap: "0.75rem",
                                alignItems: "center",
                              }}
                            >
                              <strong style={{ minWidth: 20 }}>{key}.</strong>
                              <div className="flex-grow-1">
                                {renderOptionContent(value, {
                                  compactCaption: true,
                                })}
                              </div>
                            </button>
                          )
                        )}
                      </div>
                    );
                  }

                  // Lưới 2×2
                  return (
                    <div className="mt-3" style={grid2x2Styles}>
                      {Object.entries(questionContent.choices).map(
                        ([key, value]) => (
                          <button
                            key={key}
                            className={`btn ${choiceVariantSingle(key)}`}
                            onClick={() => onChooseAnswer(key)}
                            disabled={disableChoicesSingle}
                            aria-label={`Option ${key}`}
                            style={tileButtonStyle}
                          >
                            <div className="mb-2 fw-semibold">{key}</div>
                            <div style={{ width: "100%" }}>
                              {renderOptionContent(value, {
                                compactCaption: true,
                              })}
                            </div>
                          </button>
                        )
                      )}
                    </div>
                  );
                })()}

              {/* Group items */}
              {isGroup && (
                <div className="d-flex flex-column gap-3 mt-2">
                  {questionContent.items.map((it, idx) => {
                    const qno = it.question_no;
                    const isGrid = isMostlyImageOptions(it.choices || {});
                    return (
                      <div key={qno ?? idx} className="p-2 border rounded">
                        <div className="fw-semibold mb-2">
                          {qno ? `Câu ${qno}. ` : ""}
                          {it.question}
                        </div>

                        {!isGrid ? (
                          // Dọc
                          <div className="d-flex flex-column gap-2">
                            {Object.entries(it.choices || {}).map(
                              ([key, value]) => (
                                <button
                                  key={key}
                                  className={`btn text-start ${choiceVariantFor(
                                    qno,
                                    key,
                                    it.answer
                                  )}`}
                                  onClick={() =>
                                    onChooseAnswerGroup(qno, key, it.answer)
                                  }
                                  disabled={disableChoicesFor(qno)}
                                  style={{
                                    display: "flex",
                                    gap: "0.75rem",
                                    alignItems: "center",
                                  }}
                                >
                                  <strong style={{ minWidth: 20 }}>
                                    {key}.
                                  </strong>
                                  <div className="flex-grow-1">
                                    {renderOptionContent(value, {
                                      compactCaption: true,
                                    })}
                                  </div>
                                </button>
                              )
                            )}
                          </div>
                        ) : (
                          // Lưới 2×2
                          <div style={grid2x2Styles}>
                            {Object.entries(it.choices || {}).map(
                              ([key, value]) => (
                                <button
                                  key={key}
                                  className={`btn ${choiceVariantFor(
                                    qno,
                                    key,
                                    it.answer
                                  )}`}
                                  onClick={() =>
                                    onChooseAnswerGroup(qno, key, it.answer)
                                  }
                                  disabled={disableChoicesFor(qno)}
                                  aria-label={`Option ${key}`}
                                  style={tileButtonStyle}
                                >
                                  <div className="mb-2 fw-semibold">{key}</div>
                                  <div style={{ width: "100%" }}>
                                    {renderOptionContent(value, {
                                      compactCaption: true,
                                    })}
                                  </div>
                                </button>
                              )
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {(allItemsAnswered || timeUp) && (
              <div className="d-flex justify-content-center gap-2 mt-3">
                <button
                  className="btn btn-outline-secondary"
                  onClick={() => callExplain("auto")}
                  disabled={explainLoading}
                  title="Xem giải thích"
                >
                  {explainLoading ? (
                    <span className="spinner-border spinner-border-sm me-2" />
                  ) : (
                    <span className="me-2" aria-hidden="true">
                      💡
                    </span>
                  )}
                  Giải thích
                </button>

                <button
                  className="btn btn-outline-primary"
                  onClick={() => fetchQuestion(selectedQuestion)}
                >
                  <span className="loop-icon me-2" aria-hidden="true">
                    ↻
                  </span>
                  Thử lại câu hỏi
                </button>
              </div>
            )}
          </div>
        )
      )}

      {loadingQuestion && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center"
          style={{ background: "rgba(0,0,0,0.2)", zIndex: 1000 }}
        >
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      )}

      {/* ==== EXPLAIN MODAL ==== */}
      {showExplain && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100"
          style={{ background: "rgba(0,0,0,0.35)", zIndex: 1100 }}
          onClick={() => setShowExplain(false)}
        >
          <div
            className="card shadow p-3"
            style={{
              maxWidth: 640,
              width: "94%",
              margin: "6rem auto",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h5 className="m-0">Giải thích</h5>
              <div className="d-flex align-items-center gap-2">
                {/* Chọn language (tuỳ bạn, mặc định 'vi') */}
                <select
                  className="form-select form-select-sm"
                  style={{ width: 120 }}
                  value={explainLang}
                  onChange={(e) => setExplainLang(e.target.value)}
                >
                  <option value="vi">Tiếng Việt</option>
                  <option value="ko">한국어</option>
                  <option value="en">English</option>
                </select>

                <button
                  className="btn btn-sm btn-outline-secondary"
                  onClick={() => callExplain("force")}
                  disabled={explainLoading}
                  title="Sinh lại lời giải (ghi đè cache)"
                >
                  {explainLoading ? (
                    <span className="spinner-border spinner-border-sm" />
                  ) : (
                    "Regenerate"
                  )}
                </button>

                <button
                  className="btn btn-sm btn-light"
                  onClick={() => setShowExplain(false)}
                >
                  Đóng
                </button>
              </div>
            </div>

            {explainError && (
              <div className="alert alert-danger py-2">{explainError}</div>
            )}

            {explainLoading && (
              <div className="text-center py-4">
                <div className="spinner-border text-primary" role="status" />
              </div>
            )}

            {!explainLoading && !explainError && explainData && (
              <>
                {/* SINGLE */}
                {!isGroup ? (
                  <div>
                    <div className="small text-muted mb-2">
                      {typeof explainData.is_correct === "boolean" && (
                        <span
                          className={`badge me-2 ${
                            explainData.is_correct ? "bg-success" : "bg-danger"
                          }`}
                        >
                          {explainData.is_correct ? "Đúng" : "Sai"}
                        </span>
                      )}
                      {explainData.updated === false ? (
                        <span className="badge bg-secondary me-2">cached</span>
                      ) : (
                        <span className="badge bg-info text-dark me-2">
                          generated
                        </span>
                      )}
                      {explainData.explain_model && (
                        <span className="small">
                          model: <code>{explainData.explain_model}</code>
                        </span>
                      )}
                    </div>

                    <div className="mb-2">{explainData.explanation}</div>
                    {Array.isArray(explainData.points) &&
                      explainData.points.length > 0 && (
                        <ul className="mb-0">
                          {explainData.points.map((p, i) => (
                            <li key={i}>{p}</li>
                          ))}
                        </ul>
                      )}
                  </div>
                ) : (
                  // GROUP
                  <div className="d-flex flex-column gap-3">
                    {explainData.map((row, idx) => (
                      <div key={row.qno ?? idx} className="p-2 border rounded">
                        <div className="d-flex align-items-center gap-2 mb-1">
                          <strong>Câu {row.qno}</strong>
                          {typeof row.is_correct === "boolean" && (
                            <span
                              className={`badge ${
                                row.is_correct ? "bg-success" : "bg-danger"
                              }`}
                            >
                              {row.is_correct ? "Đúng" : "Sai"}
                            </span>
                          )}
                          {row.updated ? (
                            <span className="badge bg-info text-dark">
                              generated
                            </span>
                          ) : (
                            <span className="badge bg-secondary">cached</span>
                          )}
                        </div>
                        {row.error ? (
                          <div className="text-danger small">{row.error}</div>
                        ) : (
                          <>
                            <div className="mb-1">{row.explanation}</div>
                            {Array.isArray(row.points) &&
                              row.points.length > 0 && (
                                <ul className="mb-0">
                                  {row.points.map((p, i) => (
                                    <li key={i}>{p}</li>
                                  ))}
                                </ul>
                              )}
                            {row.explain_model && (
                              <div className="small text-muted mt-1">
                                model: <code>{row.explain_model}</code>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PracticePage;
