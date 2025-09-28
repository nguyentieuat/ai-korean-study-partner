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

  // ---------- Helpers ----------
  const isGroup =
    Array.isArray(questionContent?.items) && questionContent.items.length > 0;

  const groupQnos = useMemo(() => {
    if (Array.isArray(questionContent?.question_no)) {
      return new Set(questionContent.question_no);
    }
    return new Set();
  }, [questionContent]);

  const itemsCount = isGroup ? questionContent.items.length : 1;
  const computedTimeLimit = 120 * itemsCount;

  const formatTime = (s) => {
    const total = s == null ? computedTimeLimit : s;
    const mm = String(Math.floor(total / 60)).padStart(2, "0");
    const ss = String(total % 60).padStart(2, "0");
    return `${mm}:${ss}`;
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

  // ==== NGHE: bắt đầu timer khi audio được phát lần đầu ====
  const handleAudioPlay = () => {
    if (!startTimeRef.current) startTimeRef.current = performance.now();
    audioPlayCountRef.current += 1;

    if (allItemsAnswered || timeUp) return;

    if (!sequenceActive && !audioLocked) {
      setAudioLocked(true);
      setSequenceActive(true);
      setReplaysRemaining(1);
      startCountdown(computedTimeLimit);
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
    const isUnderDevelopment =
      (level === "topik1" &&
        type === "reading" &&
        (q === 40 || q === 41 || q === 42 || q > 48)) ||
      level === "topik2" ||
      (level === "topik1" && type === "listening" && (q === 15 || q === 16));

    if (isUnderDevelopment) {
      setQuestionContent({
        title: "Thông báo",
        message: "Đang phát triển, vui lòng quay lại sau",
      });
      return;
    }

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

      // (trước đây gọi startCountdown ở đây — có thể bị race. Đã chuyển sang useEffect bên dưới)
    } catch (err) {
      console.error("Error fetching question:", err);
      setQuestionContent({ error: "Không thể lấy câu hỏi" });
    } finally {
      setLoadingQuestion(false);
    }
  };

  // ==== ĐỌC: đảm bảo bật timer NGAY khi dữ liệu đọc đã sẵn sàng ====
  useEffect(() => {
    if (type !== "reading") return;
    if (!questionContent || questionContent.error) return;
    if (remainingSec !== null || timeUp) return; // đã chạy rồi

    // bắt đầu đo thời gian phản hồi + countdown
    startTimeRef.current = performance.now();
    const items = Array.isArray(questionContent?.items)
      ? questionContent.items.length
      : 1;
    const secs = 120 * items;
    startCountdown(secs);
  }, [type, questionContent, remainingSec, timeUp]);

  const buildFeedbackPayload = (reaction, extra = {}) => {
    const isGroup =
      Array.isArray(questionContent?.items) &&
      questionContent.items.length > 0;

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
      !isGroup && selectedAnswer != null && questionContent?.answer != null
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
      answer_selected: isGroup ? selectedAnswers : selectedAnswer ?? null,
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
                  style={{ width: "40px", height: "40px", position: "relative" }}
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
            <div className="d-flex align-items-center justify-content-center position-relative mb-3">
              <h5 className="card-title text-center m-0">
                {level.toUpperCase()} -{" "}
                {type === "reading" ? "reading" : "listening"} - Câu{" "}
                {selectedQuestion}
              </h5>
              {/* Hiển thị countdown cho cả đọc & nghe */}
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
                  Bài gồm {itemsCount} câu:{" "}
                  {Array.isArray(questionContent?.question_no)
                    ? questionContent.question_no.join(", ")
                    : ""}
                </p>
              </div>
            )}

            {/* PASSAGE (ĐỌC) */}
            {type === "reading" && questionContent?.passage && (
              <div
                className="mb-3 p-3 bg-light border rounded"
                style={{ whiteSpace: "pre-wrap" }}
              >
                {questionContent.passage}
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
                      <p key={idx} className="mb-1">
                        <strong>{d.speaker}:</strong> {d.text}
                      </p>
                    ))}
                  </div>
                )}

              {/* Choices */}
              {!isGroup && selectedQuestion && questionContent?.choices && (
                <div className="d-flex flex-column gap-2 mt-3">
                  {Object.entries(questionContent.choices).map(([key, text]) => (
                    <button
                      key={key}
                      className={`btn ${choiceVariantSingle(key)}`}
                      onClick={() => onChooseAnswer(key)}
                      disabled={disableChoicesSingle}
                    >
                      {key}. {text}
                    </button>
                  ))}
                </div>
              )}

              {/* Group items */}
              {isGroup && (
                <div className="d-flex flex-column gap-3 mt-2">
                  {questionContent.items.map((it, idx) => {
                    const qno = it.question_no;
                    return (
                      <div key={qno ?? idx} className="p-2 border rounded">
                        <div className="fw-semibold mb-2">
                          {qno ? `Câu ${qno}. ` : ""}
                          {it.question}
                        </div>
                        <div className="d-flex flex-column gap-2">
                          {Object.entries(it.choices || {}).map(
                            ([key, text]) => (
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
                              >
                                {key}. {text}
                              </button>
                            )
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {(allItemsAnswered || timeUp) && (
              <div className="d-flex justify-content-center mt-3">
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
    </div>
  );
};

export default PracticePage;
