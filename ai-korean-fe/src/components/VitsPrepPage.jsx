import React, { useState, useRef } from "react";

const errorTypes = ["tone", "length", "consonant", "vowel", "stress"];

const VitsPrepPage = ({ name, phone }) => {
  const [gender, setGender] = useState("male");
  const [audioFile, setAudioFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [textClean, setTextClean] = useState("");
  const [annotations, setAnnotations] = useState([]);
  const [selectedCharIndex, setSelectedCharIndex] = useState(null);
  const [suggestedFix, setSuggestedFix] = useState("");
  const [errorType, setErrorType] = useState(errorTypes[0]);
  const [comment, setComment] = useState("");
  const [exportedData, setExportedData] = useState(null);

  const [allRecords, setAllRecords] = useState([]);
  const [editingIndex, setEditingIndex] = useState(null);

  const audioRef = useRef();
  const fileInputRef = useRef();

  // click ký tự
  const handleCharClick = (index) => {
    setSelectedCharIndex(index);
    const existing = annotations.find((a) => a.char_index === index);
    if (existing) {
      setSuggestedFix(existing.suggested_fix);
      setErrorType(existing.error_type);
      setComment(existing.comment);
    } else {
      setSuggestedFix("");
      setErrorType(errorTypes[0]);
      setComment("");
    }
  };

  const addAnnotation = () => {
    if (selectedCharIndex === null) return;
    const newAnnotation = {
      char_index: selectedCharIndex,
      char_text: textClean[selectedCharIndex],
      error_type: errorType,
      suggested_fix: suggestedFix,
      comment,
    };
    setAnnotations([
      ...annotations.filter((a) => a.char_index !== selectedCharIndex),
      newAnnotation,
    ]);
    setSelectedCharIndex(null);
  };

  const resetAnnotations = () => setAnnotations([]);

  const startRecording = async () => {
    if (!navigator.mediaDevices) {
      alert("Mic không hỗ trợ trên trình duyệt này.");
      return;
    }

    // reset file input hiển thị
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    // reset file input
    setAudioFile(null);
    setAudioURL("");

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/wav" });
      setAudioFile(blob);
      setAudioURL(URL.createObjectURL(blob));
    };
    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder) mediaRecorder.stop();
    setRecording(false);
  };

  // thêm record mới hoặc update
  const handleAddRecord = () => {
    if (!audioFile || !textClean) {
      alert("Vui lòng upload/record audio và nhập text!");
      return;
    }

    const record = {
      gender,
      audioFile,
      audioURL,
      textClean,
      annotations,
    };

    if (editingIndex !== null) {
      // update record
      setAllRecords((prev) => {
        const copy = [...prev];
        copy[editingIndex] = record;
        return copy;
      });
      setEditingIndex(null);
    } else {
      setAllRecords((prev) => [...prev, record]);
    }

    // reset form
    setAudioFile(null);
    setAudioURL("");
    setTextClean("");
    resetAnnotations();
  };

  const handleEditRecord = (index) => {
    const record = allRecords[index];
    setEditingIndex(index);
    setGender(record.gender);
    setAudioFile(record.audioFile);
    setAudioURL(record.audioURL);
    setTextClean(record.textClean);
    setAnnotations(record.annotations);
  };

  const handleDeleteRecord = (index) => {
    setAllRecords((prev) => prev.filter((_, i) => i !== index));
    if (editingIndex === index) {
      setEditingIndex(null);
      setAudioFile(null);
      setAudioURL("");
      setTextClean("");
      resetAnnotations();
    }
  };

  const handleExportAll = () => {
    const dataJSON = allRecords.map((r) => ({
      gender: r.gender,
      audio_id: r.audioFile?.name
        ? r.audioFile.name.replace(/\.[^/.]+$/, "")
        : "mic_audio_" + Date.now(),
      audio_path: r.audioURL || (r.audioFile?.name ?? ""),
      text_clean: r.textClean,
      text_phoneme: "",
      annotations: r.annotations,
    }));
    setExportedData(dataJSON);
  };

  return (
    <div className="container my-6">
      <h2 className="mb-4">VITS Data Prep</h2>

      {/* Form */}
      <div className="mb-3">
        <label>Gender</label>
        <select
          className="form-select mb-2"
          value={gender}
          onChange={(e) => setGender(e.target.value)}
        >
          <option value="male">Male</option>
          <option value="female">Female</option>
          <option value="other">Other</option>
        </select>

        <label>Audio</label>
        <input
          type="file"
          accept="audio/*"
          className="form-control mb-2"
          ref={fileInputRef}
          onChange={(e) => {
            setAudioFile(e.target.files[0]);
            setAudioURL(URL.createObjectURL(e.target.files[0]));
          }}
        />
        {!recording ? (
          <button className="btn btn-primary me-2" onClick={startRecording}>
            Record via Mic
          </button>
        ) : (
          <button className="btn btn-danger me-2" onClick={stopRecording}>
            Stop Recording
          </button>
        )}

        {audioURL && (
          <div className="d-flex align-items-center mb-2">
            <audio controls src={audioURL} className="me-2" />
            <button
              type="button"
              className="btn btn-sm btn-warning"
              onClick={() => {
                setAudioFile(null);
                setAudioURL("");
              }}
            >
              Clear Audio
            </button>
          </div>
        )}
      </div>

      <div className="mb-3">
        <label>Text Clean</label>
        <input
          type="text"
          className="form-control"
          value={textClean}
          onChange={(e) => setTextClean(e.target.value)}
        />
      </div>

      {/* Char annotation */}
      {(textClean || "").split("").map((char, idx) => {
        const activeChar = selectedCharIndex === idx;
        const existing = annotations.find((a) => a.char_index === idx);
        const isAnnotated = !!existing;
        return (
          <span
            key={idx}
            style={{ position: "relative", margin: "0 2px", cursor: "pointer" }}
          >
            <span
              onClick={() => handleCharClick(idx)}
              style={{
                padding: "2px 4px",
                border: "1px dashed gray",
                backgroundColor: activeChar
                  ? "#e0f7fa"
                  : isAnnotated
                  ? "#c8e6c9"
                  : "transparent",
              }}
            >
              {char}
            </span>

            {activeChar && (
              <div
                style={{
                  position: "absolute",
                  top: "110%",
                  left: 0,
                  background: "#fff",
                  border: "1px solid #ccc",
                  padding: "8px",
                  zIndex: 10,
                  width: 220,
                }}
              >
                <div className="mb-1">
                  <label>Error Type</label>
                  <select
                    className="form-select"
                    value={errorType}
                    onChange={(e) => setErrorType(e.target.value)}
                  >
                    {errorTypes.map((et) => (
                      <option key={et} value={et}>
                        {et}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="mb-1">
                  <label>Suggested Fix</label>
                  <input
                    type="text"
                    className="form-control"
                    value={suggestedFix}
                    onChange={(e) => setSuggestedFix(e.target.value)}
                  />
                </div>
                <div className="mb-1">
                  <label>Comment</label>
                  <input
                    type="text"
                    className="form-control"
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                  />
                </div>
                <div className="d-flex justify-content-end gap-2 mt-1">
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    onClick={addAnnotation}
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={() => setSelectedCharIndex(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    onClick={() => {
                      setAnnotations((prev) =>
                        prev.filter((a) => a.char_index !== idx)
                      );
                      setSelectedCharIndex(null);
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </span>
        );
      })}

      {/* Add/Update record */}
      <div className="my-3">
        <button className="btn btn-info me-2" onClick={handleAddRecord}>
          {editingIndex !== null ? "Update Record" : "Add Record"}
        </button>
        <button className="btn btn-success" onClick={handleExportAll}>
          Submit All
        </button>
      </div>

      {/* Records table */}
      {allRecords.length > 0 && (
        <div className="mt-4">
          <h5>Added Records (click to edit)</h5>
          <table className="table table-bordered">
            <thead>
              <tr>
                <th>#</th>
                <th>Gender</th>
                <th>Audio</th>
                <th>Text Clean</th>
                <th>Annotations</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {allRecords.map((r, i) => (
                <tr key={i}>
                  <td onClick={() => handleEditRecord(i)}>{i + 1}</td>
                  <td onClick={() => handleEditRecord(i)}>{r.gender}</td>
                  <td onClick={() => handleEditRecord(i)}>
                    {r.audioFile?.name || "mic_audio"}
                  </td>
                  <td onClick={() => handleEditRecord(i)}>{r.textClean}</td>
                  <td onClick={() => handleEditRecord(i)}>
                    {r.annotations.length}
                  </td>
                  <td>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => handleDeleteRecord(i)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Exported JSON */}
      {exportedData && (
        <div className="mt-3">
          <h5>Exported JSON</h5>
          <pre className="p-3 bg-light border">
            {JSON.stringify(exportedData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export default VitsPrepPage;
