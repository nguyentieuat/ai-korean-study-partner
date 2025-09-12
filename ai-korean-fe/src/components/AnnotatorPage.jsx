import React, { useState, useEffect } from "react";
const backendUrl = import.meta.env.VITE_API_URL;

const errorTypes = ["tone", "length", "consonant", "vowel", "stress", "other"];

const AnnotatorPage = ({ userInfo }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [data, setData] = useState(null);
  const [dataSet, setDataSet] = useState(null);
  const [textHeard, setTextHeard] = useState("");
  const [activeChar, setActiveChar] = useState(null);
  const [formState, setFormState] = useState({
    error_type: "",
    suggested_fix: "",
    comment: "",
  });
  const [tempAnnotations, setTempAnnotations] = useState([]);
  const [allAnnotations, setAllAnnotations] = useState([]);
  const [submittedData, setSubmittedData] = useState(null);
  const [editingIndex, setEditingIndex] = useState(null); // index row đang edit
  const [prevIndex, setPrevIndex] = useState(null);
  const [loading, setLoading] = useState(false);

  // fetch dataset từ API
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(
          `${backendUrl}/cooperate_annotator?offset=0&limit=10`
        );
        const result = await res.json();
        
        if (result.data && result.data.length > 0) {
          setDataSet(result.data);
          setData(result.data[0]);
        }
      } catch (err) {
        console.error("❌ Lỗi fetch dataset:", err);
      }
    };
    fetchData();
  }, []);

  // click ký tự để annotate
  const handleCharClick = (idx) => {
    setActiveChar(idx);
    const existing = tempAnnotations.find((a) => a.char_index === idx);
    setFormState(
      existing
        ? { ...existing }
        : { error_type: "", suggested_fix: "", comment: "" }
    );
  };

  // save annotation cho character
  const handleSave = () => {
    if (!formState.error_type || !formState.suggested_fix) {
      alert("Chưa nhập đủ thông tin!");
      return;
    }
    const newAnnotation = {
      char_index: activeChar,
      char_text: data.text[activeChar],
      error_type: formState.error_type,
      suggested_fix: formState.suggested_fix,
      comment: formState.comment || "",
    };

    const newTemp = [
      ...tempAnnotations.filter((a) => a.char_index !== activeChar),
      newAnnotation,
    ];
    setTempAnnotations(newTemp);
    setActiveChar(null);
  };

  const handleCancel = () => setActiveChar(null);

  const handleDelete = () => {
    const newTemp = tempAnnotations.filter((a) => a.char_index !== activeChar);
    setTempAnnotations(newTemp);
    setActiveChar(null);
  };

  const isAnnotated = (idx) =>
    tempAnnotations.some((a) => a.char_index === idx);

  // edit từ table
  const handleEditFromTable = (item, idx) => {
    setPrevIndex(currentIndex); // lưu lại vị trí hiện tại trước khi edit
    setEditingIndex(idx);
    setData({
      audio_id: item.audio_id,
      audio_path: item.audio_path,
      text_clean: item.text_clean,
    });
    setTextHeard(item.text_heard);
    setTempAnnotations(item.annotations);
    setActiveChar(null);
    setFormState({ error_type: "", suggested_fix: "", comment: "" });
  };

  const handleUpdate = () => {
    if (editingIndex === null) return;

    setAllAnnotations((prev) => {
      const copy = [...prev];
      copy[editingIndex] = {
        ...copy[editingIndex],
        text_heard: textHeard,
        annotations: tempAnnotations,
      };
      return copy;
    });

    // reset edit mode
    setEditingIndex(null);
    resetForm();

    // khôi phục index trước khi edit
    if (prevIndex !== null) {
      setCurrentIndex(prevIndex);
      setData(dataSet[prevIndex]);
      setPrevIndex(null);
    }
  };

  // next dataset
  const handleNextData = () => {
    // nếu đang annotate bình thường, lưu tempAnnotations
    if (tempAnnotations.length > 0 && textHeard && editingIndex === null) {
      setAllAnnotations((prev) => [
        ...prev,
        {
          audio_id: data.audio_id,
          audio_path: data.audio_path,
          text_clean: data.text_clean,
          text_heard: textHeard,
          annotations: tempAnnotations,
        },
      ]);
    }

    // next dataset
    const nextIndex = currentIndex + 1;
    if (nextIndex >= dataSet.length) return;
    setCurrentIndex(nextIndex);
    setData(dataSet[nextIndex]);
    resetForm();
  };
  const resetForm = () => {
    setTextHeard("");
    setTempAnnotations([]);
    setFormState({ error_type: "", suggested_fix: "", comment: "" });
    setActiveChar(null);
  };

  const handleSubmitAll = async () => {
    if (!userInfo.phone.trim()) {
      alert("Số điện thoại là bắt buộc!");
      return;
    }

    if (!textHeard.trim()) {
      alert("Text Heard là bắt buộc!");
      return;
    }

    const finalData = [
      ...allAnnotations,
      ...(tempAnnotations.length > 0
        ? [
            {
              audio_id: data.audio_id,
              audio_path: data.audio_path,
              text_clean: data.text_clean,
              text_heard: textHeard,
              annotations: tempAnnotations,
            },
          ]
        : []),
    ];

    if (finalData.length === 0) {
      alert("Chưa có annotation nào để gửi!");
      return;
    }

    const payload = {
      annotator_name: userInfo.name,
      annotator_phone: userInfo.phone,
      annotations: finalData,
    };
    setLoading(true); // bật loading
    try {
      const response = await fetch(`${backendUrl}/cooperate_annotator`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Gửi dữ liệu thất bại!");
      }

      const result = await response.json();
      console.log("✅ Server trả về:", result);

      setSubmittedData(finalData);
      alert("Đã submit tất cả annotations thành công!");
      resetForm();
    } catch (error) {
      console.error("❌ Lỗi khi gửi:", error);
      alert("Có lỗi khi gửi annotations!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h2>Annotator Page</h2>

      <audio
        controls
        src={`${data?.audio_data_url}`}
        className="mb-3"
      />

      <div className="mb-3">
        <label>Original Text (Clean):</label>
        <div
          style={{
            padding: 8,
            background: "#f5f5f5",
            border: "1px solid #ccc",
          }}
        >
          {data?.text}
        </div>
      </div>

      <div className="mb-3">
        <label>Text Heard:</label>
        <input
          type="text"
          className="form-control"
          value={textHeard}
          onChange={(e) => setTextHeard(e.target.value)}
          placeholder="Nhập bạn nghe được..."
        />
      </div>

      <div className="mb-3">
        {data?.text.split("").map((char, idx) => (
          <span
            key={idx}
            style={{ position: "relative", margin: "0 2px", cursor: "pointer" }}
          >
            <span
              onClick={() => handleCharClick(idx)}
              style={{
                padding: "2px 4px",
                border: "1px dashed gray",
                backgroundColor:
                  activeChar === idx
                    ? "#e0f7fa"
                    : isAnnotated(idx)
                    ? "#c8e6c9"
                    : "transparent",
              }}
            >
              {char}
            </span>

            {activeChar === idx && (
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
                  boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
                }}
              >
                <div className="mb-1">
                  <label>Error Type:</label>
                  <select
                    className="form-select"
                    value={formState.error_type}
                    onChange={(e) =>
                      setFormState({ ...formState, error_type: e.target.value })
                    }
                  >
                    <option value="">Chọn lỗi</option>
                    {errorTypes.map((et) => (
                      <option key={et} value={et}>
                        {et}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="mb-1">
                  <label>Suggested Fix:</label>
                  <input
                    type="text"
                    className="form-control"
                    value={formState.suggested_fix}
                    onChange={(e) =>
                      setFormState({
                        ...formState,
                        suggested_fix: e.target.value,
                      })
                    }
                  />
                </div>
                <div className="mb-1">
                  <label>Comment (optional):</label>
                  <input
                    type="text"
                    className="form-control"
                    value={formState.comment}
                    onChange={(e) =>
                      setFormState({ ...formState, comment: e.target.value })
                    }
                  />
                </div>
                <div className="d-flex justify-content-end gap-2 mt-1">
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={handleSave}
                  >
                    Save
                  </button>
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={handleCancel}
                  >
                    Cancel
                  </button>
                  <button
                    className="btn btn-sm btn-danger"
                    onClick={handleDelete}
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </span>
        ))}
      </div>

      <div className="mb-3 d-flex gap-2">
        {editingIndex !== null ? (
          <button className="btn btn-warning" onClick={handleUpdate}>
            Update
          </button>
        ) : (
          <button
            className="btn btn-info"
            onClick={handleNextData}
            disabled={currentIndex >= dataSet?.length - 1}
          >
            + Next Data
          </button>
        )}
        <button
          className="btn btn-success"
          onClick={handleSubmitAll}
          disabled={loading}
        >
          {loading ? (
            <>
              Đang gửi...
              <span
                style={{
                  marginLeft: 8,
                  width: 16,
                  height: 16,
                  border: "2px solid #fff",
                  borderTop: "2px solid transparent",
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "spin 1s linear infinite",
                }}
              />
            </>
          ) : (
            "Submit All Annotations"
          )}
        </button>
      </div>

      {allAnnotations.filter((item) => item.annotations.length > 0).length >
        0 && (
        <div className="mb-3">
          <h4>Temporary Table (Pending Submit) - click row to edit</h4>
          <table className="table table-bordered">
            <thead>
              <tr>
                <th>#</th>
                <th>Audio ID</th>
                <th>Original Text</th>
                <th>Text Heard</th>
                <th>Annotations</th>
              </tr>
            </thead>
            <tbody>
              {allAnnotations
                .filter((item) => item.annotations.length > 0)
                .map((item, idx) => (
                  <tr
                    key={idx}
                    onClick={() => handleEditFromTable(item, idx)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>{idx + 1}</td>
                    <td>{item.audio_id}</td>
                    <td>{item.text_clean}</td>
                    <td>{item.text_heard}</td>
                    <td>{item.annotations.length}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {submittedData && (
        <div>
          <h4>Submitted Data:</h4>
          <pre>{JSON.stringify(submittedData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default AnnotatorPage;
