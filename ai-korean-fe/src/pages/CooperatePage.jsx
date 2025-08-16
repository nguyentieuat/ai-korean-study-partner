import React, { useState } from "react";
import AnnotatorPage from "../components/AnnotatorPage";
import VitsPrepPage from "../components/VitsPrepPage";
import TopikAnnotationForm from "../components/TopikAnnotationForm";

const CooperatePage = () => {
  const [activeTab, setActiveTab] = useState("annotator"); // 'annotator' | 'vits' | 'topik'
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");

  // Callback submit từ component con
  const handleSubmitFromChild = (childData) => {
    if (!phone.trim()) {
      alert("Số điện thoại là bắt buộc!");
      return;
    }

    const payload = { ...childData, name, phone };
    console.log("Submit payload:", payload);

    // TODO: gửi payload về server
    // axios.post("/api/submit", payload)...
  };

  return (
    <div className="p-4">
      {/* Input thông tin người dùng */}
      <div className="mb-4 d-flex gap-2">
        <input
          type="text"
          placeholder="Tên (tùy chọn)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="form-control"
        />
        <input
          type="text"
          placeholder="Số điện thoại *"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          className="form-control"
        />
      </div>

      {/* Tabs chọn component */}
      <div className="mb-4 d-flex flex-wrap gap-2">
        <button
          className={`btn ${
            activeTab === "annotator" ? "btn-primary" : "btn-outline-primary"
          } flex-grow-1`}
          onClick={() => setActiveTab("annotator")}
        >
          Đánh nhãn audio
        </button>
        <button
          className={`btn ${
            activeTab === "vits" ? "btn-primary" : "btn-outline-primary"
          } flex-grow-1`}
          onClick={() => setActiveTab("vits")}
        >
          Chuẩn bị train VITS
        </button>
        <button
          className={`btn ${
            activeTab === "topik" ? "btn-primary" : "btn-outline-primary"
          } flex-grow-1`}
          onClick={() => setActiveTab("topik")}
        >
          Nhập câu hỏi TOPIK
        </button>
      </div>

      <div className="tab-content">
        {activeTab === "annotator" && (
          <AnnotatorPage
            userInfo={{ name, phone }}
            onSubmit={handleSubmitFromChild}
          />
        )}
        {activeTab === "vits" && (
          <VitsPrepPage
            userInfo={{ name, phone }}
            onSubmit={handleSubmitFromChild}
          />
        )}
        {activeTab === "topik" && (
          <TopikAnnotationForm
            userInfo={{ name, phone }}
            onSubmit={handleSubmitFromChild}
          />
        )}
      </div>
    </div>
  );
};

export default CooperatePage;
