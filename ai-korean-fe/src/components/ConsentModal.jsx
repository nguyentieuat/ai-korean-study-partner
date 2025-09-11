import React, { useEffect, useRef, useState } from "react";
import "../assets/css/consent.css";

export default function ConsentModal({ open, onAccept, onDecline }) {
  const dialogRef = useRef(null);
  const [analytics, setAnalytics] = useState(true);
  const [personalization, setPersonalization] = useState(true);

  // Khóa scroll nền khi mở
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  // Đóng bằng ESC
  useEffect(() => {
    const onKey = (e) => { if (open && e.key === "Escape") onDecline?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onDecline]);

  if (!open) return null;

  return (
    <div className="consent-overlay" role="dialog" aria-modal="true" aria-labelledby="consent-title">
      <div className="consent-modal" ref={dialogRef} onClick={(e)=>e.stopPropagation()}>
        <h2 id="consent-title" className="consent-title">Cho phép thu thập dữ liệu?</h2>
        <p className="consent-desc">
          Chúng tôi dùng dữ liệu để cải thiện trải nghiệm học, nâng cao chất lượng sản phẩm.
          {/*Bạn có thể xem <a href="/privacy" className="consent-link" target="_blank" rel="noreferrer"> Chính sách quyền riêng tư</a>. */}
        </p>

        {/* <div className="consent-options">
          <label className="consent-check">
            <input type="checkbox" checked={analytics} onChange={()=>setAnalytics(v=>!v)} />
            <span>Ghi nhận số liệu sử dụng (Analytics)</span>
          </label>
          <label className="consent-check">
            <input type="checkbox" checked={personalization} onChange={()=>setPersonalization(v=>!v)} />
            <span>Cá nhân hoá nội dung & gợi ý</span>
          </label>
        </div> */}

        <div className="consent-actions">
          <button
            className="btn btn-primary"
            onClick={() => onAccept({ analytics, personalization })}
          >
            Đồng ý
          </button>
          {/* <button className="btn btn-ghost" onClick={onDecline}>
            Từ chối
          </button> */}
        </div>
      </div>
    </div>
  );
}
