import React, { useState, useEffect } from "react";
import "../assets/css/footer.css";
import qrBank from "../assets/images/qr.png";

const Footer = () => {
  const [open, setOpen] = useState(false);
  const openModal = () => setOpen(true);
  const closeModal = () => setOpen(false);

  // Đóng bằng phím ESC
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") closeModal();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <footer className="app-footer">
      <div className="footer-inner">
        <small className="copyright">Develop by nguyentieuat</small>

        {/* <div className="donate" aria-label="Donate">
          <span className="donate-text">Ủng hộ dựng server:</span>
          <img
            src={qrBank}
            alt="QR ngân hàng ủng hộ"
            className="donate-qr"
            loading="lazy"
            onClick={openModal}
            title="Nhấn để phóng to"
            role="button"
          />
        </div> */}

        <div />
      </div>

      {/* Modal phóng to QR */}
      <div
        className={`qr-modal ${open ? "open" : ""}`}
        onClick={closeModal}
        role="dialog"
        aria-modal="true"
        aria-label="QR ngân hàng"
      >
        <div className="qr-box" onClick={(e) => e.stopPropagation()}>
          <img
            src={qrBank}
            alt="QR ngân hàng ủng hộ - phóng to"
            className="qr-modal-img"
          />
          {/* <button className="qr-close" onClick={closeModal} aria-label="Đóng">
            ×
          </button> */}
        </div>
      </div>
    </footer>
  );
};

export default Footer;
