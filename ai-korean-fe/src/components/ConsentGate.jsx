import React, { useEffect, useState } from "react";
import ConsentModal from "./ConsentModal";

const STORAGE_KEY = "ai_korean_privacy_consent"; // đổi tên nếu cần

export default function ConsentGate() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (!saved) setOpen(true);         // chỉ lần đầu
    } catch {
      setOpen(true);                     // fallback nếu localStorage bị chặn
    }
  }, []);

  const accept = (prefs) => {
    const payload = {
      status: "accepted",
      prefs,
      ts: new Date().toISOString(),
      ver: 1,
    };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch {}
    setOpen(false);

    // TODO: bật SDK thực tế nếu dùng (GA4, PostHog, etc.)
    // if (prefs.analytics) enableAnalytics();
    // if (prefs.personalization) enablePersonalization();
  };

  const decline = () => {
    const payload = { status: "declined", ts: new Date().toISOString(), ver: 1 };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch {}
    setOpen(false);

    // TODO: đảm bảo tắt/bỏ tải mọi script theo dõi
    // disableAnalytics();
    // disablePersonalization();
  };

  return <ConsentModal open={open} onAccept={accept} onDecline={decline} />;
}
