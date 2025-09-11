import { useState } from 'react';

function InputBar({ onSend }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSend(text);
      setText('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="d-flex gap-2">
      <input
        type="text"
        className="form-control"
        placeholder="Nhập tiếng Hàn..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button className="btn btn-success" type="submit">
        Gửi
      </button>
    </form>
  );
}

export default InputBar;
