import React, { useState } from 'react';
import axios from 'axios';

const ChatBox = () => {
  const [input, setInput] = useState('');
  const [conversation, setConversation] = useState([]); // [{role: 'user'|'bot', message: '...'}]

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Gửi lên server: chỉ lấy 5 lượt gần nhất
    const lastFive = conversation.slice(-5);

    try {
      const response = await axios.post('http://localhost:5000/chat', {
        message: input,
        conversation_history: lastFive
      });

      const { original, corrected, reply } = response.data;

      setConversation(prev => [
        ...prev,
        { role: 'user', message: original },
        { role: 'bot', message: reply }
      ]);

      setInput('');
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  return (
    <div>
      <div style={{ maxHeight: '400px', overflowY: 'scroll' }}>
        {conversation.map((msg, idx) => (
          <div key={idx} style={{ textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <strong>{msg.role === 'user' ? '🧑‍💬' : '🤖'} </strong>{msg.message}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && sendMessage()}
      />
      <button onClick={sendMessage}>Gửi</button>
    </div>
  );
};

export default ChatBox;
