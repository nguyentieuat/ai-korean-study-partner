import React from 'react';

const ChatBubble = ({ role, content, corrections = [], audio }) => {
  const highlightedContent = corrections.reduce((acc, correction) => {
    const { wrong, correct } = correction;
    const regex = new RegExp(`(${wrong})`, 'gi');
    return acc.replace(regex, `<span class="text-danger fw-bold">$1</span>`);
  }, content);

  return (
    <div className={`mb-2 p-2 rounded ${role === 'user' ? 'bg-light' : 'bg-info text-white'}`}>
      {audio && (
        <audio controls className="mb-1">
          <source src={audio} type="audio/webm" />
          Trình duyệt không hỗ trợ audio
        </audio>
      )}
      <div dangerouslySetInnerHTML={{ __html: highlightedContent }} />
    </div>
  );
};

export default ChatBubble;
