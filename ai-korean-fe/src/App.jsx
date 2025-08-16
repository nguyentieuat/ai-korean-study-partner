// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ConversationPage from './pages/ConversationPage'
import NavBar from './components/NavBar' 
import PronunciationPage from './pages/PronunciationPage'

function App() {
  return (
    <BrowserRouter>
      <NavBar />  {/* ⬅️ Đặt bên ngoài để luôn hiển thị */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/conversation" element={<ConversationPage />} />
        <Route path="/pronunciation" element={<PronunciationPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
