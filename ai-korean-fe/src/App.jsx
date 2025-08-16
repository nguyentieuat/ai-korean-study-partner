// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import NavBar from './components/NavBar' 
import ConversationPage from './pages/ConversationPage'
import PronunciationPage from './pages/PronunciationPage'
import PracticePage from './pages/PracticePage'
import MaterialsPage from './pages/MaterialsPage'
import CooperatePage from './pages/CooperatePage'

function App() {
  return (
    <BrowserRouter>
      <NavBar />  {/* ⬅️ Đặt bên ngoài để luôn hiển thị */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/conversation" element={<ConversationPage />} />
        <Route path="/pronunciation" element={<PronunciationPage />} />
        <Route path="/practice" element={<PracticePage />} />
        <Route path="/materials" element={<MaterialsPage />} />
        <Route path="/cooperate" element={<CooperatePage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
