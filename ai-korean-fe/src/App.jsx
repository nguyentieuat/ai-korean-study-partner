// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import NavBar from './components/NavBar' 
import ConversationPage from './pages/ConversationPage'
import PronunciationPage from './pages/PronunciationPage'
import PracticePage from './pages/PracticePage'
import MaterialsPage from './pages/MaterialsPage'
import CooperatePage from './pages/CooperatePage'
import Footer from './components/Footer'
import ConsentGate from './components/ConsentGate'

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">    {/* <- flex column + min-height:100vh */}
        <ConsentGate />
        <NavBar />
        <main className="app-content">   {/* <- vùng nội dung sẽ “đẩy” footer xuống */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/conversation" element={<ConversationPage />} />
            <Route path="/pronunciation" element={<PronunciationPage />} />
            <Route path="/practice" element={<PracticePage />} />
            <Route path="/materials" element={<MaterialsPage />} />
            <Route path="/cooperate" element={<CooperatePage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}

export default App
